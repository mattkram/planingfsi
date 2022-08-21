from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Literal
from typing import Type

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fmin

from planingfsi import logger
from planingfsi import math_helpers
from planingfsi import trig
from planingfsi import writers
from planingfsi.config import NUM_DIM
from planingfsi.fe import felib as fe
from planingfsi.fe.femesh import Subcomponent
from planingfsi.solver import fzero

if TYPE_CHECKING:
    from planingfsi.config import Config
    from planingfsi.fe.rigid_body import RigidBody
    from planingfsi.fe.structure import StructuralSolver
    from planingfsi.potentialflow.pressurepatch import PlaningSurface
    from planingfsi.potentialflow.pressurepatch import PressureCushion


ElementType = ClassVar[Type[fe.Element]]


@dataclass
class SubstructureLoads:
    """The integrated global loads on the substructure."""

    D: float = 0.0
    L: float = 0.0
    M: float = 0.0
    Dt: float = 0.0
    Lt: float = 0.0
    Mt: float = 0.0
    Da: float = 0.0
    La: float = 0.0
    Ma: float = 0.0


class Substructure(abc.ABC):
    """Base class for a Substructure, which is a component of a RigidBody.

    Attributes:
        name: A name for the substructure.
        seal_pressure: The internal pressure (steady).
        seal_pressure_method: Determines whether to apply a constant or linear hydrostatic internal pressure.
        seal_over_pressure_pct: A factor to apply to the seal_pressure to get the total seal pressure.
        cushion_pressure: If set, this will be the constant external pressure (i.e. for a wetdeck).
        cushion_pressure_type: If there is not interpolator and `cushion_pressure_type` is "Total", then
            a constant external pressure will be applied to the substructure.
        struct_interp_type: The method to use for interpolation of position.
        struct_extrap: If True, extrapolate the position when the arclength extends past the end nodes.
        parent: A reference to the parent `RigidBody` instance.

    """

    is_free = False
    _element_type: ElementType

    def __init__(
        self,
        *,
        name: str = "",
        seal_pressure: float = 0.0,
        seal_pressure_method: Literal["constant"] | Literal["hydrostatic"] = "constant",
        seal_over_pressure_pct: float = 1.0,
        cushion_pressure_type: Literal["Total"] | None = None,
        struct_interp_type: Literal["linear"] | Literal["quadratic"] | Literal["cubic"] = "linear",
        struct_extrap: bool = True,
        solver: StructuralSolver | None = None,
        parent: RigidBody | None = None,
        **_: Any,
    ):
        self.name = name
        self._interpolator: Interpolator | None = None

        self.seal_pressure = seal_pressure
        self.seal_pressure_method = seal_pressure_method
        self.seal_over_pressure_pct = seal_over_pressure_pct

        self.cushion_pressure: float | None = None
        self.cushion_pressure_type = cushion_pressure_type

        self.struct_interp_type = struct_interp_type
        self.struct_extrap = struct_extrap

        self._solver = solver
        self.parent = parent

        # Arrays to store air and fluid pressure profiles for plotting
        self.fluidS: np.ndarray | None = None
        self.fluidP: np.ndarray | None = None
        self.airS: np.ndarray | None = None
        self.airP: np.ndarray | None = None

        self.elements: list[fe.Element] = []
        self.node_arc_length: np.ndarray | None = None

        self.loads = SubstructureLoads()

        self._interp_coords_at_arclength: interp1d | None = None

    @property
    def solver(self) -> StructuralSolver:
        """A reference to the structural solver. Can be explicitly set, or else traverses the parents."""
        if self._solver is None and self.parent is not None and self.parent.parent is not None:
            return self.parent.parent
        if self._solver is None:
            raise AttributeError("solver must be set before use.")
        return self._solver

    @solver.setter
    def solver(self, solver: StructuralSolver) -> None:
        self._solver = solver

    @property
    def config(self) -> Config:
        """A reference to the simulation configuration."""
        return self.solver.config

    @property
    def ramp(self) -> float:
        """The ramping coefficient from the high-level simulation object."""
        if self.parent is None:
            logger.warning("No parent assigned, ramp will be set to 1.0.")
            return 1.0
        return self.solver.simulation.ramp

    @property
    def nodes(self) -> list[fe.Node]:
        """A list of all nodes in the substructure."""
        nodes = [el.start_node for el in self.elements]
        return nodes + [self.elements[-1].end_node]

    @property
    def arc_length(self) -> float:
        """The total arc length of the substructure, i.e. the sum of all of the Element lengths."""
        return max(self.node_arc_length)

    @property
    def it_dir(self) -> Path:
        """A path to the current numbered iteration directory, in which to save results."""
        return self.solver.simulation.it_dir

    @property
    def coordinates_file_path(self) -> Path:
        """A path to the coordinates file for the current iteration."""
        return self.it_dir / f"coords_{self.name}.{self.config.io.data_format}"

    def add_planing_surface(self, planing_surface: PlaningSurface, **kwargs: Any) -> PlaningSurface:
        """Add a planing surface to the substructure, and configure the interpolator.

        Args:
            planing_surface: The planing surface.
            **kwargs: Keyword arguments to pass through to the Interpolator.

        Returns:
            The same planing surface passed in as an argument (useful for chaining).

        """
        # Assign the same interpolator to both the substructure and planing_surface
        planing_surface.interpolator = self._interpolator = Interpolator(
            self, planing_surface, **kwargs
        )
        self.solver.simulation.fluid_solver.add_planing_surface(planing_surface)
        return planing_surface

    def add_pressure_cushion(self, pressure_cushion: PressureCushion) -> PressureCushion:
        """Add a constant pressure cushion to the substructure. The cushion acts as a constant
        external pressure applied to the surface, which is integrated into a normal force.

        Args:
            pressure_cushion: The planing surface.

        Returns:
            The same pressure cushion passed in as an argument (useful for chaining).

        """
        self.solver.simulation.fluid_solver.add_pressure_cushion(pressure_cushion)
        self.cushion_pressure_type = "Total"
        self.cushion_pressure = pressure_cushion.cushion_pressure
        return pressure_cushion

    def load_mesh(self, submesh: Path | Subcomponent = Path("mesh")) -> None:
        """Load the mesh from an object or file, and create all component Elements."""
        if isinstance(submesh, Subcomponent):
            nd_idx = [
                (line.start_point.index, line.end_point.index) for line in submesh.line_segments
            ]
        else:
            nd_idx = np.loadtxt(submesh / f"elements_{self.name}.txt", dtype=int)

        self.elements = [
            self._element_type(self.solver.nodes[nd_st_i], self.solver.nodes[nd_end_i], parent=self)
            for nd_st_i, nd_end_i in nd_idx
        ]
        self.update_geometry()

        # Set the interpolation method if there are not enough elements (cubic requires at least 3 elements)
        # else keep what was specified in the constructor
        if len(self.elements) == 1:
            self.struct_interp_type = "linear"
        elif len(self.elements) == 2 and not self.struct_interp_type == "linear":
            self.struct_interp_type = "quadratic"

    def update_geometry(self) -> None:
        """Update geometry and interpolation functions in the process."""
        element_lengths = [el.length for el in self.elements]
        self.node_arc_length = np.cumsum([0.0] + element_lengths)

        nodal_coordinates = np.array([nd.coordinates for nd in self.nodes])
        self._interp_coords_at_arclength = interp1d(
            self.node_arc_length,
            nodal_coordinates.T,
            kind=self.struct_interp_type,
            fill_value="extrapolate" if self.struct_extrap else np.nan,
        )

    def get_coordinates(self, si: float) -> np.ndarray:
        """Return the coordinates of the surface at a specific arclength."""
        return self._interp_coords_at_arclength(si)

    def write_coordinates(self) -> None:
        """Write the coordinates to file"""
        writers.write_as_list(
            self.coordinates_file_path,
            ["x [m]", [nd.x for nd in self.nodes]],
            ["y [m]", [nd.y for nd in self.nodes]],
        )

    def load_coordinates(self) -> None:
        """Set the coordinates of each node by loading from the saved file from the current iteration."""
        coordinates = np.loadtxt(self.coordinates_file_path)
        for coords, nd in zip(coordinates, self.nodes):
            nd.coordinates = coords
        self.update_geometry()

    def _get_loads_in_range(self, s_start, s_end, /):
        if self._interpolator is not None:
            return self._interpolator.get_loads_in_range(
                s_start,
                s_end,
                pressure_limit=(
                    self.config.flow.stagnation_pressure
                    if self.config.plotting.pressure_limiter
                    else None
                ),
            )
        else:
            s = np.array([s_start, s_end])
            return s, np.zeros_like(s), np.zeros_like(s)

    def update_fluid_forces(self) -> None:
        # TODO: Refactor this complex, critical method
        fluid_s: list[float] = []
        fluid_p: list[float] = []
        air_s: list[float] = []
        air_p: list[float] = []
        self.loads.D = 0.0
        self.loads.L = 0.0
        self.loads.M = 0.0
        self.loads.Dt = 0.0
        self.loads.Lt = 0.0
        self.loads.Mt = 0.0
        self.loads.Da = 0.0
        self.loads.La = 0.0
        self.loads.Ma = 0.0
        if self._interpolator is not None:
            s_min, s_max = self._interpolator.get_min_max_s()

        for i, el in enumerate(self.elements):
            # Get pressure & shear stress at end points and all fluid points along element
            s_start, s_end = self.node_arc_length[i], self.node_arc_length[i + 1]
            s, pressure_fluid, tau = self._get_loads_in_range(s_start, s_end)

            # TODO: Remove after refactor complete
            assert abs(s_start - s[0]) < 1e-12
            assert abs(s_end - s[-1]) < 1e-12

            # Apply ramp to hydrodynamic pressure
            pressure_fluid *= self.ramp**2

            # Add external cushion pressure to external fluid pressure
            # This is the full-resolution calculation with all structural nodes and fluid elements
            pressure_cushion = np.zeros_like(s)
            if self._interpolator is not None:
                pressure_cushion[s > s_max] = self._interpolator.fluid.upstream_pressure
                pressure_cushion[s < s_min] = self._interpolator.fluid.downstream_pressure
            elif self.cushion_pressure_type == "Total":
                pressure_cushion[:] = self.cushion_pressure or self.config.body.Pc

            # Calculate internal pressure
            pressure_internal = np.full_like(s, self.seal_pressure * self.seal_over_pressure_pct)
            if self.seal_pressure_method.lower() == "hydrostatic":
                pressure_internal += (
                    self.config.flow.density
                    * self.config.flow.gravity
                    * (
                        self.config.flow.waterline_height
                        - np.array([self.get_coordinates(si)[1] for si in s])
                    )
                )

            # Derive various combinations of pressure
            pressure_air_net = pressure_internal - pressure_cushion
            pressure_external = pressure_fluid + pressure_cushion
            pressure_total = pressure_external - pressure_internal

            # Store fluid and air pressure components for element (for plotting)
            if i == 0:
                fluid_s.append(s[0])
                fluid_p.append(pressure_fluid[0])
                air_s.append(s[0])
                air_p.append(pressure_air_net[0])

            fluid_s.extend(ss for ss in s[1:])
            fluid_p.extend(pp for pp in pressure_fluid[1:])
            air_s.append(s[-1])
            air_p.append(pressure_air_net[-1])

            if not isinstance(self, TorsionalSpringSubstructure):
                el.qp = self._distribute_integrated_load_to_nodes(s, pressure_total)
                el.qs = self._distribute_integrated_load_to_nodes(s, -tau)

            # Calculate external force and moment for rigid body calculation
            method_pressure_map = {"integrated": pressure_external, "assumed": pressure_fluid}
            if (
                p := method_pressure_map.get(self.config.body.cushion_force_method.lower())
            ) is not None:
                f_x, f_y, moment = self._get_integrated_global_loads(s, p, tau)
                self.loads.D -= f_x
                self.loads.L += f_y
                self.loads.M += moment
            elif self._interpolator is not None:
                self.loads.D = self._interpolator.fluid.drag_total
                self.loads.L = self._interpolator.fluid.lift_total
                self.loads.M = self._interpolator.fluid.moment_total

            if isinstance(self, TorsionalSpringSubstructure):
                # Apply pressure loading for moment calculation
                #      integrand = pFl
                integrand = pressure_total
                n = list(map(self.get_normal_vector, s))
                t = [trig.rotate_vec_2d(ni, -90) for ni in n]

                f = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(integrand, tau, n, t)]
                r = [
                    np.array([pt[0] - self.base_pt[0], pt[1] - self.base_pt[1]])
                    for pt in map(self.get_coordinates, s)
                ]

                m = [math_helpers.cross2(ri, fi) for ri, fi in zip(r, f)]
                fx, fy = list(zip(*f))

                self.loads.Dt += math_helpers.integrate(s, np.array(fx))
                self.loads.Lt += math_helpers.integrate(s, np.array(fy))
                self.loads.Mt += math_helpers.integrate(s, np.array(m))

            integrand = pressure_cushion

            n = list(map(self.get_normal_vector, s))
            t = [trig.rotate_vec_2d(ni, -90) for ni in n]

            f = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(integrand, tau, n, t)]

            assert self.parent is not None
            r = [
                np.array([pt[0] - self.parent.xCofR, pt[1] - self.parent.yCofR])
                for pt in map(self.get_coordinates, s)
            ]

            m = [math_helpers.cross2(ri, fi) for ri, fi in zip(r, f)]

            self.loads.Da -= math_helpers.integrate(s, np.array(list(zip(*f))[0]))
            self.loads.La += math_helpers.integrate(s, np.array(list(zip(*f))[1]))
            self.loads.Ma += math_helpers.integrate(s, np.array(m))

        self.fluidP = np.array(fluid_p)
        self.fluidS = np.array(fluid_s)
        self.airP = np.array(air_p)
        self.airS = np.array(air_s)

    @staticmethod
    def _distribute_integrated_load_to_nodes(s: np.ndarray, load: np.ndarray) -> np.ndarray:
        """Integrate a load along an element, returning the equivalent load at each endpoint."""
        integral = math_helpers.integrate(s, load)
        if integral == 0.0:
            return np.zeros(2)
        pct = (math_helpers.integrate(s, s * load) / integral - s[0]) / math_helpers.cumdiff(s)
        return integral * np.array([1 - pct, pct])

    def _get_integrated_global_loads(
        self, s: np.ndarray, p: np.ndarray, tau: np.ndarray
    ) -> tuple[float, float, float]:
        """Integrate a pressure and shear stress along an arclength to get global loads.

        The moment is calculated about the center of rotation.

        Args:
            s: The arclength array.
            p: The pressure array.
            tau: The shear stress array.

        Returns:
            A tuple containing x-force, y-force, and z-moment.

        """
        n = np.array([self.get_normal_vector(s_i) for s_i in s])
        t = np.array([trig.rotate_vec_2d(n_i, -90) for n_i in n])
        f = -p[:, np.newaxis] * n + tau[:, np.newaxis] * t

        assert self.parent is not None
        c_of_r = np.array([self.parent.xCofR, self.parent.yCofR])
        r = [self.get_coordinates(s_i) - c_of_r for s_i in s]
        m = np.array([math_helpers.cross2(r_i, f_i) for r_i, f_i in zip(r, f)])

        drag = math_helpers.integrate(s, f[:, 0])
        lift = math_helpers.integrate(s, f[:, 1])
        moment = math_helpers.integrate(s, m)
        return drag, lift, moment

    def get_normal_vector(self, s: float) -> np.ndarray:
        """Calculate the normal vector at a specific arc length."""
        dx_ds = math_helpers.deriv(lambda si: self.get_coordinates(si)[0], s)
        dy_ds = math_helpers.deriv(lambda si: self.get_coordinates(si)[1], s)

        return trig.rotate_vec_2d(trig.angd2vec2d(trig.atand2(dy_ds, dx_ds)), -90)

    def fix_all_degrees_of_freedom(self) -> None:
        """Set all degrees of freedom of all nodes in the substructure."""
        for nd in self.nodes:
            nd.is_dof_fixed = tuple(True for _ in range(NUM_DIM))


class FlexibleSubstructure(Substructure):

    is_free = True
    _element_type: ElementType = fe.TrussElement

    def __init__(
        self,
        *,
        pretension: float = -0.5,
        axial_stiffness: float = 5e7,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.pretension = pretension
        self.EA = axial_stiffness

        self.K: np.ndarray | None = None
        self.F: np.ndarray | None = None
        self.U: np.ndarray | None = None

    def assemble_global_stiffness_and_force(self) -> None:
        if self.K is None or self.F is None:
            num_dof = len(self.solver.nodes) * NUM_DIM
            self.K = np.zeros((num_dof, num_dof))
            self.F = np.zeros((num_dof, 1))
            self.U = np.zeros((num_dof, 1))
        else:
            self.K *= 0
            self.F *= 0
        for el in self.elements:
            self.add_loads_from_element(el)

    def add_loads_from_element(self, el: "fe.Element") -> None:
        assert isinstance(el, fe.TrussElement), f"Element is of type {type(el)}"
        assert self.K is not None
        assert self.F is not None
        K, F = el.get_stiffness_and_force()
        el_dof = [dof for nd in el.nodes for dof in self.parent.parent.node_dofs[nd]]
        self.K[np.ix_(el_dof, el_dof)] += K
        self.F[np.ix_(el_dof)] += F

    def load_mesh(self, submesh: Path | Subcomponent = Path("mesh")) -> None:
        """Load the mesh, and assign structural properties to the elements."""
        super().load_mesh(submesh)
        for el in self.elements:
            el.initial_axial_force = -self.pretension
            el.EA = self.EA


class RigidSubstructure(Substructure):
    """A substructure that is rigidly attached to the rigid body.

    This means that the substructure moves with the rigid body but does not deform.

    """

    _element_type: ElementType = fe.RigidElement

    def load_mesh(self, submesh: Path | Subcomponent = Path("mesh")) -> None:
        super().load_mesh(submesh)
        self.fix_all_degrees_of_freedom()


class TorsionalSpringSubstructure(FlexibleSubstructure):
    """A substructure that is locally rigid but is attached to the rigid body with a torsional spring.

    Thus, the nodes can move, but rotate about a base point with a linear spring stiffness.

    Attributes:
        initial_angle: The initial offset angle of the structure, where 0 is the equilibrium angle
            with no load applied.
        tip_load: The vertical load to apply at a specified fractional arclength.
        tip_load_pct: The fraction along arclength at which to apply a fixed tip load.
        base_pt_pct: The fraction along arclength around which the substructure rotates.
        spring_constant: The linear spring constant [(N.m)/deg].
        relaxation_angle: A relaxation factor to apply to the angle change.
        minimum_angle: The minimum allowable angle for the substructure.
        max_angle_step: The maximum absolute change in angle during a single iteration.

    """

    base_pt: np.ndarray
    is_free = True
    _element_type: ElementType = fe.RigidElement

    def __init__(
        self,
        *,
        initial_angle: float = 0.0,
        tip_load: float = 0.0,
        tip_load_pct: float = 0.0,
        base_pt_pct: float = 1.0,
        spring_constant: float = 1e3,
        relaxation_angle: float | None = None,
        minimum_angle: float = -float("Inf"),
        max_angle_step: float = float("Inf"),
        attach_pct: float = 0.0,
        attached_substructure_name: str | None = None,
        attached_substructure_end: Literal["start"] | Literal["end"] = "end",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.tip_load = tip_load
        self.tip_load_pct = tip_load_pct
        self.base_pt_pct = base_pt_pct
        if spring_constant <= 0.0:
            raise ValueError("Spring constant must be positive.")
        self.spring_constant = spring_constant

        self._theta = initial_angle
        self._relax = relaxation_angle
        self.attach_pct = attach_pct
        self.attached_node: fe.Node | None = None
        self.attached_element: fe.Element | None = None
        self.minimum_angle = minimum_angle
        self.max_angle_step = max_angle_step
        self.attached_substructure_name = attached_substructure_name
        self.attached_substructure_end = attached_substructure_end
        self.attached_ind = 0
        self.attached_substructure: Substructure | None = None
        self.residual = 1.0

    @property
    def relax(self) -> float:
        """Relaxation parameter."""
        if self._relax is None:
            return self.config.body.relax_rigid_body
        return self._relax

    @property
    def angle(self) -> float:
        """The current rotation angle of the substructure."""
        return self._theta

    @angle.setter
    def angle(self, value: float) -> None:
        nodes = self.nodes.copy()
        if self.attached_node is not None and self.attached_node not in self.nodes:
            nodes.append(self.attached_node)

        angle_change = np.max([value, self.minimum_angle]) - self._theta
        for nd in nodes:
            nd.coordinates = trig.rotate_point(nd.coordinates, self.base_pt, -angle_change)

        self._theta += angle_change
        self.residual = abs(angle_change)
        self.update_geometry()
        logger.info(f"  Deformation for substructure {self.name}: {self._theta}")

    def load_mesh(self, submesh: Path | Subcomponent = Path("mesh")) -> None:
        super().load_mesh(submesh)
        self.fix_all_degrees_of_freedom()
        if self.base_pt_pct == 1.0:
            self.base_pt = self.nodes[-1].coordinates
        elif self.base_pt_pct == 0.0:
            self.base_pt = self.nodes[0].coordinates
        else:
            self.base_pt = self.get_coordinates(self.base_pt_pct * self.arc_length)

        self._set_attachments()

    def _set_attachments(self) -> None:
        if self.attached_substructure_name is not None:
            self.attached_substructure = self.parent.get_substructure_by_name(
                self.attached_substructure_name
            )

        if self.attached_substructure_end == "start":
            self.attached_ind = 0
        else:
            self.attached_ind = -1

        if self.attached_node is None and self.attached_substructure is not None:
            self.attached_node = self.attached_substructure.nodes[self.attached_ind]
            self.attached_element = self.attached_substructure.elements[self.attached_ind]

    def _apply_tip_load(self) -> None:
        """Apply any externally tip load."""
        tip_coords = self.get_coordinates(self.tip_load_pct * self.arc_length)
        tip_rel_coords = np.array([tip_coords[i] - self.base_pt[i] for i in [0, 1]])
        tip_force = np.array([0.0, self.tip_load]) * self.ramp
        tip_moment = math_helpers.cross2(tip_rel_coords, tip_force)
        self.loads.Lt += tip_force[1]
        self.loads.Mt += tip_moment

    def update_fluid_forces(self) -> None:
        super().update_fluid_forces()
        self._apply_tip_load()

    #     fluid_s: list[float] = []
    #     fluid_p: list[float] = []
    #     air_s: list[float] = []
    #     air_p: list[float] = []
    #     self.loads.D = 0.0
    #     self.loads.L = 0.0
    #     self.loads.M = 0.0
    #     self.loads.Dt = 0.0
    #     self.loads.Lt = 0.0
    #     self.loads.Mt = 0.0
    #     self.loads.Da = 0.0
    #     self.loads.La = 0.0
    #     self.loads.Ma = 0.0
    #     if self._interpolator is not None:
    #         s_min, s_max = self._interpolator.get_min_max_s()
    #
    #     for i, el in enumerate(self.elements):
    #         # Get pressure at end points and all fluid points along element
    #         node_s = [self.node_arc_length[i], self.node_arc_length[i + 1]]
    #         if self._interpolator is not None:
    #             s, pressure_fluid, tau = self._interpolator.get_loads_in_range(node_s[0], node_s[1])
    #
    #             # Limit pressure to be below stagnation pressure
    #             if self.config.plotting.pressure_limiter:
    #                 pressure_fluid = np.min(
    #                     np.hstack(
    #                         (
    #                             pressure_fluid,
    #                             np.ones_like(pressure_fluid) * self.config.flow.stagnation_pressure,
    #                         )
    #                     ),
    #                     axis=0,
    #                 )
    #
    #         else:
    #             s = np.array(node_s)
    #             pressure_fluid = np.zeros_like(s)
    #             tau = np.zeros_like(s)
    #
    #         ss = node_s[1]
    #         Pc = 0.0
    #         if self._interpolator is not None:
    #             if ss > s_max:
    #                 Pc = self._interpolator.fluid.upstream_pressure
    #             elif ss < s_min:
    #                 Pc = self._interpolator.fluid.downstream_pressure
    #         elif self.cushion_pressure_type == "Total":
    #             Pc = self.cushion_pressure or self.config.body.Pc
    #
    #         # Store fluid and air pressure components for element (for
    #         # plotting)
    #         if i == 0:
    #             fluid_s += [s[0]]
    #             fluid_p += [pressure_fluid[0]]
    #             air_s += [node_s[0]]
    #             air_p += [Pc - self.seal_pressure]
    #
    #         fluid_s += [ss for ss in s[1:]]
    #         fluid_p += [pp for pp in pressure_fluid[1:]]
    #         air_s += [ss for ss in node_s[1:]]
    #         if self.seal_pressure_method.lower() == "hydrostatic":
    #             air_p += [
    #                 Pc
    #                 - self.seal_pressure
    #                 + self.config.flow.density
    #                 * self.config.flow.gravity
    #                 * (self.get_coordinates(si)[1] - self.config.flow.waterline_height)
    #                 for si in node_s[1:]
    #             ]
    #         else:
    #             air_p += [Pc - self.seal_pressure for _ in node_s[1:]]
    #
    #         # Apply ramp to hydrodynamic pressure
    #         pressure_fluid *= self.ramp**2
    #
    #         # Add external cushion pressure to external fluid pressure
    #         pressure_cushion = np.zeros_like(s)
    #         Pc = 0.0
    #         for ii, ss in enumerate(s):
    #             if self._interpolator is not None:
    #                 if ss > s_max:
    #                     Pc = self._interpolator.fluid.upstream_pressure
    #                 elif ss < s_min:
    #                     Pc = self._interpolator.fluid.downstream_pressure
    #             elif self.cushion_pressure_type == "Total":
    #                 Pc = self.config.body.Pc
    #
    #             pressure_cushion[ii] = Pc
    #
    #         # Calculate internal pressure
    #         if self.seal_pressure_method.lower() == "hydrostatic":
    #             pressure_internal = (
    #                 self.seal_pressure
    #                 - self.config.flow.density
    #                 * self.config.flow.gravity
    #                 * (
    #                     np.array([self.get_coordinates(si)[1] for si in s])
    #                     - self.config.flow.waterline_height
    #                 )
    #             )
    #         else:
    #             pressure_internal = (
    #                 self.seal_pressure * np.ones_like(s) * self.seal_over_pressure_pct
    #             )
    #
    #         pressure_external = pressure_fluid + pressure_cushion
    #         pressure_total = pressure_external - pressure_internal
    #
    #         if not isinstance(self, TorsionalSpringSubstructure):
    #             # Integrate pressure profile, calculate center of pressure and
    #             # distribute force to nodes
    #             integral = math_helpers.integrate(s, pressure_total)
    #             if integral == 0.0:
    #                 qp = np.zeros(2)
    #             else:
    #                 pct = (
    #                               math_helpers.integrate(s, s * pressure_total) / integral - s[0]
    #                       ) / math_helpers.cumdiff(s)
    #                 qp = integral * np.array([1 - pct, pct])
    #
    #             integral = math_helpers.integrate(s, tau)
    #             if integral == 0.0:
    #                 qs = np.zeros(2)
    #             else:
    #                 pct = (math_helpers.integrate(s, s * tau) / integral - s[0]) / math_helpers.cumdiff(
    #                     s
    #                 )
    #                 qs = -integral * np.array([1 - pct, pct])
    #
    #             el.qp = qp
    #             el.qs = qs
    #
    #         # Calculate external force and moment for rigid body calculation
    #         if (
    #             self.config.body.cushion_force_method.lower() == "integrated"
    #             or self.config.body.cushion_force_method.lower() == "assumed"
    #         ):
    #             if self.config.body.cushion_force_method.lower() == "integrated":
    #                 integrand = pressure_external
    #             elif self.config.body.cushion_force_method.lower() == "assumed":
    #                 integrand = pressure_fluid
    #             else:
    #                 raise ValueError(
    #                     'Cushion force method must be either "integrated" or "assumed"'
    #                 )
    #
    #             n = [self.get_normal_vector(s_i) for s_i in s]
    #             t = [trig.rotate_vec_2d(n_i, -90) for n_i in n]
    #
    #             # TODO: This part is different
    #             fC = [
    #                 -pi * ni + taui * ti for pi, taui, ni, ti in zip(pressure_external, tau, n, t)
    #             ]
    #             fFl = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(pressure_fluid, tau, n, t)]
    #             f = fC + fFl
    #             print(("Cushion Lift-to-Weight: {0}".format(fC[1] / self.config.body.weight)))
    #
    #             r = [
    #                 np.array([pt[0] - self.parent.xCofR, pt[1] - self.parent.yCofR])
    #                 for pt in map(self.get_coordinates, s)
    #             ]
    #
    #             m = [math_helpers.cross2(ri, fi) for ri, fi in zip(r, f)]
    #
    #             self.loads.D -= math_helpers.integrate(s, np.array(list(zip(*f))[0]))
    #             self.loads.L += math_helpers.integrate(s, np.array(list(zip(*f))[1]))
    #             self.loads.M += math_helpers.integrate(s, np.array(m))
    #         else:
    #             if self._interpolator is not None:
    #                 self.loads.D = self._interpolator.fluid.drag_total
    #                 self.loads.L = self._interpolator.fluid.lift_total
    #                 self.loads.M = self._interpolator.fluid.moment_total
    #
    #         # Apply pressure loading for moment calculation
    #         #      integrand = pFl
    #         integrand = pressure_total
    #         n = list(map(self.get_normal_vector, s))
    #         t = [trig.rotate_vec_2d(ni, -90) for ni in n]
    #
    #         f = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(integrand, tau, n, t)]
    #         r = [
    #             np.array([pt[0] - self.base_pt[0], pt[1] - self.base_pt[1]])
    #             for pt in map(self.get_coordinates, s)
    #         ]
    #
    #         m = [math_helpers.cross2(ri, fi) for ri, fi in zip(r, f)]
    #         fx, fy = list(zip(*f))
    #
    #         self.loads.Dt += math_helpers.integrate(s, np.array(fx))
    #         self.loads.Lt += math_helpers.integrate(s, np.array(fy))
    #         self.loads.Mt += math_helpers.integrate(s, np.array(m))
    #
    #         integrand = pressure_cushion
    #
    #         n = list(map(self.get_normal_vector, s))
    #         t = [trig.rotate_vec_2d(ni, -90) for ni in n]
    #
    #         f = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(integrand, tau, n, t)]
    #
    #         assert self.parent is not None
    #         r = [
    #             np.array([pt[0] - self.parent.xCofR, pt[1] - self.parent.yCofR])
    #             for pt in map(self.get_coordinates, s)
    #         ]
    #
    #         m = [math_helpers.cross2(ri, fi) for ri, fi in zip(r, f)]
    #
    #         self.loads.Da -= math_helpers.integrate(s, np.array(list(zip(*f))[0]))
    #         self.loads.La += math_helpers.integrate(s, np.array(list(zip(*f))[1]))
    #         self.loads.Ma += math_helpers.integrate(s, np.array(m))
    #
    #     # Apply tip load
    #     tipC = self.get_coordinates(self.tip_load_pct * self.arc_length)
    #     tipR = np.array([tipC[i] - self.base_pt[i] for i in [0, 1]])
    #     tipF = np.array([0.0, self.tip_load]) * self.ramp
    #     tipM = math_helpers.cross2(tipR, tipF)
    #     self.loads.Lt += tipF[1]
    #     self.loads.Mt += tipM
    #
    #     self.fluidP = np.array(fluid_p)
    #     self.fluidS = np.array(fluid_s)
    #     self.airP = np.array(air_p)
    #     self.airS = np.array(air_s)
    #
    #     # Apply moment from attached substructure
    #
    # #    el = self.attachedEl
    # #    attC = self.attachedNode.get_coordinates()
    # #    attR = np.array([attC[i] - self.basePt[i] for i in [0,1]])
    # #    attF = el.axialForce * kp.ang2vec(el.gamma + 180)
    # #    attM = kp.cross2(attR, attF) * config.ramp
    # #    attM = np.min([np.abs(attM), np.abs(self.loads.Mt)]) * kp.sign(attM)
    # # if np.abs(attM) > 2 * np.abs(tipM):
    # #      attM = attM * np.abs(tipM) / np.abs(attM)
    # #    self.loads.Mt += attM

    def update_angle(self) -> None:
        """Update the angle of the substructure."""
        if np.isnan(self.loads.Mt):
            theta = 0.0
        else:
            theta = -self.loads.Mt / self.spring_constant

        # Ramp and limit the angle change
        angle_change = (theta - self._theta) * self.relax
        angle_change = min(abs(angle_change), self.max_angle_step) * np.sign(angle_change)

        self.angle = self._theta + angle_change

    def write_deformation(self) -> None:
        """Write the deformation angle to a file."""
        writers.write_as_dict(
            self.it_dir / f"deformation_{self.name}.{self.config.io.data_format}",
            ["angle", self._theta],
        )


class Interpolator:
    """An object that handles two-way communication between the structural and fluid solvers."""

    def __init__(
        self,
        solid: Substructure,
        fluid: PlaningSurface,
        *,
        waterline_height: float = 0.0,
        separation_arclength_start_pct: float = 0.5,
        immersion_arclength_start_pct: float = 0.9,
        **_: Any,
    ):
        self.solid = solid
        self.fluid = fluid

        self._waterline_height = waterline_height
        self._separation_arclength_start_pct = separation_arclength_start_pct
        self._immersion_arclength_start_pct = immersion_arclength_start_pct

        self._separation_arclength: float | None = None
        self._immersed_arclength: float | None = None

    def get_body_height(self, x: float) -> float:
        """Return the elevation (height) of the substructure surface at a given x-coordinate.

        Args:
            x: The x-coordinate.

        Returns:
            The elevation of the surface.

        """
        s = np.max([self.get_s_fixed_x(x), 0.0])
        return self.get_coordinates(s)[1]

    def get_coordinates(self, s: float) -> np.ndarray:
        """The surface coordinates at a given arclength."""
        return self.solid.get_coordinates(s)

    def get_min_max_s(self) -> tuple[float, float]:
        """Return the min and max arclength that the fluid elements cover.

        Returns:
            A tuple of (min, max) arclength.

        """
        pts = self.fluid.get_element_coords()
        return self.get_s_fixed_x(pts[0]), self.get_s_fixed_x(pts[-1])

    def get_s_fixed_x(self, x: float, so_pct: float = 0.5) -> float:
        """Return the arclength of the substructure surface at a given x-coordinate.

        Args:
            x: The x-coordinate.
            so_pct: The starting arclength guess, as a fraction of the total arclength.

        Returns:
            The arclength of the surface.

        """
        return fzero(lambda s: self.get_coordinates(s)[0] - x, so_pct * self.solid.arc_length)

    @property
    def immersed_length(self) -> float:
        """The total immersed length of the substructure.

        This is the length of the surface below the undisturbed waterline.

        """
        if self._immersed_arclength is None:
            self._immersed_arclength = self._immersion_arclength_start_pct * self.solid.arc_length

        self._immersed_arclength = fzero(
            lambda s: self.get_coordinates(s)[1] - self._waterline_height,
            self._immersed_arclength,
        )

        return self.get_coordinates(self._immersed_arclength)[0]

    def get_separation_point(self) -> np.ndarray:
        """Return the coordinates of the assumed separation point.

        The separation is assumed to occur at the point of lowest elevation along the surface.

        Returns:
            A 2-d array containing the (x,y) coordinates of the separation point.

        """

        def get_y_coords(s: float) -> float:
            return self.get_coordinates(s)[1]

        if self._separation_arclength is None:
            self._separation_arclength = (
                self._separation_arclength_start_pct * self.solid.arc_length
            )

        self._separation_arclength = fmin(
            get_y_coords, self._separation_arclength, disp=False, xtol=1e-6
        )[0]
        self._separation_arclength = float(np.max([self._separation_arclength, 0.0]))
        return self.get_coordinates(self._separation_arclength)

    def get_loads_in_range(
        self, s0: float, s1: float, /, *, pressure_limit: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the fluid pressure and shear stress at any pressure elements within the provided arclength range.

        Args:
            s0: The lower-bound arclength.
            s1: The upper-bound arclength.
            pressure_limit: Optionally limit the pressure to be below a specified value (e.g. stagnation pressure).

        Returns:
            A tuple of arrays containing the arclength, pressure, and shear stress at each pressure element point.

        """
        x, p, tau = self.fluid.get_loads_in_range(
            self.solid.get_coordinates(s0)[0],
            self.solid.get_coordinates(s1)[0],
            pressure_limit=pressure_limit,
        )
        s = np.array([self.get_s_fixed_x(xx) for xx in x])
        s[0], s[-1] = s0, s1  # Force ends to be exactly the same
        return s, p, tau
