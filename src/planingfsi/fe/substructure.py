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
class GlobalLoads:
    """The integrated global loads on the substructure.

    Attributes:
        D: The hydrodynamic drag.
        L: The hydrodynamic lift.
        M: The hydrodynamic moment (about the body center of rotation).
        Da: The air drag.
        La: The air lift.
        Ma: The air moment (about the body center of rotation).
        Dtip: The drag externally applied to the torsional substructure tip.
        Ltip: The lift externally applied to the torsional substructure tip.
        Mtip: The moment externally applied to the torsional substructure tip
            (about the body center of rotation).

    """

    D: float = 0.0
    L: float = 0.0
    M: float = 0.0
    Da: float = 0.0
    La: float = 0.0
    Ma: float = 0.0
    Dtip: float = 0.0
    Ltip: float = 0.0
    Mtip: float = 0.0

    def __add__(self, other: GlobalLoads) -> GlobalLoads:
        if not isinstance(other, GlobalLoads):
            raise TypeError("Must add GlobalLoads to GlobalLoads")

        return GlobalLoads(
            D=self.D + other.D,
            L=self.L + other.L,
            M=self.M + other.M,
            Da=self.Da + other.Da,
            La=self.La + other.La,
            Ma=self.Ma + other.Ma,
            Dtip=self.Dtip + other.Dtip,
            Ltip=self.Ltip + other.Ltip,
            Mtip=self.Mtip + other.Mtip,
        )


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
        rigid_body: A reference to the parent `RigidBody` instance.

    """

    is_free = False
    _element_type: ElementType

    def __new__(cls, *, type: str | None = None, **kwargs):
        """This is a factory pattern. If the type is provided, that class will be used
        to create the new object. If not provided, whichever class is instantiated will be used.
        """
        if type is None:
            ss_class = cls
        elif type.lower() == "flexible" or type.lower() == "truss":
            ss_class = FlexibleMembraneSubstructure
        elif type.lower() == "torsionalspring":
            ss_class = TorsionalSpringSubstructure
        else:
            ss_class = RigidSubstructure
        return super().__new__(ss_class)

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

        self.seal_pressure = seal_pressure
        self.seal_pressure_method = seal_pressure_method
        self.seal_over_pressure_pct = seal_over_pressure_pct

        self.cushion_pressure: float | None = None
        self.cushion_pressure_type = cushion_pressure_type

        self.struct_interp_type = struct_interp_type
        self.struct_extrap = struct_extrap

        self._solver = solver
        self.rigid_body = parent

        # Arrays to store air and fluid pressure profiles for plotting
        self.s_hydro: np.ndarray | None = None
        self.p_hydro: np.ndarray | None = None
        self.s_air: np.ndarray | None = None
        self.p_air: np.ndarray | None = None

        self.elements: list[fe.Element] = []
        self.node_arc_length: np.ndarray | None = None

        self.loads = GlobalLoads()

        self._interpolator: Interpolator | None = None
        self._interp_coords_at_arclength: interp1d | None = None

    @property
    def solver(self) -> StructuralSolver:
        """A reference to the structural solver. Can be explicitly set, or else traverses the parents."""
        if (
            self._solver is None
            and self.rigid_body is not None
            and self.rigid_body.parent is not None
        ):
            return self.rigid_body.parent
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
        if self.rigid_body is None:
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
        return self.node_arc_length[-1]

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
        """Write the coordinates of all component nodes to file."""
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
        """Update the fluid forces on all structural elements.

        The pressure is interpolated, both hydrostatic and air, and applied in various
        manners depending on the type of substructure.

        """

        s_hydro: list[float] = []
        p_hydro: list[float] = []
        s_air: list[float] = []
        p_air: list[float] = []
        self.loads = GlobalLoads()
        if self._interpolator is not None:
            s_min, s_max = self._interpolator.get_min_max_s()

        for i, el in enumerate(self.elements):
            # Get pressure & shear stress at end points and all fluid points along element
            s_start, s_end = self.node_arc_length[i], self.node_arc_length[i + 1]
            s, pressure_hydro, tau = self._get_loads_in_range(s_start, s_end)

            # Apply ramp to hydrodynamic pressure
            pressure_hydro *= self.ramp**2

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
            pressure_external = pressure_hydro + pressure_cushion
            pressure_total = pressure_external - pressure_internal

            # Store fluid and air pressure components for element (for plotting)
            if i == 0:
                s_hydro.append(s[0])
                p_hydro.append(pressure_hydro[0])
                s_air.append(s[0])
                p_air.append(pressure_air_net[0])

            s_hydro.extend(s[1:])
            p_hydro.extend(pressure_hydro[1:])
            s_air.append(s[-1])
            p_air.append(pressure_air_net[-1])

            if isinstance(self, FlexibleMembraneSubstructure):
                el.qp = self._distribute_integrated_load_to_nodes(s, pressure_total)
                el.qs = self._distribute_integrated_load_to_nodes(s, -tau)

            # Calculate external force and moment for rigid body calculation
            if self._interpolator is not None:
                self.loads.D = self._interpolator.fluid.drag_total
                self.loads.L = self._interpolator.fluid.lift_total
                self.loads.M = self._interpolator.fluid.moment_total
            else:
                method_pressure_map = {"integrated": pressure_external, "assumed": pressure_hydro}
                if (
                    p := method_pressure_map.get(self.config.body.cushion_force_method.lower())
                ) is not None:
                    f_x, f_y, moment = self._get_integrated_global_loads(s, p, tau)
                    self.loads.D -= f_x
                    self.loads.L += f_y
                    self.loads.M += moment

            # Integrate the total pressure for torsional spring calculations
            if isinstance(self, TorsionalSpringSubstructure):
                _, _, moment = self._get_integrated_global_loads(
                    s, pressure_total, tau, moment_about=self.base_pt
                )
                self._applied_moment += moment

            # Integrate global cushion pressure force and moment
            f_x, f_y, moment = self._get_integrated_global_loads(s, pressure_cushion)
            self.loads.Da -= f_x
            self.loads.La += f_y
            self.loads.Ma += moment

        self.p_hydro = np.array(p_hydro)
        self.s_hydro = np.array(s_hydro)
        self.p_air = np.array(p_air)
        self.s_air = np.array(s_air)

    @staticmethod
    def _distribute_integrated_load_to_nodes(s: np.ndarray, load: np.ndarray) -> np.ndarray:
        """Integrate a load along an element, returning the equivalent load at each endpoint."""
        integral = math_helpers.integrate(s, load)
        if integral == 0.0:
            return np.zeros(2)
        pct = (math_helpers.integrate(s, s * load) / integral - s[0]) / math_helpers.cumdiff(s)
        return integral * np.array([1 - pct, pct])

    def _get_integrated_global_loads(
        self,
        s: np.ndarray,
        p: np.ndarray,
        tau: np.ndarray | None = None,
        *,
        moment_about: np.ndarray | None = None,
    ) -> tuple[float, float, float]:
        """Integrate a pressure and shear stress along an arclength to get global loads.

        The moment is calculated about the center of rotation unless an alternative point is provided.

        Args:
            s: The arclength array.
            p: The pressure array.
            tau: The shear stress array.
            moment_about: The point about which to calculate the moment.

        Returns:
            A tuple containing x-force, y-force, and z-moment.

        """
        if tau is None:
            tau = np.zeros_like(p)

        n = np.array([self.get_normal_vector(s_i) for s_i in s])
        t = np.array([trig.rotate_vec_2d(n_i, -90) for n_i in n])
        f = -p[:, np.newaxis] * n + tau[:, np.newaxis] * t

        assert self.rigid_body is not None
        if moment_about is None:
            moment_about = np.array([self.rigid_body.x_cr, self.rigid_body.y_cr])

        r = [self.get_coordinates(s_i) - moment_about for s_i in s]
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


class RigidSubstructure(Substructure):
    """A substructure that is rigidly attached to the rigid body.

    This means that the substructure moves with the rigid body but does not deform.

    """

    _element_type: ElementType = fe.RigidElement

    def load_mesh(self, submesh: Path | Subcomponent = Path("mesh")) -> None:
        super().load_mesh(submesh)
        self.fix_all_degrees_of_freedom()


class FlexibleMembraneSubstructure(Substructure):
    """A flexible membrane structure, composed of truss finite elements.

    The structure supports axial loading only (i.e. no bending), and supports large deformations.

    Attributes:
        pretension: An optional pretension used to initialize the elements.
        EA: The axial stiffness of the component `Element`s.

    """

    is_free = True
    _element_type: ElementType = fe.TrussElement
    elements: list[fe.TrussElement]

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

    def assemble_global_stiffness_and_force(
        self, K_global: np.ndarray, F_global: np.ndarray
    ) -> None:
        """Assemble element stiffness and force into the global matrices, which are passed by reference and modified."""
        for el in self.elements:
            K_el, F_el = el.get_stiffness_and_force()
            el_dof = [dof for nd in el.nodes for dof in self.rigid_body.parent.node_dofs[nd]]
            K_global[np.ix_(el_dof, el_dof)] += K_el
            F_global[np.ix_(el_dof)] += F_el

    def load_mesh(self, submesh: Path | Subcomponent = Path("mesh")) -> None:
        """Load the mesh, and assign structural properties to the elements."""
        super().load_mesh(submesh)
        for el in self.elements:
            el.initial_axial_force = -self.pretension
            el.EA = self.EA


class TorsionalSpringSubstructure(Substructure):
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
        self._applied_moment = 0.0

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
            self.attached_substructure = self.rigid_body.get_substructure_by_name(
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
        tip_rel_coords = tip_coords - self.base_pt
        tip_force = np.array([0.0, self.tip_load]) * self.ramp
        tip_moment = math_helpers.cross2(tip_rel_coords, tip_force)
        self._applied_moment += tip_moment
        # Add to the global forces
        self.loads.Ltip += tip_force[1]
        self.loads.Mtip += math_helpers.cross2(
            tip_coords - np.array([self.rigid_body.x_cr, self.rigid_body.x_cr]), tip_force
        )

    def update_fluid_forces(self) -> None:
        self._applied_moment = 0.0
        super().update_fluid_forces()
        self._apply_tip_load()

    def update_angle(self) -> None:
        """Update the angle of the substructure."""
        if np.isnan(self._applied_moment):
            theta = 0.0
        else:
            theta = -self._applied_moment / self.spring_constant

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
        pts = self.fluid.element_coords
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
