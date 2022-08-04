from __future__ import annotations

import abc
from collections.abc import Callable
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Literal

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fmin

from planingfsi import logger
from planingfsi import math_helpers
from planingfsi import trig
from planingfsi import writers
from planingfsi.config import NUM_DIM
from planingfsi.fe import felib as fe
from planingfsi.fe.femesh import Submesh
from planingfsi.solver import fzero

if TYPE_CHECKING:
    from planingfsi.config import Config
    from planingfsi.fe.rigid_body import RigidBody
    from planingfsi.fe.structure import StructuralSolver
    from planingfsi.potentialflow.pressurepatch import PlaningSurface
    from planingfsi.potentialflow.pressurepatch import PressureCushion


class Substructure(abc.ABC):

    __all: list["Substructure"] = []

    _element_type: ClassVar[type[fe.Element]]

    is_free = False

    @classmethod
    def find_by_name(cls, name: str) -> "Substructure":
        """Return a substructure whose name matches the argument."""
        for o in cls.__all:
            if o.name == name:
                return o
        raise NameError(f"Cannot find Substructure with name {name}")

    def __init__(
        self,
        *,
        name: str = "",
        seal_pressure: float = 0.0,
        seal_pressure_method: Literal["constant"] | Literal["hydrostatic"] = "constant",
        seal_over_pressure_pct: float = 1.0,
        cushion_pressure_type: str | None = None,
        tip_load: float = 0.0,
        tip_constraint_height: float | None = None,
        struct_interp_type: Literal["linear"] | Literal["quadratic"] | Literal["cubic"] = "linear",
        struct_extrap: bool = True,
        solver: "StructuralSolver" | None = None,
        parent: RigidBody | None = None,
        **_: Any,
    ):
        self.index = len(self.__all)
        Substructure.__all.append(self)

        self.name = name
        self.interpolator: Interpolator | None = None

        self.external_pressure: float | None = None
        self.seal_pressure = seal_pressure
        self.seal_pressure_method = seal_pressure_method

        self.seal_over_pressure_pct = seal_over_pressure_pct
        self.cushion_pressure_type = cushion_pressure_type
        self.tip_load = tip_load
        self.tip_constraint_height = tip_constraint_height
        self.struct_interp_type = struct_interp_type
        self.struct_extrap = struct_extrap

        self._solver = solver
        self.parent = parent

        self.line_fluid_pressure = None
        self.line_air_pressure = None
        self.fluidS: np.ndarray | None = None
        self.fluidP: np.ndarray | None = None
        self.airS: np.ndarray | None = None
        self.airP: np.ndarray | None = None
        self.U: np.ndarray | None = None
        self.node: list[fe.Node] = []
        self.el: list[fe.Element] = []
        self.node_arc_length = np.zeros(len(self.node))

        self.D = 0.0
        self.L = 0.0
        self.M = 0.0
        self.Dt = 0.0
        self.Lt = 0.0
        self.Da = 0.0
        self.La = 0.0
        self.Ma = 0.0

        self.interp_func_x: interp1d | None = None
        self.interp_func_y: interp1d | None = None

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

    def add_planing_surface(self, planing_surface: PlaningSurface, **kwargs: Any) -> PlaningSurface:
        """Add a planing surface to the substructure, and configure the interpolator.

        Args:
            planing_surface: The planing surface.
            **kwargs: Keyword arguments to pass through to the Interpolator.

        Returns:
            The same planing surface passed in as an argument (useful for chaining).

        """
        # Assign the same interpolator to both the substructure and planing_surface
        planing_surface.interpolator = self.interpolator = Interpolator(
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
        self.external_pressure = pressure_cushion.cushion_pressure
        return pressure_cushion

    def set_element_properties(self) -> None:
        """Set the properties of each element."""
        for el in self.el:
            el.set_properties(length=self.arc_length / len(self.el))

    def load_mesh(self, submesh: Path | Submesh = Path("mesh")) -> None:
        if isinstance(submesh, Submesh):
            nd_st, nd_end = [], []
            for curve in submesh.curves:
                for line in curve.lines:
                    nd_st.append(line.pt[0].index)
                    nd_end.append(line.pt[1].index)
        else:
            nd_st_arr, nd_end_arr = np.loadtxt(submesh / f"elements_{self.name}.txt", unpack=True)
            nd_st, nd_end = list(nd_st_arr), list(nd_end_arr)

        if isinstance(nd_st, float):
            nd_st = [int(nd_st)]
            nd_end = [int(nd_end)]
        else:
            nd_st = [int(nd) for nd in nd_st]
            nd_end = [int(nd) for nd in nd_end]
        ndInd = nd_st + [nd_end[-1]]

        # Generate Element list
        self.node = [fe.Node.get_index(i) for i in ndInd]

        self.set_interp_function()
        self.el = [self._element_type(parent=self) for _ in nd_st]
        self.set_element_properties()
        for ndSti, ndEndi, el in zip(nd_st, nd_end, self.el):
            el.set_nodes([fe.Node.get_index(ndSti), fe.Node.get_index(ndEndi)])

    def set_interp_function(self) -> None:
        self.node_arc_length = np.zeros(len(self.node))
        for i, nd0, nd1 in zip(list(range(len(self.node) - 1)), self.node[:-1], self.node[1:]):
            self.node_arc_length[i + 1] = (
                self.node_arc_length[i] + ((nd1.x - nd0.x) ** 2 + (nd1.y - nd0.y) ** 2) ** 0.5
            )

        if len(self.node_arc_length) == 2:
            self.struct_interp_type = "linear"
        elif len(self.node_arc_length) == 3 and not self.struct_interp_type == "linear":
            self.struct_interp_type = "quadratic"

        x, y = [np.array(xx) for xx in zip(*[(nd.x, nd.y) for nd in self.node])]
        self.interp_func_x, self.interp_func_y = (
            interp1d(self.node_arc_length, x),
            interp1d(self.node_arc_length, y, kind=self.struct_interp_type),
        )

        if self.struct_extrap:
            assert self.interp_func_x is not None
            assert self.interp_func_y is not None
            self.interp_func_x, self.interp_func_y = self._extrap_coordinates(
                self.interp_func_x, self.interp_func_y
            )

    @staticmethod
    def _extrap_coordinates(fxi: Callable, fyi: Callable) -> tuple[Callable, Callable]:
        """Return a new callable that provides extrapolation."""

        def extrap1d(interpolator: interp1d) -> Callable:
            xs = interpolator.x
            ys = interpolator.y

            def pointwise(xi: float) -> float:
                if xi < xs[0]:
                    return ys[0] + (xi - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
                elif xi > xs[-1]:
                    return ys[-1] + (xi - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
                else:
                    return interpolator(xi)

            def ufunclike(xs: float) -> np.ndarray:
                return np.array(list(map(pointwise, np.array([xs]))))[0]

            return ufunclike

        return extrap1d(fxi), extrap1d(fyi)

    def get_coordinates(self, si: float) -> np.ndarray:
        assert self.interp_func_x is not None
        assert self.interp_func_y is not None
        return np.array([self.interp_func_x(si), self.interp_func_y(si)])

    def get_x_coordinates(self, s: float) -> float:
        return self.get_coordinates(s)[0]

    def get_y_coordinates(self, s: float) -> float:
        return self.get_coordinates(s)[1]

    @property
    def arc_length(self) -> float:
        return max(self.node_arc_length)

    @property
    def it_dir(self) -> Path:
        return self.solver.simulation.it_dir

    def write_coordinates(self) -> None:
        """Write the coordinates to file"""
        writers.write_as_list(
            self.it_dir / f"coords_{self.name}.{self.config.io.data_format}",
            ["x [m]", [nd.x for nd in self.node]],
            ["y [m]", [nd.y for nd in self.node]],
        )

    def load_coordinates(self) -> None:
        x, y = np.loadtxt(
            str(self.it_dir / f"coords_{self.name}.{self.config.io.data_format}"),
            unpack=True,
        )
        for xx, yy, nd in zip(x, y, self.node):
            nd.set_coordinates(xx, yy)

    def update_fluid_forces(self) -> None:
        fluid_s: list[float] = []
        fluid_p: list[float] = []
        air_s: list[float] = []
        air_p: list[float] = []
        self.D = 0.0
        self.L = 0.0
        self.M = 0.0
        self.Da = 0.0
        self.La = 0.0
        self.Ma = 0.0
        if self.interpolator is not None:
            s_min, s_max = self.interpolator.get_min_max_s()

        for i, el in enumerate(self.el):
            # Get pressure at end points and all fluid points along element
            node_s = [self.node_arc_length[i], self.node_arc_length[i + 1]]
            if self.interpolator is not None:
                s, pressure_fluid, tau = self.interpolator.get_loads_in_range(node_s[0], node_s[1])
                # Limit pressure to be below stagnation pressure
                if self.config.plotting.pressure_limiter:
                    pressure_fluid = np.min(
                        np.hstack(
                            (
                                pressure_fluid,
                                np.ones_like(pressure_fluid) * self.config.flow.stagnation_pressure,
                            )
                        ),
                        axis=0,
                    )

            else:
                s = np.array(node_s)
                pressure_fluid = np.zeros_like(s)
                tau = np.zeros_like(s)

            ss = node_s[1]
            Pc = 0.0
            if self.interpolator is not None:
                if ss > s_max:
                    Pc = self.interpolator.fluid.upstream_pressure
                elif ss < s_min:
                    Pc = self.interpolator.fluid.downstream_pressure
            elif self.cushion_pressure_type == "Total":
                Pc = self.external_pressure or self.config.body.Pc

            # Store fluid and air pressure components for element (for
            # plotting)
            if i == 0:
                fluid_s += [s[0]]
                fluid_p += [pressure_fluid[0]]
                air_s += [node_s[0]]
                air_p += [Pc - self.seal_pressure]

            fluid_s += [ss for ss in s[1:]]
            fluid_p += [pp for pp in pressure_fluid[1:]]
            air_s += [ss for ss in node_s[1:]]
            if self.seal_pressure_method.lower() == "hydrostatic":
                assert self.interp_func_y is not None
                air_p += [
                    Pc
                    - self.seal_pressure
                    + self.config.flow.density
                    * self.config.flow.gravity
                    * (self.interp_func_y(si) - self.config.flow.waterline_height)
                    for si in node_s[1:]
                ]
            else:
                air_p += [Pc - self.seal_pressure for _ in node_s[1:]]

            # Apply ramp to hydrodynamic pressure
            pressure_fluid *= self.ramp**2

            # Add external cushion pressure to external fluid pressure
            pressure_cushion = np.zeros_like(s)
            Pc = 0.0
            for ii, ss in enumerate(s):
                if self.interpolator is not None:
                    if ss > s_max:
                        Pc = self.interpolator.fluid.upstream_pressure
                    elif ss < s_min:
                        Pc = self.interpolator.fluid.downstream_pressure
                elif self.cushion_pressure_type == "Total":
                    Pc = self.config.body.Pc

                pressure_cushion[ii] = Pc

            # Calculate internal pressure
            if self.seal_pressure_method.lower() == "hydrostatic":
                assert self.interp_func_y is not None
                pressure_internal = (
                    self.seal_pressure
                    - self.config.flow.density
                    * self.config.flow.gravity
                    * (
                        np.array([self.interp_func_y(si) for si in s])
                        - self.config.flow.waterline_height
                    )
                )
            else:
                pressure_internal = (
                    self.seal_pressure * np.ones_like(s) * self.seal_over_pressure_pct
                )

            pressure_external = pressure_fluid + pressure_cushion
            pressure_total = pressure_external - pressure_internal

            # Integrate pressure profile, calculate center of pressure and
            # distribute force to nodes
            integral = math_helpers.integrate(s, pressure_total)
            if integral == 0.0:
                qp = np.zeros(2)
            else:
                pct = (
                    math_helpers.integrate(s, s * pressure_total) / integral - s[0]
                ) / math_helpers.cumdiff(s)
                qp = integral * np.array([1 - pct, pct])

            integral = math_helpers.integrate(s, tau)
            if integral == 0.0:
                qs = np.zeros(2)
            else:
                pct = (math_helpers.integrate(s, s * tau) / integral - s[0]) / math_helpers.cumdiff(
                    s
                )
                qs = -integral * np.array([1 - pct, pct])

            el.qp = qp
            el.qs = qs

            # Calculate external force and moment for rigid body calculation
            if (
                self.config.body.cushion_force_method.lower() == "integrated"
                or self.config.body.cushion_force_method.lower() == "assumed"
            ):
                if self.config.body.cushion_force_method.lower() == "integrated":
                    integrand = pressure_external
                elif self.config.body.cushion_force_method.lower() == "assumed":
                    integrand = pressure_fluid
                else:
                    raise ValueError(
                        'Cushion force method must be either "integrated" or "assumed"'
                    )

                n = [self.get_normal_vector(s_i) for s_i in s]
                t = [trig.rotate_vec_2d(n_i, -90) for n_i in n]

                f = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(integrand, tau, n, t)]

                assert self.parent is not None
                r = [
                    np.array([pt[0] - self.parent.xCofR, pt[1] - self.parent.yCofR])
                    for pt in map(self.get_coordinates, s)
                ]

                m = [math_helpers.cross2(ri, fi) for ri, fi in zip(r, f)]

                self.D -= math_helpers.integrate(s, np.array(list(zip(*f))[0]))
                self.L += math_helpers.integrate(s, np.array(list(zip(*f))[1]))
                self.M += math_helpers.integrate(s, np.array(m))
            else:
                if self.interpolator is not None:
                    self.D = self.interpolator.fluid.drag_total
                    self.L = self.interpolator.fluid.lift_total
                    self.M = self.interpolator.fluid.moment_total

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

            self.Da -= math_helpers.integrate(s, np.array(list(zip(*f))[0]))
            self.La += math_helpers.integrate(s, np.array(list(zip(*f))[1]))
            self.Ma += math_helpers.integrate(s, np.array(m))
            self.fluidP = np.array(fluid_p)
            self.fluidS = np.array(fluid_s)
            self.airP = np.array(air_p)
            self.airS = np.array(air_s)

    def get_normal_vector(self, s: float) -> np.ndarray:
        """Calculate the normal vector at a specific arc length."""
        dx_ds = math_helpers.deriv(lambda si: self.get_coordinates(si)[0], s)
        dy_ds = math_helpers.deriv(lambda si: self.get_coordinates(si)[1], s)

        return trig.rotate_vec_2d(trig.angd2vec2d(trig.atand2(dy_ds, dx_ds)), -90)

    def plot_pressure_profiles(self) -> None:
        # TODO: Move to plotting directory
        if self.line_fluid_pressure is not None:
            self.line_fluid_pressure.set_data(
                self.get_pressure_plot_points(self.fluidS, self.fluidP)
            )
        if self.line_air_pressure is not None:
            self.line_air_pressure.set_data(self.get_pressure_plot_points(self.airS, self.airP))

    def get_pressure_plot_points(self, s0: np.ndarray, p0: np.ndarray) -> Iterable[Iterable]:

        sp = [(s, p) for s, p in zip(s0, p0) if not np.abs(p) < 1e-4]

        if len(sp) > 0:
            s0, p0 = list(zip(*sp))
            nVec = list(map(self.get_normal_vector, s0))
            coords0 = [np.array(self.get_coordinates(s)) for s in s0]
            coords1 = [
                c + self.config.plotting.pressure_scale * p * n
                for c, p, n in zip(coords0, p0, nVec)
            ]

            return list(
                zip(
                    *[
                        xyi
                        for c0, c1 in zip(coords0, coords1)
                        for xyi in [c0, c1, np.ones(2) * np.nan]
                    ]
                )
            )
        else:
            return [], []

    def update_geometry(self) -> None:
        self.set_interp_function()

    def plot(self) -> None:
        # TODO: Move to plotting module
        for el in self.el:
            el.plot()
        #    for nd in [self.node[0],self.node[-1]]:
        #      nd.plot()
        self.plot_pressure_profiles()

    def set_attachments(self) -> None:
        return None

    def set_angle(self, _: float) -> None:
        return None

    @abc.abstractmethod
    def set_fixed_dof(self) -> None:
        raise NotImplementedError


class FlexibleSubstructure(Substructure):

    __all: list["FlexibleSubstructure"] = []
    res = 0.0
    is_free = True
    _element_type: ClassVar[type[fe.Element]] = fe.TrussElement

    @classmethod
    def update_all(cls, rigid_body: "RigidBody") -> None:
        # TODO: This functionality should be moved to the rigid body

        num_dof = fe.Node.count() * NUM_DIM
        Kg = np.zeros((num_dof, num_dof))
        Fg = np.zeros((num_dof, 1))
        Ug = np.zeros((num_dof, 1))

        # Assemble global matrices for all substructures together
        for ss in cls.__all:
            ss.update_fluid_forces()
            ss.assemble_global_stiffness_and_force()

            # TODO: Consider removing this and fixing static types
            assert ss.K is not None
            assert ss.F is not None

            Kg += ss.K
            Fg += ss.F

        for nd in fe.Node.all():
            for i in range(2):
                Fg[nd.dof[i]] += nd.fixed_load[i]

        # Determine fixed degrees of freedom
        dof = [False for _ in Fg]

        for nd in fe.Node.all():
            for dofi, fdofi in zip(nd.dof, nd.is_dof_fixed):
                dof[dofi] = not fdofi

        # Solve FEM linear matrix equation
        if any(dof):
            Ug[np.ix_(dof)] = np.linalg.solve(Kg[np.ix_(dof, dof)], Fg[np.ix_(dof)])

        cls.res = np.max(np.abs(Ug))

        Ug *= rigid_body.config.solver.relax_FEM
        Ug *= np.min([rigid_body.config.solver.max_FEM_disp / np.max(Ug), 1.0])

        for nd in fe.Node.all():
            nd.move_coordinates(Ug[nd.dof[0], 0], Ug[nd.dof[1], 0])

        for ss in cls.__all:
            ss.update_geometry()

    def __init__(
        self,
        *,
        pretension: float = -0.5,
        axial_stiffness: float = 5e7,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.__all.append(self)
        self.pretension = pretension
        self.EA = axial_stiffness

        self.K: np.ndarray | None = None
        self.F: np.ndarray | None = None
        self.U: np.ndarray | None = None

    def get_residual(self) -> float:
        # TODO: Consider removing this and fixing static types
        assert self.U is not None
        return np.max(np.abs(self.U))

    def initialize_matrices(self) -> None:
        num_dof = fe.Node.count() * NUM_DIM
        self.K = np.zeros((num_dof, num_dof))
        self.F = np.zeros((num_dof, 1))
        self.U = np.zeros((num_dof, 1))

    def assemble_global_stiffness_and_force(self) -> None:
        if self.K is None or self.F is None:
            self.initialize_matrices()
        else:
            self.K *= 0
            self.F *= 0
        for el in self.el:
            self.add_loads_from_element(el)

    def add_loads_from_element(self, el: "fe.Element") -> None:
        assert isinstance(el, fe.TrussElement)
        assert self.K is not None
        assert self.F is not None
        K, F = el.get_stiffness_and_force()
        self.K[np.ix_(el.dof, el.dof)] += K
        self.F[np.ix_(el.dof)] += F

    #  def getPtDispFEM(self):
    # if self.K is None:
    # self.initializeMatrices()
    #    self.U *= 0.0
    # self.update_fluid_forces()
    # self.assembleGlobalStiffnessAndForce()
    #
    #    dof = [False for dof in self.F]
    #    for nd in self.node:
    #      for dofi, fdofi in zip(nd.dof, nd.fixedDOF):
    #        dof[dofi] = not fdofi
    # if any(dof):
    #      self.U[np.ix_(dof)] = np.linalg.solve(self.K[np.ix_(dof,dof)], self.F[np.ix_(dof)])
    #
    #    # Relax displacement and limit step if necessary
    #    self.U *= config.relaxFEM
    #    self.U *= np.min([config.maxFEMDisp / np.max(self.U), 1.0])
    #
    #    for nd in self.node:
    #      nd.moveCoordinates(self.U[nd.dof[0],0], self.U[nd.dof[1],0])
    #
    #    self.update_geometry()

    def set_element_properties(self) -> None:
        super().set_element_properties()
        for el in self.el:
            el.set_properties(axialForce=-self.pretension, EA=self.EA)

    def set_fixed_dof(self) -> None:
        pass

    def update_geometry(self) -> None:
        for el in self.el:
            el.update_geometry()
        super().set_interp_function()


class RigidSubstructure(Substructure):
    _element_type: ClassVar[type[fe.Element]] = fe.RigidElement

    def set_attachments(self) -> None:
        return None

    def update_angle(self) -> None:
        return None

    def set_fixed_dof(self) -> None:
        """Set all degrees of freedom of all nodes in the substructure."""
        for nd in self.node:
            for j in range(NUM_DIM):
                nd.is_dof_fixed[j] = True


class TorsionalSpringSubstructure(FlexibleSubstructure, RigidSubstructure):
    base_pt: np.ndarray
    is_free = True
    _element_type: ClassVar[type[fe.Element]] = fe.RigidElement

    def __init__(
        self,
        *,
        initial_angle: float = 0.0,
        tip_load_pct: float = 0.0,
        base_pt_pct: float = 1.0,
        spring_constant: float = 1e3,
        relaxation_angle: float | None = None,
        attach_pct: float = 0.0,
        minimum_angle: float = -float("Inf"),
        max_angle_step: float = float("Inf"),
        attached_substructure_name: str | None = None,
        attached_substructure_end: Literal["start"] | Literal["end"] = "end",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.initial_angle = initial_angle
        self.tip_load_pct = tip_load_pct
        self.base_pt_pct = base_pt_pct
        self.spring_constant = spring_constant

        self.theta = 0.0
        self.Mt = 0.0  # TODO
        self.MOld: float | None = None
        self.relax = relaxation_angle or self.config.body.relax_rigid_body
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

    def load_mesh(self, submesh: Path | Submesh = Path("mesh")) -> None:
        super().load_mesh(submesh)
        self.set_fixed_dof()
        if self.base_pt_pct == 1.0:
            self.base_pt = self.node[-1].get_coordinates()
        elif self.base_pt_pct == 0.0:
            self.base_pt = self.node[0].get_coordinates()
        else:
            self.base_pt = self.get_coordinates(self.base_pt_pct * self.arc_length)

        self.set_element_properties()

        self.set_angle(self.initial_angle)

    def set_attachments(self) -> None:
        if self.attached_substructure_name is not None:
            self.attached_substructure = Substructure.find_by_name(self.attached_substructure_name)
        else:
            self.attached_substructure = None

        if self.attached_substructure_end == "start":
            self.attached_ind = 0
        else:
            self.attached_ind = -1

        if self.attached_node is None and self.attached_substructure is not None:
            self.attached_node = self.attached_substructure.node[self.attached_ind]
            self.attached_element = self.attached_substructure.el[self.attached_ind]

    def update_fluid_forces(self) -> None:
        fluidS: list[float] = []
        fluidP: list[float] = []
        airS: list[float] = []
        airP: list[float] = []
        self.D = 0.0
        self.L = 0.0
        self.M = 0.0
        self.Dt = 0.0
        self.Lt = 0.0
        self.Mt = 0.0
        self.Da = 0.0
        self.La = 0.0
        self.Ma = 0.0
        if self.interpolator is not None:
            s_min, s_max = self.interpolator.get_min_max_s()

        for i, el in enumerate(self.el):
            # Get pressure at end points and all fluid points along element
            node_s = [self.node_arc_length[i], self.node_arc_length[i + 1]]
            if self.interpolator is not None:
                s, pressure_fluid, tau = self.interpolator.get_loads_in_range(node_s[0], node_s[1])

                # Limit pressure to be below stagnation pressure
                if self.config.plotting.pressure_limiter:
                    pressure_fluid = np.min(
                        np.hstack(
                            (
                                pressure_fluid,
                                np.ones_like(pressure_fluid) * self.config.flow.stagnation_pressure,
                            )
                        ),
                        axis=0,
                    )

            else:
                s = np.array(node_s)
                pressure_fluid = np.zeros_like(s)
                tau = np.zeros_like(s)

            ss = node_s[1]
            Pc = 0.0
            if self.interpolator is not None:
                if ss > s_max:
                    Pc = self.interpolator.fluid.upstream_pressure
                elif ss < s_min:
                    Pc = self.interpolator.fluid.downstream_pressure
            elif self.cushion_pressure_type == "Total":
                Pc = self.external_pressure or self.config.body.Pc

            # Store fluid and air pressure components for element (for
            # plotting)
            if i == 0:
                fluidS += [s[0]]
                fluidP += [pressure_fluid[0]]
                airS += [node_s[0]]
                airP += [Pc - self.seal_pressure]

            fluidS += [ss for ss in s[1:]]
            fluidP += [pp for pp in pressure_fluid[1:]]
            airS += [ss for ss in node_s[1:]]
            airP += [Pc - self.seal_pressure for _ in node_s[1:]]

            # Apply ramp to hydrodynamic pressure
            pressure_fluid *= self.ramp**2

            # Add external cushion pressure to external fluid pressure
            pressure_cushion = np.zeros_like(s)
            Pc = 0.0
            for ii, ss in enumerate(s):
                if self.interpolator is not None:
                    if ss > s_max:
                        Pc = self.interpolator.fluid.upstream_pressure
                    elif ss < s_min:
                        Pc = self.interpolator.fluid.downstream_pressure
                elif self.cushion_pressure_type == "Total":
                    Pc = self.config.body.Pc

                pressure_cushion[ii] = Pc

            pressure_internal = self.seal_pressure * np.ones_like(s)

            pressure_external = pressure_fluid + pressure_cushion
            pressure_total = pressure_external - pressure_internal

            # Calculate external force and moment for rigid body calculation
            if (
                self.config.body.cushion_force_method.lower() == "integrated"
                or self.config.body.cushion_force_method.lower() == "assumed"
            ):
                if self.config.body.cushion_force_method.lower() == "integrated":
                    integrand = pressure_external
                elif self.config.body.cushion_force_method.lower() == "assumed":
                    integrand = pressure_fluid

                n = list(map(self.get_normal_vector, s))
                t = [trig.rotate_vec_2d(ni, -90) for ni in n]

                fC = [
                    -pi * ni + taui * ti for pi, taui, ni, ti in zip(pressure_external, tau, n, t)
                ]
                fFl = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(pressure_fluid, tau, n, t)]
                f = fC + fFl
                print(("Cushion Lift-to-Weight: {0}".format(fC[1] / self.config.body.weight)))

                r = [
                    np.array([pt[0] - self.config.body.xCofR, pt[1] - self.config.body.yCofR])
                    for pt in map(self.get_coordinates, s)
                ]

                m = [math_helpers.cross2(ri, fi) for ri, fi in zip(r, f)]

                self.D -= math_helpers.integrate(s, np.array(list(zip(*f))[0]))
                self.L += math_helpers.integrate(s, np.array(list(zip(*f))[1]))
                self.M += math_helpers.integrate(s, np.array(m))
            else:
                if self.interpolator is not None:
                    self.D = self.interpolator.fluid.drag_total
                    self.L = self.interpolator.fluid.lift_total
                    self.M = self.interpolator.fluid.moment_total

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

            self.Dt += math_helpers.integrate(s, np.array(fx))
            self.Lt += math_helpers.integrate(s, np.array(fy))
            self.Mt += math_helpers.integrate(s, np.array(m))

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

            self.Da -= math_helpers.integrate(s, np.array(list(zip(*f))[0]))
            self.La += math_helpers.integrate(s, np.array(list(zip(*f))[1]))
            self.Ma += math_helpers.integrate(s, np.array(m))

        # Apply tip load
        tipC = self.get_coordinates(self.tip_load_pct * self.arc_length)
        tipR = np.array([tipC[i] - self.base_pt[i] for i in [0, 1]])
        tipF = np.array([0.0, self.tip_load]) * self.ramp
        tipM = math_helpers.cross2(tipR, tipF)
        self.Lt += tipF[1]
        self.Mt += tipM
        self.fluidP = np.array(fluidP)
        self.fluidS = np.array(fluidS)
        self.airP = np.array(airP)
        self.airS = np.array(airS)

        # Apply moment from attached substructure

    #    el = self.attachedEl
    #    attC = self.attachedNode.get_coordinates()
    #    attR = np.array([attC[i] - self.basePt[i] for i in [0,1]])
    #    attF = el.axialForce * kp.ang2vec(el.gamma + 180)
    #    attM = kp.cross2(attR, attF) * config.ramp
    #    attM = np.min([np.abs(attM), np.abs(self.Mt)]) * kp.sign(attM)
    # if np.abs(attM) > 2 * np.abs(tipM):
    #      attM = attM * np.abs(tipM) / np.abs(attM)
    #    self.Mt += attM

    def update_angle(self) -> None:

        if np.isnan(self.Mt):
            theta = 0.0
        else:
            theta = -self.Mt

        if not self.spring_constant == 0.0:
            theta /= self.spring_constant

        dTheta = (theta - self.theta) * self.relax
        dTheta = np.min([np.abs(dTheta), self.max_angle_step]) * np.sign(dTheta)

        self.set_angle(self.theta + dTheta)

    def set_angle(self, ang: float) -> None:
        dTheta = np.max([ang, self.minimum_angle]) - self.theta

        if self.attached_node is not None and not any(
            [nd == self.attached_node for nd in self.node]
        ):
            attNd = [self.attached_node]
        else:
            attNd = []

        #    basePt = np.array([c for c in self.basePt])
        basePt = np.array([c for c in self.node[-1].get_coordinates()])
        for nd in self.node + attNd:
            oldPt = np.array([c for c in nd.get_coordinates()])
            newPt = trig.rotate_point(oldPt, basePt, -dTheta)
            nd.set_coordinates(newPt[0], newPt[1])

        self.theta += dTheta
        self.residual = dTheta
        self.update_geometry()
        print(("  Deformation for substructure {0}: {1}".format(self.name, self.theta)))

    def get_residual(self) -> float:
        return self.residual

    #    return self.theta + self.Mt / self.spring_constant

    def write_deformation(self) -> None:
        writers.write_as_dict(
            self.it_dir / f"deformation_{self.name}.{self.config.io.data_format}",
            ["angle", self.theta],
        )


class Interpolator:
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

        self.solid.interpolator = self
        self.fluid.interpolator = self

        self.solid_position_function: Callable[[float], np.ndarray] = solid.get_coordinates
        self.fluid_pressure_function: Callable[
            [float, float], tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = fluid.get_loads_in_range
        self.get_body_height = self.get_surface_height_fixed_x

        self._separation_arclength: float | None = None
        self._immersed_arclength: float | None = None

        self._waterline_height = waterline_height
        self._separation_arclength_start_pct = separation_arclength_start_pct
        self._immersion_arclength_start_pct = immersion_arclength_start_pct

    def get_surface_height_fixed_x(self, x: float) -> float:
        s = np.max([self.get_s_fixed_x(x, 0.5), 0.0])
        return self.get_coordinates(s)[1]

    def get_coordinates(self, s: float) -> np.ndarray:
        assert self.solid_position_function is not None
        return self.solid_position_function(s)

    def get_min_max_s(self) -> list[float]:
        pts = self.fluid.get_element_coords()
        return [self.get_s_fixed_x(x) for x in [pts[0], pts[-1]]]

    def get_s_fixed_x(self, x: float, so_pct: float = 0.5) -> float:
        return fzero(lambda s: self.get_coordinates(s)[0] - x, so_pct * self.solid.arc_length)

    def get_s_fixed_y(self, y: float, so_pct: float) -> float:
        return fzero(lambda s: self.get_coordinates(s)[1] - y, so_pct * self.solid.arc_length)

    @property
    def immersed_length(self) -> float:
        if self._immersed_arclength is None:
            self._immersed_arclength = self._immersion_arclength_start_pct * self.solid.arc_length

        self._immersed_arclength = fzero(
            lambda s: self.get_coordinates(s)[1] - self._waterline_height,
            self._immersed_arclength,
        )

        return self.get_coordinates(self._immersed_arclength)[0]

    def get_separation_point(self) -> np.ndarray:
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

    def get_loads_in_range(self, s0: float, s1: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.solid_position_function is not None
        assert self.fluid_pressure_function is not None
        xs = [self.solid_position_function(s)[0] for s in [s0, s1]]
        x, p, tau = self.fluid_pressure_function(xs[0], xs[1])
        s = np.array([self.get_s_fixed_x(xx, 0.5) for xx in x])
        return s, p, tau
