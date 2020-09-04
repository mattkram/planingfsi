from typing import Dict, Any, List, Optional, Callable

import numpy as np
from planingfsi.solver import RootFinder

from . import felib as fe
from . import structure
from . import substructure
from .. import config, general, logger, trig, solver
from ..dictionary import load_dict_from_file
from ..general import write_as_dict


class RigidBody:
    # TODO: Yuck
    max_draft_step: float
    max_trim_step: float
    free_in_draft: bool
    free_in_trim: bool
    draft_damping: float
    trim_damping: float
    max_draft_acc: float
    max_trim_acc: float
    xCofG: float
    yCofG: float
    xCofR: float
    yCofR: float
    initial_draft: float
    initial_trim: float
    relax_draft: float
    relax_trim: float
    time_step: float
    num_damp: int

    def __init__(self, dict_: Dict[str, Any], parent: "structure.StructuralSolver"):
        self.parent = parent
        self.num_dim = 2
        self.draft = 0.0
        self.trim = 0.0

        self.name = dict_.get("bodyName", "default")
        self.weight = (
            dict_.get("W", dict_.get("loadPct", 1.0) * config.body.weight)
            * config.body.seal_load_pct
        )
        self.m = dict_.get("m", self.weight / config.flow.gravity)
        self.Iz = dict_.get("Iz", self.m * config.body.reference_length ** 2 / 12)
        self.has_planing_surface = dict_.get("hasPlaningSurface", False)

        var = [
            "max_draft_step",
            "max_trim_step",
            "free_in_draft",
            "free_in_trim",
            "draft_damping",
            "trim_damping",
            "max_draft_acc",
            "max_trim_acc",
            "xCofG",
            "yCofG",
            "xCofR",
            "yCofR",
            "initial_draft",
            "initial_trim",
            "relax_draft",
            "relax_trim",
            "time_step",
            "num_damp",
        ]
        for v in var:
            if v in dict_:
                setattr(self, v, dict_.get(v))
            elif hasattr(config.body, v):
                setattr(self, v, getattr(config.body, v))
            elif hasattr(config.solver, v):
                setattr(self, v, getattr(config.solver, v))
            elif hasattr(config, v):
                setattr(self, v, getattr(config, v))
            else:
                raise ValueError("Cannot find symbol: {0}".format(v))
            # setattr(self, v, dict_.read(v, getattr(config.body, getattr(config, v))))

        #    self.xCofR = dict_.read('xCofR', self.xCofG)
        #    self.yCofR = dict_.read('yCofR', self.yCofG)

        self.xCofR0 = self.xCofR
        self.yCofR0 = self.yCofR

        self.max_disp = np.array([self.max_draft_step, self.max_trim_step])
        self.free_dof = np.array([self.free_in_draft, self.free_in_trim])
        self.c_damp = np.array([self.draft_damping, self.trim_damping])
        self.max_acc = np.array([self.max_draft_acc, self.max_trim_acc])
        self.relax = np.array([self.relax_draft, self.relax_trim])

        # TODO: The config.has_free_structure variable should be factored out
        if self.free_in_draft or self.free_in_trim:
            config.has_free_structure = True

        self.v = np.zeros((self.num_dim,))
        self.a = np.zeros((self.num_dim,))
        self.v_old = np.zeros((self.num_dim,))
        self.a_old = np.zeros((self.num_dim,))

        self.beta = dict_.get("beta", 0.25)
        self.gamma = dict_.get("gamma", 0.5)

        self.D = 0.0
        self.L = 0.0
        self.M = 0.0
        self.Da = 0.0
        self.La = 0.0
        self.Ma = 0.0

        self.solver: Optional[solver.RootFinder] = None
        self.disp_old: Optional[np.ndarray] = None
        self.res_old: Optional[np.ndarray] = None
        self.two_ago_disp: Optional[np.ndarray] = None
        self.predictor = True
        self.f_old: Optional[np.ndarray] = None
        self.two_ago_f: Optional[np.ndarray] = None
        self.res_l = 1.0
        self.res_m = 1.0

        self.J: Optional[np.ndarray] = None
        self.J_tmp: Optional[np.ndarray] = None
        self.Jfo: Optional[np.ndarray] = None
        self.Jit = 0
        self.x: Optional[np.ndarray] = None
        self.f: Optional[np.ndarray] = None
        self.step = 0
        self.resFun: Optional[Callable[[np.ndarray], np.ndarray]] = None

        # Assign displacement function depending on specified method
        self.get_disp = lambda: (0.0, 0.0)
        if any(self.free_dof):
            if config.body.motion_method == "Secant":
                self.get_disp = self.get_disp_secant
            elif config.body.motion_method == "Broyden":
                self.get_disp = self.get_disp_broyden
            elif config.body.motion_method == "BroydenNew":
                self.get_disp = self.get_disp_broyden_new
            elif config.body.motion_method == "Physical":
                self.get_disp = self.get_disp_physical
            elif config.body.motion_method == "Newmark-Beta":
                self.get_disp = self.get_disp_newmark_beta
            elif config.body.motion_method == "PhysicalNoMass":
                self.get_disp = self.get_disp_physical_no_mass
            elif config.body.motion_method == "Sep":
                self.get_disp = self.get_disp_secant
                self.trim_solver = None
                self.draft_solver = None

        self.substructure: List["substructure.Substructure"] = []
        self.node: List[fe.Node] = []

        print(("Adding Rigid Body: {0}".format(self.name)))

    def add_substructure(self, ss: "substructure.Substructure") -> None:
        """Add a substructure to the rigid body."""
        self.substructure.append(ss)
        ss.parent = self

    def store_nodes(self) -> None:
        """Store references to all nodes in each substructure."""
        for ss in self.substructure:
            for nd in ss.node:
                if not any([n.node_num == nd.node_num for n in self.node]):
                    self.node.append(nd)

    def initialize_position(self) -> None:
        """Initialize the position of the rigid body."""
        self.set_position(self.initial_draft, self.initial_trim)

    def set_position(self, draft: float, trim: float) -> None:
        """Set the position of the rigid body."""
        self.update_position(draft - self.draft, trim - self.trim)

    def update_position(self, draft_delta: float = None, trim_delta: float = None) -> None:
        """Update the position of the rigid body by passing the change in draft and trim."""
        if draft_delta is None or trim_delta is None:
            draft_delta, trim_delta = self.get_disp()
            if np.isnan(draft_delta):
                draft_delta = 0.0
            if np.isnan(trim_delta):
                trim_delta = 0.0

        if not self.node:
            self.store_nodes()

        for nd in self.node:
            xo, yo = nd.get_coordinates()
            new_pos = general.rotate_point(
                np.array([xo, yo]), np.array([self.xCofR, self.yCofR]), trim_delta
            )
            nd.move_coordinates(new_pos[0] - xo, new_pos[1] - yo - draft_delta)

        for s in self.substructure:
            s.update_geometry()

        self.xCofG, self.yCofG = general.rotate_point(
            np.array([self.xCofG, self.yCofG]), np.array([self.xCofR, self.yCofR]), trim_delta
        )
        self.yCofG -= draft_delta
        self.yCofR -= draft_delta

        self.draft += draft_delta
        self.trim += trim_delta

        self.print_motion()

    def update_substructure_positions(self) -> None:
        """Update the positions of all substructures."""
        substructure.FlexibleSubstructure.update_all()
        for ss in self.substructure:
            logger.info(f"Updating position for substructure: {ss.name}")
            if isinstance(ss, substructure.RigidSubstructure):
                ss.update_angle()

    def update_fluid_forces(self) -> None:
        """Update the fluid forces by summing the force from each substructure."""
        self.reset_loads()
        for ss in self.substructure:
            ss.update_fluid_forces()
            self.D += ss.D
            self.L += ss.L
            self.M += ss.M
            self.Da += ss.Da
            self.La += ss.La
            self.Ma += ss.Ma

        self.res_l = self.get_res_lift()
        self.res_m = self.get_res_moment()

    def reset_loads(self) -> None:
        """Reset the loads to zero."""
        self.D *= 0.0
        self.L *= 0.0
        self.M *= 0.0
        if config.body.cushion_force_method.lower() == "assumed":
            self.L += (
                config.body.Pc * config.body.reference_length * trig.cosd(config.body.initial_trim)
            )

    def get_disp_physical(self) -> np.ndarray:
        """Get the rigid body displacement using a time-domain model with inertia."""
        disp = self.limit_disp(self.time_step * self.v)

        for i in range(self.num_dim):
            if np.abs(disp[i]) == np.abs(self.max_disp[i]):
                self.v[i] = disp[i] / self.time_step

        self.v += self.time_step * self.a

        self.a = np.array([self.weight - self.L, self.M - self.weight * (self.xCofG - self.xCofR)])
        #    self.a -= self.Cdamp * (self.v + self.v**3) * config.ramp
        self.a -= self.c_damp * self.v * config.ramp
        self.a /= np.array([self.m, self.Iz])
        self.a = (
            np.min(
                np.vstack((np.abs(self.a), np.array([self.max_draft_acc, self.max_trim_acc]))),
                axis=0,
            )
            * np.sign(self.a)
        )

        #    accLimPct = np.min(np.vstack((np.abs(self.a), self.maxAcc)), axis=0) * np.sign(self.a)
        #    for i in range(len(self.a)):
        #      if self.a[i] == 0.0 or not self.freeDoF[i]:
        #        accLimPct[i] = 1.0
        #      else:
        #        accLimPct[i] /= self.a[i]
        #
        #    self.a *= np.min(accLimPct)
        disp *= config.ramp
        return disp

    def get_disp_newmark_beta(self) -> np.ndarray:
        """Get the rigid body displacement using the Newmark-Beta method."""
        self.a = np.array([self.weight - self.L, self.M - self.weight * (self.xCofG - self.xCofR)])
        #    self.a -= self.Cdamp * self.v * config.ramp
        self.a /= np.array([self.m, self.Iz])
        self.a = (
            np.min(
                np.vstack((np.abs(self.a), np.array([self.max_draft_acc, self.max_trim_acc]))),
                axis=0,
            )
            * np.sign(self.a)
        )

        dv = (1 - self.gamma) * self.a_old + self.gamma * self.a
        dv *= self.time_step
        dv *= 1 - self.num_damp
        self.v += dv

        disp = 0.5 * (1 - 2 * self.beta) * self.a_old + self.beta * self.a
        disp *= self.time_step
        disp += self.v_old
        disp *= self.time_step

        self.a_old = self.a
        self.v_old = self.v

        disp *= self.relax
        disp *= config.ramp
        disp = self.limit_disp(disp)
        #    disp *= config.ramp

        return disp

    def get_disp_physical_no_mass(self) -> np.ndarray:
        """Get the rigid body displacement using a time-domain model without inertia."""
        force_moment = np.array(
            [self.weight - self.L, self.M - self.weight * (config.body.xCofG - config.body.xCofR)]
        )
        #    F -= self.Cdamp * self.v

        if self.predictor:
            disp = force_moment / self.c_damp * self.time_step
            self.predictor = False
        else:
            disp = 0.5 * self.time_step / self.c_damp * (force_moment - self.two_ago_f)
            self.predictor = True

        disp *= self.relax * config.ramp
        disp = self.limit_disp(disp)

        #    self.v = disp / self.timeStep

        self.two_ago_f = self.f_old
        self.f_old = disp * self.c_damp / self.time_step

        return disp

    def get_disp_secant(self) -> np.ndarray:
        """Get the rigid body displacement using the Secant method."""
        if self.solver is None:
            self.resFun = lambda x: np.array(
                [
                    self.L - self.weight,
                    self.M - self.weight * (config.body.xCofG - config.body.xCofR),
                ]
            )
            self.solver = solver.RootFinder(
                self.resFun,
                np.array([config.body.initial_draft, config.body.initial_trim]),
                "secant",
                dxMax=self.max_disp * self.free_dof,
            )

        if self.disp_old is not None:
            self.solver.take_step(self.disp_old)

        # Solve for limited displacement
        disp = self.solver.limit_step(self.solver.get_step())

        self.two_ago_disp = self.disp_old
        self.disp_old = disp

        return disp

    def reset_jacobian(self) -> np.ndarray:
        """Reset the solver Jacobian and modify displacement."""
        assert self.resFun is not None
        if self.J_tmp is None:
            self.Jit = 0
            self.J_tmp = np.zeros((self.num_dim, self.num_dim))
            self.step = 0
            self.Jfo = self.resFun(self.x)
            self.res_old = self.Jfo * 1.0
        else:
            f = self.resFun(self.x)
            assert isinstance(self.disp_old, np.ndarray)
            self.J_tmp[:, self.Jit] = (f - self.Jfo) / self.disp_old[self.Jit]
            self.Jit += 1

        disp = np.zeros((self.num_dim,))
        if self.Jit < self.num_dim:
            disp[self.Jit] = float(config.body.motion_jacobian_first_step)

        if self.Jit > 0:
            disp[self.Jit - 1] = -float(config.body.motion_jacobian_first_step)
        self.disp_old = disp
        if self.Jit >= self.num_dim:
            assert isinstance(self.J, np.ndarray)
            self.J[:] = self.J_tmp
            self.J_tmp = None
            self.disp_old = None

        return disp

    def get_disp_broyden_new(self) -> np.ndarray:
        """Get the rigid body displacement using Broyden's method.

        Todo:
            There should only be one Broyden's method

        """
        if self.solver is None:
            self.resFun = lambda x: np.array(
                [
                    f
                    for f, free_dof in zip(
                        [self.get_res_lift(), self.get_res_moment()], self.free_dof
                    )
                    if free_dof
                ]
            )
            max_disp = [m for m, free_dof in zip(self.max_disp, self.free_dof) if free_dof]

            self.x = np.array(
                [f for f, free_dof in zip([self.draft, self.trim], self.free_dof) if free_dof]
            )
            self.solver = solver.RootFinder(self.resFun, self.x, "broyden", dxMax=max_disp)

        if self.disp_old is not None:
            self.solver.take_step(self.disp_old)

        # Solve for limited displacement
        disp = self.solver.limit_step(self.solver.get_step())

        self.two_ago_disp = self.disp_old
        self.disp_old = disp
        return disp

    def get_disp_broyden(self) -> np.ndarray:
        """Get the rigid body displacement using Broyden's method."""
        if self.solver is None:
            self.resFun = lambda x: np.array(
                [self.L - self.weight, self.M - self.weight * (self.xCofG - self.xCofR)]
            )
            #      self.resFun = lambda x: np.array([self.get_res_moment(), self.get_res_lift()])
            self.x = np.array([self.draft, self.trim])
            self.solver = RootFinder(self.resFun, self.x, method="Broyden")

        if self.J is None:
            disp = self.reset_jacobian()
        else:
            assert self.resFun is not None
            self.f = self.resFun(self.x)
            if self.disp_old is not None:
                self.x += self.disp_old

                dx = np.reshape(self.disp_old, (self.num_dim, 1))
                df = np.reshape(self.f - self.res_old, (self.num_dim, 1))

                self.J += np.dot(df - np.dot(self.J, dx), dx.T) / np.linalg.norm(dx) ** 2

            dof = self.free_dof
            dx = np.zeros_like(self.x)
            dx[np.ix_(dof)] = np.linalg.solve(
                -self.J[np.ix_(dof, dof)], self.f.reshape(self.num_dim, 1)[np.ix_(dof)]
            )

            if self.res_old is not None:
                if any(np.abs(self.f) - np.abs(self.res_old) > 0.0):
                    self.step += 1

            if self.step >= 6:
                print("\nResetting Jacobian for Motion\n")
                self.reset_jacobian()

            disp = dx.reshape(self.num_dim)

            disp *= self.relax
            disp = self.limit_disp(disp)

            self.disp_old = disp

            self.res_old = self.f * 1.0
            self.step += 1
        return disp

    def limit_disp(self, disp: np.ndarray) -> np.ndarray:
        """Limit the body displacement."""
        disp_lim_pct = np.min(np.vstack((np.abs(disp), self.max_disp)), axis=0) * np.sign(disp)
        for i in range(len(disp)):
            if disp[i] == 0.0 or not self.free_dof[i]:
                disp_lim_pct[i] = 1.0
            else:
                disp_lim_pct[i] /= disp[i]

        return disp * np.min(disp_lim_pct) * self.free_dof

    def get_res_lift(self) -> float:
        """Get the residual of the vertical force balance."""
        if np.isnan(self.L):
            res = 1.0
        else:
            res = (self.L - self.weight) / (
                config.flow.stagnation_pressure * config.body.reference_length + 1e-6
            )
        return np.abs(res * self.free_in_draft)

    def get_res_moment(self) -> float:
        """Get the residual of the trim moment balance."""
        if np.isnan(self.M):
            res = 1.0
        else:
            if self.xCofG == self.xCofR and self.M == 0.0:
                res = 1.0
            else:
                res = (self.M - self.weight * (self.xCofG - self.xCofR)) / (
                    config.flow.stagnation_pressure * config.body.reference_length ** 2 + 1e-6
                )
        return np.abs(res * self.free_in_trim)

    def print_motion(self) -> None:
        """Print the moment for debugging."""
        lines = [
            f"Rigid Body Motion: {self.name}",
            f"  CofR: ({self.xCofR}, {self.yCofR})",
            f"  CofG: ({self.xCofG}, {self.yCofG})",
            f"  Draft:      {self.draft:5.4e}",
            f"  Trim Angle: {self.trim:5.4e}",
            f"  Lift Force: {self.L:5.4e}",
            f"  Drag Force: {self.D:5.4e}",
            f"  Moment:     {self.M:5.4e}",
            f"  Lift Force Air: {self.La:5.4e}",
            f"  Drag Force Air: {self.Da:5.4e}",
            f"  Moment Air:     {self.Ma:5.4e}",
            f"  Lift Res:   {self.res_l:5.4e}",
            f"  Moment Res: {self.res_m:5.4e}",
        ]
        for line in lines:
            logger.debug(line)

    def write_motion(self) -> None:
        """Write the motion results to file."""
        write_as_dict(
            self.parent.simulation.it_dir / f"motion_{self.name}.{config.io.data_format}",
            ["xCofR", self.xCofR],
            ["yCofR", self.yCofR],
            ["xCofG", self.xCofG],
            ["yCofG", self.yCofG],
            ["draft", self.draft],
            ["trim", self.trim],
            ["liftRes", self.res_l],
            ["momentRes", self.res_m],
            ["Lift", self.L],
            ["Drag", self.D],
            ["Moment", self.M],
            ["LiftAir", self.La],
            ["DragAir", self.Da],
            ["MomentAir", self.Ma],
        )
        for ss in self.substructure:
            if isinstance(ss, substructure.TorsionalSpringSubstructure):
                ss.write_deformation()

    def load_motion(self) -> None:
        dict_ = load_dict_from_file(
            self.parent.simulation.it_dir / f"motion_{self.name}.{config.io.data_format}"
        )
        self.xCofR = dict_.get("xCofR", np.nan)
        self.yCofR = dict_.get("yCofR", np.nan)
        self.xCofG = dict_.get("xCofG", np.nan)
        self.yCofG = dict_.get("yCofG", np.nan)
        self.draft = dict_.get("draft", np.nan)
        self.trim = dict_.get("trim", np.nan)
        self.res_l = dict_.get("liftRes", np.nan)
        self.res_m = dict_.get("momentRes", np.nan)
        self.L = dict_.get("Lift", np.nan)
        self.D = dict_.get("Drag", np.nan)
        self.M = dict_.get("Moment", np.nan)
        self.La = dict_.get("LiftAir", np.nan)
        self.Da = dict_.get("DragAir", np.nan)
        self.Ma = dict_.get("MomentAir", np.nan)
