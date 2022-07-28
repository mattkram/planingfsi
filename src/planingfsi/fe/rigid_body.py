from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from planingfsi import logger
from planingfsi import solver
from planingfsi import trig
from planingfsi import writers
from planingfsi.config import NUM_DIM
from planingfsi.config import Config
from planingfsi.dictionary import load_dict_from_file
from planingfsi.fe import substructure
from planingfsi.solver import RootFinder

if TYPE_CHECKING:
    from planingfsi.fe import felib as fe
    from planingfsi.fe.structure import StructuralSolver


class RigidBody:
    """A rigid body, consisting of multiple substructures.

    The rigid body is used as a container for collecting attached substructures, and also
    allows motion in heave & trim, which is calculated by integrating the total forces
    and moments applied to the substructures.

    Args:
        name: A name for the RigidBody instance
        weight: If provided, the weight to use for the rigid body (scaled by `config.body.seal_load_pct`).
        load_pct: If `weight` is not provided, this value is multiplied by the global `config.body.weight`,
            which is then also scaled by `config.body.seal_load_pct`.
        mass: An optional override for the mass. Otherwise, mass = weight / gravity.
        rotational_inertia: An optional override for the rotational moment of inertia. Otherwise, will be
            approximated from the mass and body reference length.
        x_cg: The x-coordinate of the center of gravity.
        y_cg: The y-coordinate of the center of gravity.
        x_cr: The x-coordinate of the center of rotation. Defaults to the CG.
        y_cr: The y-coordinate of the center of rotation. Defaults to the CG.
        initial_draft: The initial draft of the body.
        initial_trim: The initial trim of the body.
        relax_draft: The relaxation parameter to use for draft.
        relax_trim: The relaxation parameter to use for trim.
        time_step: The time step to use when quasi-time-domain method is used.
        num_damp: Numerical damping coefficient when quasi-time-domain method is used.
        max_draft_step: The maximum allowable change in draft during an iteration.
        max_trim_step: The maximum allowable change in trim during an iteration.
        free_in_draft: True if body is free to move in draft.
        free_in_trim: True if body is free to move in trim.
        draft_damping: Damping coefficient to use for draft.
        trim_damping: Damping coefficient to use for trim.
        max_draft_acc: The maximum draft acceleration when quasi-time-domain method is used.
        max_trim_acc: The maximum trim acceleration when quasi-time-domain method is used.
        beta: The beta parameter when using Newmark-Beta method.
        gamma: The gamma parameter when using Newmark-Beta method.
        parent: An optional reference to the parent structural solver.

    """

    def __init__(
        self,
        name: str = "default",
        weight: float | None = None,
        load_pct: float = 1.0,
        mass: float | None = None,
        rotational_inertia: float | None = None,
        x_cg: float | None = None,
        y_cg: float | None = None,
        x_cr: float | None = None,
        y_cr: float | None = None,
        initial_draft: float | None = None,
        initial_trim: float | None = None,
        relax_draft: float | None = None,
        relax_trim: float | None = None,
        time_step: float | None = None,
        num_damp: float | None = None,
        max_draft_step: float | None = None,
        max_trim_step: float | None = None,
        free_in_draft: bool | None = None,
        free_in_trim: bool | None = None,
        draft_damping: float | None = None,
        trim_damping: float | None = None,
        max_draft_acc: float | None = None,
        max_trim_acc: float | None = None,
        beta: float = 0.25,
        gamma: float = 0.5,
        parent: StructuralSolver | None = None,
        **_: Any,
    ):
        self.parent = parent

        self.draft = 0.0
        self.trim = 0.0

        self.name = name

        if weight is not None:
            self.weight = weight
        else:
            self.weight = load_pct * self.config.body.weight

        self.weight *= self.config.body.seal_load_pct

        self.m = mass if mass is not None else self.weight / self.config.flow.gravity
        self.Iz = (
            rotational_inertia
            if rotational_inertia is not None
            else self.m * self.config.body.reference_length**2 / 12
        )

        # TODO: The override logic should probably happen in the input file parsing, and not here
        self.xCofG = x_cg or self.config.body.xCofG
        self.yCofG = y_cg or self.config.body.yCofG
        self.xCofR = x_cr or self.config.body.xCofR
        self.yCofR = y_cr or self.config.body.yCofR

        self.xCofR0 = self.xCofR
        self.yCofR0 = self.yCofR

        self.initial_draft = initial_draft or self.config.body.initial_draft
        self.initial_trim = initial_trim or self.config.body.initial_trim

        free_in_draft = free_in_draft or self.config.body.free_in_draft
        free_in_trim = free_in_trim or self.config.body.free_in_trim
        max_draft_step = max_draft_step or self.config.body.max_draft_step
        max_trim_step = max_trim_step or self.config.body.max_trim_step
        draft_damping = draft_damping or self.config.body.draft_damping
        trim_damping = trim_damping or self.config.body.trim_damping
        max_draft_acc = max_draft_acc or self.config.body.max_draft_acc
        max_trim_acc = max_trim_acc or self.config.body.max_trim_acc
        relax_draft = relax_draft or self.config.body.relax_draft
        relax_trim = relax_trim or self.config.body.relax_trim

        self.time_step = time_step or self.config.body.time_step
        self.free_dof = np.array([free_in_draft, free_in_trim])
        self.max_disp = np.array([max_draft_step, max_trim_step])
        self.c_damp = np.array([draft_damping, trim_damping])
        self.max_acc = np.array([max_draft_acc, max_trim_acc])
        self.relax = np.array([relax_draft, relax_trim])

        self.v = np.zeros((NUM_DIM,))
        self.a = np.zeros((NUM_DIM,))
        self.v_old = np.zeros((NUM_DIM,))
        self.a_old = np.zeros((NUM_DIM,))

        # Parameters for Newmark-Beta method
        self.beta = beta
        self.gamma = gamma
        self.num_damp = num_damp or self.config.solver.num_damp

        self.D = 0.0
        self.L = 0.0
        self.M = 0.0
        self.Da = 0.0
        self.La = 0.0
        self.Ma = 0.0

        self.solver: solver.RootFinder | None = None
        self.disp_old: np.ndarray | None = None
        self.res_old: np.ndarray | None = None
        self.two_ago_disp: np.ndarray | None = None
        self.predictor = True
        self.f_old: np.ndarray | None = None
        self.two_ago_f: np.ndarray | None = None
        self.res_l = 1.0
        self.res_m = 1.0

        self.J: np.ndarray | None = None
        self.J_tmp: np.ndarray | None = None
        self.Jfo: np.ndarray | None = None
        self.Jit = 0
        self.x: np.ndarray | None = None
        self.f: np.ndarray | None = None
        self.step = 0
        self.resFun: Callable[[np.ndarray], np.ndarray] | None = None

        # Assign displacement function depending on specified method
        self.get_disp = lambda: np.array((0.0, 0.0))
        if any(self.free_dof):
            if self.config.body.motion_method == "Secant":
                self.get_disp = self.get_disp_secant
            elif self.config.body.motion_method == "Broyden":
                self.get_disp = self.get_disp_broyden
            elif self.config.body.motion_method == "BroydenNew":
                self.get_disp = self.get_disp_broyden_new
            elif self.config.body.motion_method == "Physical":
                self.get_disp = self.get_disp_physical
            elif self.config.body.motion_method == "Newmark-Beta":
                self.get_disp = self.get_disp_newmark_beta
            elif self.config.body.motion_method == "PhysicalNoMass":
                self.get_disp = self.get_disp_physical_no_mass
            elif self.config.body.motion_method == "Sep":
                self.get_disp = self.get_disp_secant
                self.trim_solver = None
                self.draft_solver = None

        self.substructures: list["substructure.Substructure"] = []
        self.nodes: list[fe.Node] = []

    @property
    def config(self) -> Config:
        """A reference to the simulation configuration."""
        if self.parent is None:
            raise AttributeError("parent must be set before config can be accessed.")
        return self.parent.config

    @property
    def ramp(self) -> float:
        """The ramping coefficient from the high-level simulation object."""
        if self.parent is None:
            raise AttributeError("parent must be set before simulation can be accessed.")
        return self.parent.simulation.ramp

    @property
    def free_in_draft(self) -> bool:
        return self.free_dof[0]

    @free_in_draft.setter
    def free_in_draft(self, value: bool) -> None:
        self.free_dof[0] = value

    @property
    def free_in_trim(self) -> bool:
        return self.free_dof[1]

    @free_in_trim.setter
    def free_in_trim(self, value: bool) -> None:
        self.free_dof[1] = value

    def add_substructure(self, ss: "substructure.Substructure") -> substructure.Substructure:
        """Add a substructure to the rigid body."""
        self.substructures.append(ss)
        ss.parent = self
        return ss

    def store_nodes(self) -> None:
        """Store references to all nodes in each substructure."""
        for ss in self.substructures:
            for nd in ss.node:
                if not any([n.node_num == nd.node_num for n in self.nodes]):
                    self.nodes.append(nd)

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
            # TODO: Consider removing this and fixing static types
            assert draft_delta is not None
            assert trim_delta is not None
            if np.isnan(draft_delta):
                draft_delta = 0.0
            if np.isnan(trim_delta):
                trim_delta = 0.0

        if not self.nodes:
            self.store_nodes()

        for nd in self.nodes:
            xo, yo = nd.get_coordinates()
            new_pos = trig.rotate_point(
                np.array([xo, yo]), np.array([self.xCofR, self.yCofR]), trim_delta
            )
            nd.move_coordinates(new_pos[0] - xo, new_pos[1] - yo - draft_delta)

        for s in self.substructures:
            s.update_geometry()

        self.xCofG, self.yCofG = trig.rotate_point(
            np.array([self.xCofG, self.yCofG]), np.array([self.xCofR, self.yCofR]), trim_delta
        )
        self.yCofG -= draft_delta
        self.yCofR -= draft_delta

        self.draft += draft_delta
        self.trim += trim_delta

        self.print_motion()

    def update_substructure_positions(self) -> None:
        """Update the positions of all substructures."""
        substructure.FlexibleSubstructure.update_all(self)
        for ss in self.substructures:
            logger.info(f"Updating position for substructure: {ss.name}")
            if isinstance(ss, substructure.RigidSubstructure):
                ss.update_angle()

    def update_fluid_forces(self) -> None:
        """Update the fluid forces by summing the force from each substructure."""
        self.reset_loads()
        for ss in self.substructures:
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
        if self.config.body.cushion_force_method.lower() == "assumed":
            self.L += (
                self.config.body.Pc
                * self.config.body.reference_length
                * trig.cosd(self.config.body.initial_trim)
            )

    def get_disp_physical(self) -> np.ndarray:
        """Get the rigid body displacement using a time-domain model with inertia."""
        disp = self.limit_disp(self.time_step * self.v)

        for i in range(NUM_DIM):
            if np.abs(disp[i]) == np.abs(self.max_disp[i]):
                self.v[i] = disp[i] / self.time_step

        self.v += self.time_step * self.a

        self.a = np.array([self.weight - self.L, self.M - self.weight * (self.xCofG - self.xCofR)])
        #    self.a -= self.Cdamp * (self.v + self.v**3) * config.ramp
        self.a -= self.c_damp * self.v * self.ramp
        self.a /= np.array([self.m, self.Iz])
        self.a = np.min(
            np.vstack((np.abs(self.a), self.max_acc)),
            axis=0,
        ) * np.sign(self.a)

        #    accLimPct = np.min(np.vstack((np.abs(self.a), self.maxAcc)), axis=0) * np.sign(self.a)
        #    for i in range(len(self.a)):
        #      if self.a[i] == 0.0 or not self.freeDoF[i]:
        #        accLimPct[i] = 1.0
        #      else:
        #        accLimPct[i] /= self.a[i]
        #
        #    self.a *= np.min(accLimPct)
        disp *= self.ramp
        return disp

    def get_disp_newmark_beta(self) -> np.ndarray:
        """Get the rigid body displacement using the Newmark-Beta method."""
        self.a = np.array([self.weight - self.L, self.M - self.weight * (self.xCofG - self.xCofR)])
        #    self.a -= self.Cdamp * self.v * config.ramp
        self.a /= np.array([self.m, self.Iz])
        self.a = np.min(
            np.vstack((np.abs(self.a), self.max_acc)),
            axis=0,
        ) * np.sign(self.a)

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
        disp *= self.ramp
        disp = self.limit_disp(disp)

        return disp

    def get_disp_physical_no_mass(self) -> np.ndarray:
        """Get the rigid body displacement using a time-domain model without inertia."""
        force_moment = np.array(
            [
                self.weight - self.L,
                self.M - self.weight * (self.config.body.xCofG - self.config.body.xCofR),
            ]
        )
        #    F -= self.Cdamp * self.v

        if self.predictor:
            disp = force_moment / self.c_damp * self.time_step
            self.predictor = False
        else:
            disp = 0.5 * self.time_step / self.c_damp * (force_moment - self.two_ago_f)
            self.predictor = True

        disp *= self.relax * self.ramp
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
                    self.M - self.weight * (self.config.body.xCofG - self.config.body.xCofR),
                ]
            )
            self.solver = solver.RootFinder(
                self.resFun,
                np.array([self.config.body.initial_draft, self.config.body.initial_trim]),
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
        # TODO: These are here for mypy, fix the types instead
        assert self.resFun is not None
        assert self.x is not None
        if self.J_tmp is None:
            self.Jit = 0
            self.J_tmp = np.zeros((NUM_DIM, NUM_DIM))
            self.step = 0
            self.Jfo = self.resFun(self.x)
            self.res_old = self.Jfo * 1.0
        else:
            f = self.resFun(self.x)
            assert isinstance(self.disp_old, np.ndarray)
            self.J_tmp[:, self.Jit] = (f - self.Jfo) / self.disp_old[self.Jit]
            self.Jit += 1

        disp = np.zeros((NUM_DIM,))
        if self.Jit < NUM_DIM:
            disp[self.Jit] = float(self.config.body.motion_jacobian_first_step)

        if self.Jit > 0:
            disp[self.Jit - 1] = -float(self.config.body.motion_jacobian_first_step)
        self.disp_old = disp
        if self.Jit >= NUM_DIM:
            if self.J is None:
                self.J = self.J_tmp.copy()
            else:
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
        # TODO: These are here for mypy, fix the types instead
        assert self.resFun is not None
        assert self.x is not None

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

                dx = np.reshape(self.disp_old, (NUM_DIM, 1))
                df = np.reshape(self.f - self.res_old, (NUM_DIM, 1))

                self.J += np.dot(df - np.dot(self.J, dx), dx.T) / np.linalg.norm(dx) ** 2

            dof = self.free_dof
            dx = np.zeros_like(self.x)
            dx[np.ix_(dof)] = np.linalg.solve(
                -self.J[np.ix_(dof, dof)], self.f.reshape(NUM_DIM, 1)[np.ix_(dof)]
            ).flatten()  # TODO: Check that the flatten() is required

            if self.res_old is not None:
                if any(np.abs(self.f) - np.abs(self.res_old) > 0.0):
                    self.step += 1

            if self.step >= 6:
                print("\nResetting Jacobian for Motion\n")
                self.reset_jacobian()

            disp = dx.reshape(NUM_DIM)

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
                self.config.flow.stagnation_pressure * self.config.body.reference_length + 1e-6
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
                    self.config.flow.stagnation_pressure * self.config.body.reference_length**2
                    + 1e-6
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
        if self.parent is None:
            raise AttributeError("parent must be set before simulation can be accessed.")
        writers.write_as_dict(
            self.parent.simulation.it_dir / f"motion_{self.name}.{self.config.io.data_format}",
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
        for ss in self.substructures:
            if isinstance(ss, substructure.TorsionalSpringSubstructure):
                ss.write_deformation()

    def load_motion(self) -> None:
        if self.parent is None:
            raise AttributeError("parent must be set before simulation can be accessed.")
        dict_ = load_dict_from_file(
            self.parent.simulation.it_dir / f"motion_{self.name}.{self.config.io.data_format}"
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
