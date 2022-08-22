from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
from pathlib import Path
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
from planingfsi.fe import felib as fe
from planingfsi.fe import substructure
from planingfsi.solver import RootFinder

if TYPE_CHECKING:
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
        x_cg: The x-coordinate of the center of gravity.
        y_cg: The y-coordinate of the center of gravity.
        x_cr: The x-coordinate of the center of rotation. Defaults to the CG.
        y_cr: The y-coordinate of the center of rotation. Defaults to the CG.
        initial_draft: The initial draft of the body.
        initial_trim: The initial trim of the body.
        relax_draft: The relaxation parameter to use for draft.
        relax_trim: The relaxation parameter to use for trim.
        max_draft_step: The maximum allowable change in draft during an iteration.
        max_trim_step: The maximum allowable change in trim during an iteration.
        free_in_draft: True if body is free to move in draft.
        free_in_trim: True if body is free to move in trim.
        parent: An optional reference to the parent structural solver.

    """

    def __init__(
        self,
        name: str = "default",
        weight: float | None = None,
        load_pct: float = 1.0,
        x_cg: float | None = None,
        y_cg: float | None = None,
        x_cr: float | None = None,
        y_cr: float | None = None,
        initial_draft: float | None = None,
        initial_trim: float | None = None,
        relax_draft: float | None = None,
        relax_trim: float | None = None,
        max_draft_step: float | None = None,
        max_trim_step: float | None = None,
        free_in_draft: bool | None = None,
        free_in_trim: bool | None = None,
        parent: StructuralSolver | None = None,
        **_: Any,
    ):
        self.name = name
        self.parent = parent

        if weight is not None:
            self.weight = weight
        else:
            self.weight = load_pct * self.config.body.weight

        self.weight *= self.config.body.seal_load_pct

        # TODO: The override logic should probably happen in the input file parsing, and not here
        self.x_cg = x_cg or self.config.body.xCofG
        self.y_cg = y_cg or self.config.body.yCofG
        self.x_cr = x_cr or self.config.body.xCofR
        self.y_cr = y_cr or self.config.body.yCofR

        self.x_cr_init = self.x_cr
        self.y_cr_init = self.y_cr

        self.draft = 0.0
        self.trim = 0.0
        self.initial_draft = initial_draft or self.config.body.initial_draft
        self.initial_trim = initial_trim or self.config.body.initial_trim

        free_in_draft = free_in_draft or self.config.body.free_in_draft
        free_in_trim = free_in_trim or self.config.body.free_in_trim
        max_draft_step = max_draft_step or self.config.body.max_draft_step
        max_trim_step = max_trim_step or self.config.body.max_trim_step
        relax_draft = relax_draft or self.config.body.relax_draft
        relax_trim = relax_trim or self.config.body.relax_trim

        self.free_dof = np.array([free_in_draft, free_in_trim])
        self.max_disp = np.array([max_draft_step, max_trim_step])
        self.relax = np.array([relax_draft, relax_trim])

        self.D = 0.0
        self.L = 0.0
        self.M = 0.0
        self.Da = 0.0
        self.La = 0.0
        self.Ma = 0.0

        self.res_l = 1.0
        self.res_m = 1.0

        # Attributes related to solver
        self.solver: solver.RootFinder | None = None
        self.disp_old: np.ndarray | None = None
        self.res_old: np.ndarray | None = None
        self.J: np.ndarray | None = None
        self.J_tmp: np.ndarray | None = None
        self.Jfo: np.ndarray | None = None
        self.Jit = 0
        self.x: np.ndarray | None = None
        self.f: np.ndarray | None = None
        self.step = 0
        self.resFun: Callable[[np.ndarray], np.ndarray] | None = None

        # Assign displacement function depending on specified method
        self.flexible_substructure_residual = 0.0
        self.substructures: list["substructure.Substructure"] = []
        self._nodes: list[fe.Node] = []

    @cached_property
    def config(self) -> Config:
        """A reference to the simulation configuration."""
        if self.parent is None:
            return Config()
        return self.parent.config

    @property
    def ramp(self) -> float:
        """The ramping coefficient from the high-level simulation object."""
        if self.parent is None:
            return 1.0
        return self.parent.simulation.ramp

    @property
    def residual(self) -> float:
        """The combined max residual of lift, moment, and structural node displacement."""
        res_l = self.res_l if self.free_in_draft else 0.0
        res_m = self.res_m if self.free_in_trim else 0.0
        res_node_disp = self.flexible_substructure_residual
        res_torsion = max(
            [
                ss.residual
                for ss in self.substructures
                if isinstance(ss, substructure.TorsionalSpringSubstructure)
            ],
            default=0.0,
        )
        return max(res_l, res_m, res_node_disp, res_torsion)

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

    @property
    def motion_file_path(self) -> Path:
        """A path to the file containing the rigid body motinon at a given iteration."""
        return self.parent.simulation.it_dir / f"motion_{self.name}.{self.config.io.data_format}"

    @property
    def nodes(self) -> list[fe.Node]:
        """Get a list of all unique `Node`s from all component substructures."""
        if not self._nodes:
            for ss in self.substructures:
                for nd in ss.nodes:
                    if nd in self._nodes:
                        self._nodes.append(nd)
        return self._nodes

    def add_substructure(self, ss: "substructure.Substructure") -> substructure.Substructure:
        """Add a substructure to the rigid body."""
        self.substructures.append(ss)
        ss.rigid_body = self
        return ss

    def get_substructure_by_name(self, name: str) -> substructure.Substructure:
        for ss in self.substructures:
            if ss.name == name:
                return ss
        raise LookupError(f"Cannot find substructure with name '{name}'")

    def initialize_position(self) -> None:
        """Initialize the position of the rigid body."""
        self.update_position(self.initial_draft - self.draft, self.initial_trim - self.trim)

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

        for nd in self.nodes:
            coords = nd.coordinates
            new_pos = trig.rotate_point(coords, np.array([self.x_cr, self.y_cr]), trim_delta)
            nd.move(*(new_pos - coords - np.array([0.0, draft_delta])))

        for s in self.substructures:
            s.update_geometry()

        self.x_cg, self.y_cg = trig.rotate_point(
            np.array([self.x_cg, self.y_cg]), np.array([self.x_cr, self.y_cr]), trim_delta
        )
        self.y_cg -= draft_delta
        self.y_cr -= draft_delta

        self.draft += draft_delta
        self.trim += trim_delta

        self.print_motion()

    def update_flexible_substructure_positions(self) -> None:
        """Update the nodal positions of all component flexible substructures."""
        flexible_substructures = [
            ss
            for ss in self.substructures
            if isinstance(ss, substructure.FlexibleMembraneSubstructure)
        ]

        num_dof = len(self.parent.nodes) * NUM_DIM
        Kg = np.zeros((num_dof, num_dof))
        Fg = np.zeros((num_dof, 1))
        Ug = np.zeros((num_dof, 1))

        # Assemble global matrices for all substructures together
        for ss in flexible_substructures:
            ss.update_fluid_forces()
            ss.assemble_global_stiffness_and_force(Kg, Fg)

        for nd in self.parent.nodes:
            node_dof = self.parent.node_dofs[nd]
            for i in range(2):
                Fg[node_dof[i]] += nd.fixed_load[i]

        # Determine fixed degrees of freedom
        dof = [False for _ in Fg]

        for nd in self.parent.nodes:
            node_dof = self.parent.node_dofs[nd]
            for dofi, fdofi in zip(node_dof, nd.is_dof_fixed):
                dof[dofi] = not fdofi

        # Solve FEM linear matrix equation
        if any(dof):
            Ug[np.ix_(dof)] = np.linalg.solve(Kg[np.ix_(dof, dof)], Fg[np.ix_(dof)])

        self.flexible_substructure_residual = np.max(np.abs(Ug))

        Ug *= self.config.solver.relax_FEM
        Ug *= np.min([self.config.solver.max_FEM_disp / np.max(Ug), 1.0])

        for nd in self.parent.nodes:
            node_dof = self.parent.node_dofs[nd]
            nd.move(Ug[node_dof[0], 0], Ug[node_dof[1], 0])

        for ss in flexible_substructures:
            ss.update_geometry()

    def update_substructure_positions(self) -> None:
        """Update the positions of all substructures."""
        self.update_flexible_substructure_positions()
        for ss in self.substructures:
            logger.info(f"Updating position for substructure: {ss.name}")
            if isinstance(ss, substructure.TorsionalSpringSubstructure):
                ss.update_angle()

    def update_fluid_forces(self) -> None:
        """Update the fluid forces by summing the force from each substructure."""
        self.reset_loads()
        for ss in self.substructures:
            ss.update_fluid_forces()
            self.D += ss.loads.D
            self.L += ss.loads.L
            self.M += ss.loads.M
            self.Da += ss.loads.Da
            self.La += ss.loads.La
            self.Ma += ss.loads.Ma

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

    def get_disp(self) -> np.ndarray:
        """Get the rigid body displacement using Broyden's method."""
        # TODO: These are here for mypy, fix the types instead
        # assert self.resFun is not None
        # assert self.x is not None

        if self.solver is None:
            self.resFun = lambda x: np.array(
                [self.L - self.weight, self.M - self.weight * (self.x_cg - self.x_cr)]
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
            if self.x_cg == self.x_cr and self.M == 0.0:
                res = 1.0
            else:
                res = (self.M - self.weight * (self.x_cg - self.x_cr)) / (
                    self.config.flow.stagnation_pressure * self.config.body.reference_length**2
                    + 1e-6
                )
        return np.abs(res * self.free_in_trim)

    def print_motion(self) -> None:
        """Print the moment for debugging."""
        lines = [
            f"Rigid Body Motion: {self.name}",
            f"  CofR: ({self.x_cr}, {self.y_cr})",
            f"  CofG: ({self.x_cg}, {self.y_cg})",
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
            logger.info(line)

    def write_motion(self) -> None:
        """Write the motion results to file."""
        if self.parent is None:
            raise AttributeError("parent must be set before simulation can be accessed.")
        writers.write_as_dict(
            self.motion_file_path,
            ["xCofR", self.x_cr],
            ["yCofR", self.y_cr],
            ["xCofG", self.x_cg],
            ["yCofG", self.y_cg],
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
        """Load the body motion from a file for the current iteration."""
        if self.parent is None:
            raise AttributeError("parent must be set before simulation can be accessed.")
        dict_ = load_dict_from_file(self.motion_file_path)
        self.x_cr = dict_.get("xCofR", np.nan)
        self.y_cr = dict_.get("yCofR", np.nan)
        self.x_cg = dict_.get("xCofG", np.nan)
        self.y_cg = dict_.get("yCofG", np.nan)
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
