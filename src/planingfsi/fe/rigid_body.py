from __future__ import annotations

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
from planingfsi.fe.substructure import FlexibleMembraneSubstructure
from planingfsi.fe.substructure import GlobalLoads
from planingfsi.fe.substructure import TorsionalSpringSubstructure
from planingfsi.solver import RootFinder

if TYPE_CHECKING:
    from planingfsi.fe.structure import StructuralSolver
    from planingfsi.fe.substructure import Substructure


class RigidBody:
    """A rigid body, consisting of multiple substructures.

    The rigid body is used as a container for collecting attached substructures, and also
    allows motion in heave & trim, which is calculated by integrating the total forces
    and moments applied to the substructures.

    Args:
        name: A name for the RigidBody instance
        weight: If provided, the weight to use for the rigid body (scaled by `config.body.seal_load_pct`).
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
        weight: float = 1.0,
        x_cg: float = 0.0,
        y_cg: float = 0.0,
        x_cr: float | None = None,
        y_cr: float | None = None,
        initial_draft: float = 0.0,
        initial_trim: float = 0.0,
        relax_draft: float = 1.0,
        relax_trim: float = 1.0,
        max_draft_step: float = 1e-3,
        max_trim_step: float = 1e-3,
        free_in_draft: bool = False,
        free_in_trim: bool = False,
        parent: StructuralSolver | None = None,
        **_: Any,
    ):
        self.name = name
        self.parent = parent

        self.weight = weight

        self.x_cg = x_cg
        self.y_cg = y_cg
        self.x_cr = x_cr if x_cr is not None else x_cg
        self.y_cr = y_cr if y_cr is not None else y_cg

        self.x_cr_init = self.x_cr
        self.y_cr_init = self.y_cr

        self.draft = 0.0
        self.trim = 0.0
        self.initial_draft = initial_draft
        self.initial_trim = initial_trim

        self._free_dof = np.array([free_in_draft, free_in_trim])
        self._max_disp = np.array([max_draft_step, max_trim_step])
        self._relax = np.array([relax_draft, relax_trim])

        self.loads = GlobalLoads()

        self._res_l = 1.0
        self._res_m = 1.0

        self._motion_solver = RigidBodyMotionSolver(self)

        self._flexible_substructure_residual = 0.0
        self.substructures: list[Substructure] = []

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
        res_l = self._res_l if self.free_in_draft else 0.0
        res_m = self._res_m if self.free_in_trim else 0.0
        res_node_disp = self._flexible_substructure_residual
        res_torsion = max(
            [
                ss.residual
                for ss in self.substructures
                if isinstance(ss, TorsionalSpringSubstructure)
            ],
            default=0.0,
        )
        return max(res_l, res_m, res_node_disp, res_torsion)

    @property
    def has_free_dof(self) -> bool:
        return self._free_dof.any()

    @property
    def free_in_draft(self) -> bool:
        return self._free_dof[0]

    @free_in_draft.setter
    def free_in_draft(self, value: bool) -> None:
        self._free_dof[0] = value

    @property
    def free_in_trim(self) -> bool:
        return self._free_dof[1]

    @free_in_trim.setter
    def free_in_trim(self, value: bool) -> None:
        self._free_dof[1] = value

    @property
    def motion_file_path(self) -> Path:
        """A path to the file containing the rigid body motinon at a given iteration."""
        return self.parent.simulation.it_dir / f"motion_{self.name}.{self.config.io.data_format}"

    @cached_property
    def nodes(self) -> list[fe.Node]:
        """A list of all unique `Node`s from all component substructures."""
        nodes = set()
        for ss in self.substructures:
            for nd in ss.nodes:
                nodes.add(nd)
        return list(nodes)

    def add_substructure(self, ss: Substructure) -> Substructure:
        """Add a substructure to the rigid body."""
        self.substructures.append(ss)
        ss.rigid_body = self
        return ss

    def get_substructure_by_name(self, name: str) -> Substructure:
        for ss in self.substructures:
            if ss.name == name:
                return ss
        raise LookupError(f"Cannot find substructure with name '{name}'")

    def initialize_position(self) -> None:
        """Initialize the position of the rigid body."""
        self.update_position(self.initial_draft - self.draft, self.initial_trim - self.trim)

    def update_position(self, draft_delta: float = None, trim_delta: float = None) -> None:
        """Update the position of the rigid body by passing the change in draft and trim."""
        if not self.has_free_dof:
            return

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

    def _update_flexible_substructure_positions(self) -> None:
        """Update the nodal positions of all component flexible substructures."""
        flexible_substructures = [
            ss for ss in self.substructures if isinstance(ss, FlexibleMembraneSubstructure)
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
            Fg[node_dof] += nd.fixed_load[:, np.newaxis]

        # Determine fixed degrees of freedom
        dof = [False for _ in Fg]

        for nd in self.parent.nodes:
            node_dof = self.parent.node_dofs[nd]
            for dofi, fdofi in zip(node_dof, nd.is_dof_fixed):
                dof[dofi] = not fdofi

        # Solve FEM linear matrix equation
        if any(dof):
            Ug[np.ix_(dof)] = np.linalg.solve(Kg[np.ix_(dof, dof)], Fg[np.ix_(dof)])

        self._flexible_substructure_residual = np.max(np.abs(Ug))

        Ug *= self.config.solver.relax_FEM
        Ug *= np.min([self.config.solver.max_FEM_disp / np.max(Ug), 1.0])

        for nd in self.parent.nodes:
            node_dof = self.parent.node_dofs[nd]
            nd.move(Ug[node_dof[0], 0], Ug[node_dof[1], 0])

        for ss in flexible_substructures:
            ss.update_geometry()

    def update_substructure_positions(self) -> None:
        """Update the positions of all substructures."""
        self._update_flexible_substructure_positions()
        for ss in self.substructures:
            logger.info(f"Updating position for substructure: {ss.name}")
            if isinstance(ss, TorsionalSpringSubstructure):
                ss.update_angle()

    def update_fluid_forces(self) -> None:
        """Update the fluid forces by summing the force from each substructure."""
        self.loads = GlobalLoads()
        if self.config.body.cushion_force_method.lower() == "assumed":
            self.loads.L += (
                self.config.body.Pc
                * self.config.body.reference_length
                * trig.cosd(self.config.body.initial_trim)
            )

        for ss in self.substructures:
            ss.update_fluid_forces()
            self.loads += ss.loads

        self._res_l = self.get_res_lift()
        self._res_m = self.get_res_moment()

    def get_disp(self) -> np.ndarray:
        """Get the rigid body displacement using Broyden's method."""
        return self._motion_solver.get_disp()

    def get_res_lift(self) -> float:
        """Get the residual of the vertical force balance."""
        if np.isnan(self.loads.L):
            res = 1.0
        else:
            res = (self.loads.L - self.weight) / (
                self.config.flow.stagnation_pressure * self.config.body.reference_length + 1e-6
            )
        return np.abs(res * self.free_in_draft)

    def get_res_moment(self) -> float:
        """Get the residual of the trim moment balance."""
        if np.isnan(self.loads.M):
            res = 1.0
        else:
            if self.x_cg == self.x_cr and self.loads.M == 0.0:
                res = 1.0
            else:
                res = (self.loads.M - self.weight * (self.x_cg - self.x_cr)) / (
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
            f"  Lift Force: {self.loads.L:5.4e}",
            f"  Drag Force: {self.loads.D:5.4e}",
            f"  Moment:     {self.loads.M:5.4e}",
            f"  Lift Force Air: {self.loads.La:5.4e}",
            f"  Drag Force Air: {self.loads.Da:5.4e}",
            f"  Moment Air:     {self.loads.Ma:5.4e}",
            f"  Lift Res:   {self._res_l:5.4e}",
            f"  Moment Res: {self._res_m:5.4e}",
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
            ["liftRes", self._res_l],
            ["momentRes", self._res_m],
            ["Lift", self.loads.L],
            ["Drag", self.loads.D],
            ["Moment", self.loads.M],
            ["LiftAir", self.loads.La],
            ["DragAir", self.loads.Da],
            ["MomentAir", self.loads.Ma],
        )
        for ss in self.substructures:
            if isinstance(ss, TorsionalSpringSubstructure):
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
        self._res_l = dict_.get("liftRes", np.nan)
        self._res_m = dict_.get("momentRes", np.nan)
        self.loads.L = dict_.get("Lift", np.nan)
        self.loads.D = dict_.get("Drag", np.nan)
        self.loads.M = dict_.get("Moment", np.nan)
        self.loads.La = dict_.get("LiftAir", np.nan)
        self.loads.Da = dict_.get("DragAir", np.nan)
        self.loads.Ma = dict_.get("MomentAir", np.nan)


class RigidBodyMotionSolver:
    """A customized Broyden-method solver to allow manual stepping required for rigid body motion solve."""

    def __init__(self, parent: RigidBody):
        self.parent = parent

        self._jacobian_reset_interval = 6
        self.solver: solver.RootFinder | None = None
        self.disp_old: np.ndarray | None = None
        self.res_old: np.ndarray | None = None
        self.J: np.ndarray | None = None
        self._J_tmp: np.ndarray | None = None
        self._J_fo: np.ndarray | None = None
        self._J_it = 0
        self._step = 0

    def _reset_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Reset the solver Jacobian and modify displacement."""
        f = self._get_residual(x)
        if self._J_tmp is None:
            self._J_tmp = np.zeros((NUM_DIM, NUM_DIM))
            self._J_it = 0
            self._step = 0
            self._J_fo = f
            self.res_old = self._J_fo.copy()
        else:
            self._J_tmp[:, self._J_it] = (f - self._J_fo) / self.disp_old[self._J_it]
            self._J_it += 1

        disp = np.zeros((NUM_DIM,))
        if self._J_it < NUM_DIM:
            disp[self._J_it] = self.parent.config.body.motion_jacobian_first_step

        self.disp_old = disp
        if self._J_it >= NUM_DIM:
            self.J = self._J_tmp.copy()
            self._J_tmp = None
            self.disp_old = None

        return disp

    def _limit_disp(self, disp: np.ndarray) -> np.ndarray:
        """Limit the body displacement.

        If both degrees of freedom are free, we ensure the same ratio of draft-to-trim is maintained.
        If only one is free, or we have a zero displacement in either degree of freedom, then we only step
        in the single direction.

        """
        limited_disp = np.min(np.abs(np.vstack((disp, self.parent._max_disp))), axis=0)

        ind = disp != 0 & self.parent._free_dof
        limit_fraction = min(limited_disp[ind] / np.abs(disp[ind]), default=0.0)

        return disp * limit_fraction * self.parent._free_dof

    def _get_residual(self, _):
        return np.array([self.parent.get_res_lift(), self.parent.get_res_moment()])

    def get_disp(self) -> np.ndarray:
        """Get the rigid body displacement using Broyden's method."""
        x = np.array([self.parent.draft, self.parent.trim])
        if self.solver is None:
            self.solver = RootFinder(self._get_residual, x, method="Broyden")

        if self.J is None:
            return self._reset_jacobian(x)

        if self._step >= self._jacobian_reset_interval:
            logger.debug("Resetting Jacobian for Motion")
            self._reset_jacobian(x)

        f = self._get_residual(x)
        if self.disp_old is not None:
            x += self.disp_old

            dx = self.disp_old[:, np.newaxis]
            df = (f - self.res_old)[:, np.newaxis]
            self.J += (df - self.J @ dx) @ dx.T / np.linalg.norm(dx) ** 2

        dof = self.parent._free_dof
        dx = np.zeros_like(x)
        dx[np.ix_(dof)] = np.linalg.solve(-self.J[np.ix_(dof, dof)], f[np.ix_(dof)])

        disp = self._limit_disp(dx[:] * self.parent._relax)

        self.disp_old = disp
        self.res_old = f[:]
        self._step += 1

        return disp
