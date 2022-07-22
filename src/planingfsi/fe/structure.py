from __future__ import annotations

import os
from typing import Any

import numpy as np

from planingfsi import logger
from planingfsi.config import Config
from planingfsi.fe import felib as fe
from planingfsi.fe import rigid_body as rigid_body_mod
from planingfsi.fe.substructure import FlexibleSubstructure
from planingfsi.fe.substructure import RigidSubstructure
from planingfsi.fe.substructure import Substructure
from planingfsi.fe.substructure import TorsionalSpringSubstructure
from planingfsi.fsi import simulation as fsi_simulation


class StructuralSolver:
    """Parent object for solving the finite-element structure. Consists of
    several rigid bodies and substructures.
    """

    def __init__(self, simulation: "fsi_simulation.Simulation") -> None:
        self.simulation = simulation
        self.rigid_body: list["rigid_body_mod.RigidBody"] = []
        self.res = 1.0  # TODO: Can this be a property instead?

    @property
    def config(self) -> Config:
        """A reference to the simulation configuration."""
        return self.simulation.config

    @property
    def has_free_structure(self) -> bool:
        """True if any rigid body or substructure are free to move."""
        for rigid_body in self.rigid_body:
            if rigid_body.free_in_draft or rigid_body.free_in_trim:
                return True

            for ss in rigid_body.substructure:
                if ss.is_free:
                    return True
        return False

    @property
    def res_l(self) -> float:
        """The lift residual."""
        # TODO: This should handle multiple rigid bodies
        return self.rigid_body[0].get_res_lift()

    @property
    def res_m(self) -> float:
        """The trim moment residual."""
        # TODO: This should handle multiple rigid bodies
        return self.rigid_body[0].get_res_moment()

    @property
    def substructure(self) -> list["Substructure"]:
        """A combined list of substructures from all rigid bodies."""
        return [ss for body in self.rigid_body for ss in body.substructure]

    @property
    def node(self) -> list["fe.Node"]:
        """A combined list of nodes from all substructures."""
        return [nd for ss in self.substructure for nd in ss.node]

    def add_rigid_body(self, dict_: dict[str, Any] = None) -> "rigid_body_mod.RigidBody":
        """Add a rigid body to the structure.

        Args
        ----
        dict_: A dictionary containing rigid body specifications.

        """
        if dict_ is None:
            dict_ = {}
        rigid_body = rigid_body_mod.RigidBody(dict_, parent=self)
        self.rigid_body.append(rigid_body)
        return rigid_body

    def add_substructure(
        self, dict_or_instance: dict[str, Any] | Substructure | None = None
    ) -> "Substructure":
        """Add a substructure to the structure, whose type is determined at run-time.

        Args:
            dict_or_instance: A dictionary of values, or a Substructure instance.

        """
        if isinstance(dict_or_instance, Substructure):
            ss = dict_or_instance
            dict_ = {}
        else:
            dict_ = dict_or_instance or {}

            # TODO: This logic is better handled by the factory pattern
            ss_type = dict_.get("substructureType", "rigid")
            ss_class: type[Substructure]
            if ss_type.lower() == "flexible" or ss_type.lower() == "truss":
                ss_class = FlexibleSubstructure
            elif ss_type.lower() == "torsionalspring":
                ss_class = TorsionalSpringSubstructure
            else:
                ss_class = RigidSubstructure
            ss = ss_class(**dict_)
        ss.solver = self
        self.substructure.append(ss)

        self._assign_substructure_to_body(ss, **dict_)

        return ss

    def _assign_substructure_to_body(
        self, ss: "Substructure", body_name: str = "default", **_: Any
    ) -> None:
        """Find parent body and add substructure to it."""
        bodies = [b for b in self.rigid_body if b.name == body_name]
        if bodies:
            body = bodies[0]
        else:
            body = self.rigid_body[0]
        body.add_substructure(ss)
        logger.info(
            f"Adding Substructure {ss.name} of type {type(ss).__name__} to rigid body {body.name}"
        )

    def initialize_rigid_bodies(self) -> None:
        """Initialize the position of all rigid bodies."""
        for bd in self.rigid_body:
            bd.initialize_position()

    def update_fluid_forces(self) -> None:
        """Update fluid forces on all rigid bodies."""
        for bd in self.rigid_body:
            bd.update_fluid_forces()

    def calculate_response(self) -> None:
        """Calculate the structural response, or load from files."""
        if self.config.io.results_from_file:
            self._load_response()
        else:
            for bd in self.rigid_body:
                bd.update_position()
                bd.update_substructure_positions()

    def get_residual(self) -> None:
        """Calculate the residual."""
        # TODO: Remove after circular dependencies resolved
        from planingfsi.fe.substructure import FlexibleSubstructure  # noqa: F811

        self.res = 0.0
        for bd in self.rigid_body:
            if bd.free_in_draft or bd.free_in_trim:
                self.res = np.max([np.abs(bd.res_l), self.res])
                self.res = np.max([np.abs(bd.res_m), self.res])
            self.res = np.max([FlexibleSubstructure.res, self.res])

    def _load_response(self) -> None:
        """Load the response from files."""
        self.update_fluid_forces()

        for bd in self.rigid_body:
            bd.load_motion()
            for ss in bd.substructure:
                ss.load_coordinates()
                ss.update_geometry()

    def write_results(self) -> None:
        """Write the results to file."""
        for bd in self.rigid_body:
            bd.write_motion()
            for ss in bd.substructure:
                ss.write_coordinates()

    def plot(self) -> None:
        """Plot the results."""
        # TODO: Move to figure module
        for body in self.rigid_body:
            for struct in body.substructure:
                struct.plot()

    def load_mesh(self) -> None:
        """Load the mesh from files."""
        # Create all nodes
        x, y = np.loadtxt(os.path.join(self.config.path.mesh_dir, "nodes.txt"), unpack=True)
        xf, yf = np.loadtxt(os.path.join(self.config.path.mesh_dir, "fixedDOF.txt"), unpack=True)
        fx, fy = np.loadtxt(os.path.join(self.config.path.mesh_dir, "fixedLoad.txt"), unpack=True)

        for xx, yy, xxf, yyf, ffx, ffy in zip(x, y, xf, yf, fx, fy):
            nd = fe.Node()
            nd.set_coordinates(xx, yy)
            nd.is_dof_fixed = [bool(xxf), bool(yyf)]
            nd.fixed_load = np.array([ffx, ffy])
            self.node.append(nd)

        for struct in self.substructure:
            struct.load_mesh()
            if isinstance(struct, (RigidSubstructure, TorsionalSpringSubstructure)):
                struct.set_fixed_dof()

        for ss in self.substructure:
            ss.set_attachments()
