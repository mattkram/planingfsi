import os
import weakref
from typing import Dict, Any, List

import numpy as np

from . import felib as fe
from . import rigid_body as rigid_body_mod
from .substructure import (
    Substructure,
    FlexibleSubstructure,
    RigidSubstructure,
    TorsionalSpringSubstructure,
)
from .. import config, logger
from ..fsi import simulation as fsi_simulation


class StructuralSolver:
    """Parent object for solving the finite-element structure. Consists of
    several rigid bodies and substructures.
    """

    def __init__(self, simulation: "fsi_simulation.Simulation") -> None:
        self._simulation = weakref.ref(simulation)
        self.rigid_body: List["rigid_body_mod.RigidBody"] = []
        self.res = 1.0  # TODO: Can this be a property instead?

    @property
    def simulation(self) -> "fsi_simulation.Simulation":
        """A reference to the simulation object by resolving the weak reference."""
        simulation = self._simulation()
        if simulation is None:
            raise ReferenceError("Simulation object cannot be accessed.")
        return simulation

    @property
    def substructure(self) -> List[Substructure]:
        """A combined list of substructures from all rigid bodies."""
        return [ss for body in self.rigid_body for ss in body.substructure]

    @property
    def node(self) -> List[fe.Node]:
        """A combined list of nodes from all substructures."""
        return [nd for ss in self.substructure for nd in ss.node]

    def add_rigid_body(self, dict_: Dict[str, Any] = None) -> "rigid_body_mod.RigidBody":
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

    def add_substructure(self, dict_: Dict[str, Any] = None) -> "Substructure":
        """Add a substructure to the structure, whose type is determined at run-time.

        Args
        ----
        dict_: A dictionary containing substructure specifications.

        """
        if dict_ is None:
            dict_ = {}
        # TODO: This logic is better handled by the factory pattern
        ss_type = dict_.get("substructureType", "rigid")
        ss: Substructure
        if ss_type.lower() == "flexible" or ss_type.lower() == "truss":
            ss = FlexibleSubstructure(dict_)
        elif ss_type.lower() == "torsionalspring":
            ss = TorsionalSpringSubstructure(dict_)
        else:
            ss = RigidSubstructure(dict_)
        self.substructure.append(ss)

        self._assign_substructure_to_body(dict_, ss)

        return ss

    def _assign_substructure_to_body(self, dict_: Dict[str, Any], ss: "Substructure") -> None:
        """Find parent body and add substructure to it."""
        bodies = [b for b in self.rigid_body if b.name == dict_.get("bodyName", "default")]
        if bodies:
            body = bodies[0]
        else:
            body = self.rigid_body[0]
        body.add_substructure(ss)
        logger.info(f"Adding Substructure {ss.name} of type {ss.type_} to rigid body {body.name}")

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
        if config.io.results_from_file:
            self._load_response()
        else:
            for bd in self.rigid_body:
                bd.update_position()
                bd.update_substructure_positions()

    def get_residual(self) -> None:
        """Calculate the residual."""
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
        x, y = np.loadtxt(os.path.join(config.path.mesh_dir, "nodes.txt"), unpack=True)
        xf, yf = np.loadtxt(os.path.join(config.path.mesh_dir, "fixedDOF.txt"), unpack=True)
        fx, fy = np.loadtxt(os.path.join(config.path.mesh_dir, "fixedLoad.txt"), unpack=True)

        for xx, yy, xxf, yyf, ffx, ffy in zip(x, y, xf, yf, fx, fy):
            nd = fe.Node()
            nd.set_coordinates(xx, yy)
            nd.is_dof_fixed = [bool(xxf), bool(yyf)]
            nd.fixed_load = np.array([ffx, ffy])
            self.node.append(nd)

        for struct in self.substructure:
            struct.load_mesh()
            if (
                struct.type_ == "rigid"
                or struct.type_ == "rotating"
                or struct.type_ == "torsionalSpring"
            ):
                struct.set_fixed_dof()

        for ss in self.substructure:
            ss.set_attachments()
