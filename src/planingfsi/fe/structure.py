from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from planingfsi import logger
from planingfsi.config import Config
from planingfsi.fe.felib import Node
from planingfsi.fe.femesh import Mesh
from planingfsi.fe.rigid_body import RigidBody
from planingfsi.fe.substructure import FlexibleSubstructure
from planingfsi.fe.substructure import RigidSubstructure
from planingfsi.fe.substructure import Substructure
from planingfsi.fe.substructure import TorsionalSpringSubstructure

if TYPE_CHECKING:
    from planingfsi.simulation import Simulation


class StructuralSolver:
    """High-level finite-element structural solver, consisting of a number of rigid bodies.

    Attributes:
        simulation: A reference to the parent `Simulation` object.
        rigid_bodies: A list of all `RigidBody` instances in the simulation.
        residual: The current residual of the structural solver.

    """

    def __init__(self, simulation: Simulation):
        self.simulation = simulation
        self.rigid_bodies: list[RigidBody] = []
        self.residual = 1.0  # TODO: Can this be a property instead?

    @property
    def config(self) -> Config:
        """A reference to the simulation configuration."""
        return self.simulation.config

    @property
    def has_free_structure(self) -> bool:
        """True if any rigid body or substructure are free to move."""
        for rigid_body in self.rigid_bodies:
            if any(rigid_body.free_dof):
                return True

            for ss in rigid_body.substructures:
                if ss.is_free:
                    return True
        return False

    @property
    def lift_residual(self) -> float:
        """The maximum lift force residual."""
        return max(body.get_res_lift() for body in self.rigid_bodies)

    @property
    def moment_residual(self) -> float:
        """The maximum trim moment residual."""
        return max(body.get_res_moment() for body in self.rigid_bodies)

    @property
    def substructures(self) -> list[Substructure]:
        """A combined list of all substructures from all rigid bodies."""
        return [ss for body in self.rigid_bodies for ss in body.substructures]

    @property
    def nodes(self) -> list[Node]:
        """A combined list of all nodes from all substructures."""
        return [nd for ss in self.substructures for nd in ss.node]

    def add_rigid_body(
        self, dict_or_instance: dict[str, Any] | RigidBody | None = None
    ) -> RigidBody:
        """Add a rigid body to the structure.

        Args:
            dict_or_instance: A dictionary of values, or a RigidBody instance.

        """
        if isinstance(dict_or_instance, RigidBody):
            rigid_body = dict_or_instance
            rigid_body.parent = self
        else:
            dict_ = dict_or_instance or {}
            rigid_body = RigidBody(**dict_, parent=self)
        self.rigid_bodies.append(rigid_body)
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
        self.substructures.append(ss)

        self._assign_substructure_to_body(ss, **dict_)

        return ss

    def _assign_substructure_to_body(
        self, ss: "Substructure", body_name: str = "default", **_: Any
    ) -> None:
        """Find parent body and add substructure to it."""
        bodies = [b for b in self.rigid_bodies if b.name == body_name]
        if bodies:
            body = bodies[0]
        else:
            body = self.rigid_bodies[0]
        body.add_substructure(ss)
        logger.info(
            f"Adding Substructure {ss.name} of type {type(ss).__name__} to rigid body {body.name}"
        )

    def initialize_rigid_bodies(self) -> None:
        """Initialize the position of all rigid bodies."""
        for bd in self.rigid_bodies:
            bd.initialize_position()

    def update_fluid_forces(self) -> None:
        """Update fluid forces on all rigid bodies."""
        for bd in self.rigid_bodies:
            bd.update_fluid_forces()

    def calculate_response(self) -> None:
        """Calculate the structural response, or load from files."""
        if self.config.io.results_from_file:
            self._load_response()
        else:
            for bd in self.rigid_bodies:
                bd.update_position()
                bd.update_substructure_positions()

    def get_residual(self) -> None:
        """Calculate the residual."""
        self.residual = 0.0
        for bd in self.rigid_bodies:
            if any(bd.free_dof):
                self.residual = np.max([np.abs(bd.res_l), self.residual])
                self.residual = np.max([np.abs(bd.res_m), self.residual])
            self.residual = np.max([FlexibleSubstructure.res, self.residual])

    def _load_response(self) -> None:
        """Load the response from files."""
        self.update_fluid_forces()

        for bd in self.rigid_bodies:
            bd.load_motion()
            for ss in bd.substructures:
                ss.load_coordinates()
                ss.update_geometry()

    def write_results(self) -> None:
        """Write the results to file."""
        for bd in self.rigid_bodies:
            bd.write_motion()
            for ss in bd.substructures:
                ss.write_coordinates()

    def plot(self) -> None:
        """Plot the results."""
        # TODO: Move to figure module
        for body in self.rigid_bodies:
            for struct in body.substructures:
                struct.plot()

    def _load_mesh_from_object(self, mesh: Mesh) -> None:
        """Load a mesh from an existing object."""
        for pt in mesh.points:
            nd = Node()
            nd.set_coordinates(pt.position[0], pt.position[1])
            nd.is_dof_fixed[:] = pt.is_dof_fixed
            nd.fixed_load[:] = pt.fixed_load
            self.nodes.append(nd)

        for struct in self.substructures:
            # Find submesh with same name as substructure
            submesh = [submesh for submesh in mesh.submesh if submesh.name == struct.name][0]
            struct.load_mesh(submesh)
            if isinstance(struct, (RigidSubstructure, TorsionalSpringSubstructure)):
                struct.set_fixed_dof()
            struct.set_attachments()

    def _load_mesh_from_dir(self, mesh_dir: Path) -> None:
        """Load the mesh from a directory of files."""
        # TODO: Can we do Mesh.from_dir() instead and then integrate the repeated logic into _load_mesh_from_object?
        coords = np.loadtxt(mesh_dir / "nodes.txt")
        fixed_dofs = np.loadtxt(mesh_dir / "fixedDOF.txt")
        loads = np.loadtxt(mesh_dir / "fixedLoad.txt")
        for c, fixed_dof, load in zip(coords, fixed_dofs, loads):
            nd = Node()
            nd.set_coordinates(*c)
            nd.is_dof_fixed[:] = map(bool, fixed_dof)
            nd.fixed_load[:] = load
            self.nodes.append(nd)
        for struct in self.substructures:
            struct.load_mesh(mesh_dir)
            if isinstance(struct, (RigidSubstructure, TorsionalSpringSubstructure)):
                struct.set_fixed_dof()
            struct.set_attachments()

    def load_mesh(self, mesh: Path | Mesh = Path("mesh")) -> None:
        """Load the mesh from a directory of files or an existing mesh object."""
        # Create all nodes
        if isinstance(mesh, Mesh):
            self._load_mesh_from_object(mesh)
        else:
            self._load_mesh_from_dir(mesh)
