from __future__ import annotations

import abc
from collections.abc import Iterable
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from planingfsi import logger
from planingfsi.config import NUM_DIM

if TYPE_CHECKING:
    from planingfsi.fe.substructure import Substructure


class Node:
    """A finite-element node, used to represent the end coordinates of truss elements."""

    def __init__(self, node_num: int) -> None:
        self.node_num = node_num
        self.x = 0.0
        self.y = 0.0
        self.dof = [self.node_num * NUM_DIM + i for i in [0, 1]]
        self.is_dof_fixed = [True] * NUM_DIM
        self.fixed_load = np.zeros(NUM_DIM)

    @property
    def coordinates(self) -> np.ndarray:
        """A 2-d array of nodal coordinates."""
        return np.array([self.x, self.y])

    @coordinates.setter
    def coordinates(self, value: Iterable[float]) -> None:
        self.x, self.y = value

    def move(self, dx: float, dy: float) -> None:
        """Move the node by a given displacement in x & y directions.

        Args:
            dx: The displacement in x-direction.
            dy: The displacement in y-direction.

        """
        self.x += dx
        self.y += dy


class Element(abc.ABC):
    """A generic element.

    Attributes:
        dof: A list of the degrees of freedom in the global array for the start and end nodes (length 4).
        initial_length: The initial length of the element, i.e. distance between Nodes.
        qp: The external forces applied in the perpendicular direction.
        qs: The external forces applied in the shear direction.
        init_pos: The initial nodal coordinates.
        parent: A reference to the parent substructure.

    """

    def __init__(self, parent: Substructure | None = None):
        self._nodes: list[Node] = []
        self.dof: list[int] = []
        self.initial_length: float | None = None
        self.qp = np.zeros((2,))
        self.qs = np.zeros((2,))

        self.lineEl = None
        self.lineEl0 = None

        self.init_pos: list[np.ndarray] = []

        self.parent = parent

    @property
    def ramp(self) -> float:
        """The ramping coefficient from the high-level simulation object."""
        if self.parent is None:
            logger.warning("No parent assigned, ramp will be set to 1.0.")
            return 1.0
        return self.parent.ramp

    @property
    def nodes(self) -> list[Node]:
        """A list containing references to the start and end nodes.

        When setting, the degrees of freedom and initial positions are stored.

        """
        return self._nodes

    @nodes.setter
    def nodes(self, node_list: list[Node]) -> None:
        self.nodes[:] = node_list
        self.dof[:] = [dof for nd in self.nodes for dof in nd.dof]
        self.initial_length = self.length
        self.init_pos[:] = [nd.coordinates for nd in self.nodes]

    @property
    def length(self) -> float:
        """The element length, or Cartesian distance between nodes."""
        return np.linalg.norm(self.nodes[1].coordinates - self.nodes[0].coordinates)


class TrussElement(Element):
    """A truss element, used for large-deformation membrane structures.

    Attributes:
        initial_axial_force: The axial force in the element at the beginning of the simulation.
            Used to apply pre-tension. This value is ramped up to avoid numerical instability.
        EA: The axial stiffness of the element.

    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.initial_axial_force = 0.0
        self.EA = 0.0

    def get_stiffness_and_force(self) -> tuple[np.ndarray, np.ndarray]:
        """The elemental stiffness matrix and force vector, in the global coordinate system."""
        # Stiffness matrices in local coordinates
        stiffness_linear_local = (
            np.array([[1, 0, -1, 0], [0, 0, 0, 0], [-1, 0, 1, 0], [0, 0, 0, 0]])
            * self.EA
            / self.length
        )

        stiffness_nonlinear_local = (
            np.array([[1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1]])
            * self.axial_force
            / self.length
        )

        # Force vectors in local coordinates
        force_linear_local = np.array([[self.qs[0]], [self.qp[0]], [self.qs[1]], [self.qp[1]]])
        force_nonlinear_local = np.array([[1], [0], [-1], [0]]) * self.axial_force

        # Add linear and nonlinear components
        stiffness_total_local = stiffness_linear_local + stiffness_nonlinear_local
        force_total_local = force_linear_local + force_nonlinear_local

        # Rotate stiffness and force matrices into global coordinates
        gamma = np.arctan2(self.nodes[1].y - self.nodes[0].y, self.nodes[1].x - self.nodes[0].x)
        c, s = np.cos(gamma), np.sin(gamma)

        transformation_matrix = np.array([[c, s, 0, 0], [-s, c, 0, 0], [0, 0, c, s], [0, 0, -s, c]])

        stiffness_total_global = (
            transformation_matrix.T @ stiffness_total_local @ transformation_matrix
        )
        force_total_global = transformation_matrix.T @ force_total_local

        return stiffness_total_global, force_total_global

    @property
    def axial_force(self) -> float:
        """The axial force in the element due to extension/compression."""
        return (1.0 - self.ramp) * self.initial_axial_force + self.EA * (
            self.length - self.initial_length
        ) / self.initial_length


class RigidElement(Element):
    """A rigid element, which is not subject to deformation and is generally used for drawing."""
