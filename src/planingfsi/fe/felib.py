from __future__ import annotations

import abc
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable

import numpy as np

from planingfsi import logger
from planingfsi.config import NUM_DIM

if TYPE_CHECKING:
    from planingfsi.fe.substructure import Substructure


class Node:
    """A finite-element node, used to represent the end coordinates of truss elements.

    Attributes:
        coordinates: An array of (x, y) coordinates representing the nodal location.
        is_dof_fixed: A length-two tuple corresponding to whether the node is fixed in (x, y), respectively.
        fixed_load: An array of (x, y) external forces to apply to the node.

    """

    def __init__(
        self,
        coordinates: np.ndarray,
        *,
        is_dof_fixed: Iterable[bool] | None = None,
        fixed_load: np.ndarray | None = None,
    ) -> None:
        self.coordinates = np.array(coordinates, dtype=np.float64)
        if is_dof_fixed is not None:
            self.is_dof_fixed = tuple(bool(dof) for dof in is_dof_fixed)
        else:
            self.is_dof_fixed = tuple([False] * NUM_DIM)
        self.fixed_load = (
            np.array(fixed_load, dtype=np.float64) if fixed_load is not None else np.zeros(NUM_DIM)
        )

    @property
    def x(self) -> float:
        """The x-coordinate."""
        return self.coordinates[0]

    @property
    def y(self) -> float:
        """The y-coordinate."""
        return self.coordinates[1]

    def move(self, dx: float, dy: float) -> None:
        """Move the node by a given displacement in x & y directions.

        Args:
            dx: The displacement in x-direction.
            dy: The displacement in y-direction.

        """
        self.coordinates += np.array([dx, dy])


class Element(abc.ABC):
    """A generic element.

    Attributes:
        start_node: The start node.
        end_node: The end node.
        initial_length: The initial length of the element, i.e. distance between start & end node.
        qp: The external forces applied in the perpendicular direction.
        qs: The external forces applied in the shear direction.
        parent: A reference to the parent substructure.

    """

    def __init__(self, start_node: Node, end_node: Node, *, parent: Substructure | None = None):
        self.start_node = start_node
        self.end_node = end_node
        self.initial_length = self.length
        self.qp = np.zeros(2)
        self.qs = np.zeros(2)
        self.parent = parent
        self._initial_coordinates = [nd.coordinates for nd in self.nodes]  # save for plotting

    @property
    def nodes(self) -> tuple[Node, Node]:
        """A tuple containing both nodes in the Element."""
        return self.start_node, self.end_node

    @property
    def ramp(self) -> float:
        """The ramping coefficient from the high-level simulation object."""
        if self.parent is None:
            logger.warning("No parent assigned, ramp will be set to 1.0.")
            return 1.0
        return self.parent.ramp

    @property
    def length(self) -> float:
        """The element length, or Cartesian distance between nodes."""
        return np.linalg.norm(self.end_node.coordinates - self.start_node.coordinates)


class RigidElement(Element):
    """A rigid element, which is not subject to deformation and is generally used for drawing."""


class TrussElement(Element):
    """A truss element, used for large-deformation membrane structures.

    Attributes:
        initial_axial_force: The axial force in the element at the beginning of the simulation.
            Used to apply pre-tension. This value is ramped up to avoid numerical instability.
        EA: The axial stiffness of the element.

    """

    def __init__(self, *args, initial_axial_force: float = 0.0, EA: float = 0.0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.initial_axial_force = initial_axial_force
        self.EA = EA

    @property
    def axial_force(self) -> float:
        """The axial force in the element due to extension/compression."""
        axial_force = (1.0 - self.ramp) * self.initial_axial_force
        axial_force += self.EA * (self.length - self.initial_length) / self.initial_length
        return axial_force

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
        gamma = np.arctan2(self.end_node.y - self.start_node.y, self.end_node.x - self.start_node.x)
        c, s = np.cos(gamma), np.sin(gamma)

        transformation_matrix = np.array([[c, s, 0, 0], [-s, c, 0, 0], [0, 0, c, s], [0, 0, -s, c]])

        stiffness_total_global = (
            transformation_matrix.T @ stiffness_total_local @ transformation_matrix
        )
        force_total_global = transformation_matrix.T @ force_total_local

        return stiffness_total_global, force_total_global
