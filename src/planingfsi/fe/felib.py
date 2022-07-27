from __future__ import annotations

import abc
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from planingfsi import logger
from planingfsi import trig
from planingfsi.config import NUM_DIM

if TYPE_CHECKING:
    from planingfsi.fe.substructure import Substructure


class Node:
    __all: list["Node"] = []

    @classmethod
    def get_index(cls, ind: int) -> "Node":
        return cls.__all[ind]

    @classmethod
    def all(cls) -> list["Node"]:
        return list(cls.__all)

    @classmethod
    def count(cls) -> int:
        return len(cls.__all)

    def __init__(self) -> None:
        self.node_num = len(Node.__all)
        Node.__all.append(self)

        self.x = 0.0
        self.y = 0.0
        self.dof = [self.node_num * NUM_DIM + i for i in [0, 1]]
        self.is_dof_fixed = [True] * NUM_DIM
        self.fixed_load = np.zeros(NUM_DIM)
        self.line_xy = None

    def set_coordinates(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def move_coordinates(self, dx: float, dy: float) -> None:
        self.x += dx
        self.y += dy

    def get_coordinates(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def plot(self, _: str = None) -> None:
        # TODO: Move to plotting module
        if self.line_xy is not None:
            self.line_xy.set_data(self.x, self.y)


class Element(abc.ABC):
    __all: list["Element"] = []

    def __init__(self, parent: Substructure | None = None) -> None:
        self.element_num = len(Element.__all)
        Element.__all.append(self)

        self.node: list[Node] = []
        self.dof = [0] * NUM_DIM
        self.length = 0.0
        self.initial_length: float | None = None
        self.qp = np.zeros((2,))
        self.qs = np.zeros((2,))

        self.lineEl = None
        self.lineEl0 = None
        self.plot_on = True

        self.axial_force: float | None = None
        self.initial_axial_force: float | None = None
        self.EA: float | None = None

        self.gamma = 0.0
        self.init_pos: list[np.ndarray] = []

        self.parent = parent

    @property
    def ramp(self) -> float:
        """The ramping coefficient from the high-level simulation object."""
        if self.parent is None:
            logger.warning("No parent assigned, ramp will be set to 1.0.")
            return 1.0
        return self.parent.ramp

    def set_properties(self, **kwargs: Any) -> None:
        length = kwargs.get("length")
        axial_force = kwargs.get("axialForce")
        EA = kwargs.get("EA")

        if length is not None:
            self.length = length
            self.initial_length = length

        if axial_force is not None:
            self.axial_force = axial_force
            self.initial_axial_force = axial_force

        if EA is not None:
            self.EA = EA

    def set_nodes(self, node_list: list[Node]) -> None:
        self.node = node_list
        self.dof = [dof for nd in self.node for dof in nd.dof]
        self.update_geometry()
        self.init_pos = [np.array(nd.get_coordinates()) for nd in self.node]

    def update_geometry(self) -> None:
        # TODO: We can replace many of these with properties
        x = [nd.x for nd in self.node]
        y = [nd.y for nd in self.node]

        self.length = ((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2) ** 0.5
        if self.initial_length is None:
            self.initial_length = self.length
        self.gamma = trig.atand2(y[1] - y[0], x[1] - x[0])

    def plot(self) -> None:
        # TODO: Move to plotting module
        if self.lineEl is not None and self.plot_on:
            self.lineEl.set_data([nd.x for nd in self.node], [nd.y for nd in self.node])

        if self.lineEl0 is not None and self.plot_on:
            base_pt = [self.parent.parent.xCofR0, self.parent.parent.yCofR0]
            pos = [
                trig.rotate_point(pos, base_pt, self.parent.parent.trim)
                - np.array([0, self.parent.parent.draft])
                for pos in self.init_pos
            ]
            x, y = list(zip(*[[posi[i] for i in range(2)] for posi in pos]))
            self.lineEl0.set_data(x, y)


class TrussElement(Element):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.initial_axial_force = 0.0
        self.EA = 0.0

    def get_stiffness_and_force(self) -> tuple[np.ndarray, np.ndarray]:
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
        c, s = trig.cosd(self.gamma), trig.sind(self.gamma)

        transformation_matrix = np.array([[c, s, 0, 0], [-s, c, 0, 0], [0, 0, c, s], [0, 0, -s, c]])

        stiffness_total_global = np.dot(
            np.dot(transformation_matrix.T, stiffness_total_local), transformation_matrix
        )
        force_total_global = np.dot(transformation_matrix.T, force_total_local)

        return stiffness_total_global, force_total_global

    def update_geometry(self) -> None:
        Element.update_geometry(self)
        assert self.initial_axial_force is not None
        assert self.initial_length is not None
        assert self.EA is not None
        self.axial_force = (1.0 - self.ramp) * self.initial_axial_force + self.EA * (
            self.length - self.initial_length
        ) / self.initial_length


class RigidElement(Element):
    pass
