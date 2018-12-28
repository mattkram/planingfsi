import numpy as np
from scipy.interpolate import interp1d

import planingfsi.config as config

# import planingfsi.krampy as kp


class Node:
    obj = []

    @classmethod
    def get_index(cls, ind):
        return cls.obj[ind]

    @classmethod
    def All(cls):
        return [o for o in cls.obj]

    @classmethod
    def count(cls):
        return len(cls.All())

    @classmethod
    def find_nearest(cls):
        return [o for o in cls.obj]

    def __init__(self):
        self.node_num = len(Node.obj)
        Node.obj.append(self)

        self.x = 0.0
        self.y = 0.0
        self.dof = [self.node_num * config.flow.num_dim + i for i in [0, 1]]
        self.is_dof_fixed = [True] * config.flow.num_dim
        self.fixed_load = np.zeros(config.flow.num_dim)
        self.line_xy = None

    def set_coordinates(self, x, y):
        self.x = x
        self.y = y

    def move_coordinates(self, dx, dy):
        self.x += dx
        self.y += dy

    def get_coordinates(self):
        return self.x, self.y

    def plot(self, sty=None):
        if self.line_xy is not None:
            self.line_xy.set_data(self.x, self.y)


class Element:
    obj = []

    def __init__(self):
        self.element_num = len(Element.obj)
        Element.obj.append(self)

        # self.fluidS = []
        # self.fluidP = []
        self.node = [None] * 2
        self.dof = [0] * config.flow.num_dim
        self.length = 0.0
        self.initial_length = None
        self.qp = np.zeros((2))
        self.qs = np.zeros((2))

        self.lineEl = None
        self.lineEl0 = None
        self.plot_on = True

    def set_properties(self, **kwargs):
        length = kwargs.get("length", None)
        axialForce = kwargs.get("axialForce", None)
        EA = kwargs.get("EA", None)

        if not length == None:
            self.length = length
            self.initial_length = length

        if not axialForce == None:
            self.axial_force = axialForce
            self.initial_axial_force = axialForce

        if not EA == None:
            self.EA = EA

    def set_nodes(self, nodeList):
        self.node = nodeList
        self.dof = [dof for nd in self.node for dof in nd.dof]
        self.update_geometry()
        self.init_pos = [np.array(nd.get_coordinates()) for nd in self.node]

    def set_parent(self, parent):
        self.parent = parent

    def update_geometry(self):
        x = [nd.x for nd in self.node]
        y = [nd.y for nd in self.node]

        self.length = ((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2) ** 0.5
        if self.initial_length is None:
            self.initial_length = self.length
        self.gamma = kp.atand2(y[1] - y[0], x[1] - x[0])

    def set_pressure_and_shear(self, qp, qs):
        self.qp = qp
        self.qs = qs

    def plot(self):
        if self.lineEl is not None and self.plot_on:
            self.lineEl.set_data([nd.x for nd in self.node], [nd.y for nd in self.node])

        if self.lineEl0 is not None and self.plot_on:
            basePt = [self.parent.parent.xCofR0, self.parent.parent.yCofR0]
            pos = [
                kp.rotatePt(pos, basePt, self.parent.parent.trim)
                - np.array([0, self.parent.parent.draft])
                for pos in self.init_pos
            ]
            x, y = list(zip(*[[posi[i] for i in range(2)] for posi in pos]))
            self.lineEl0.set_data(x, y)


class TrussElement(Element):
    def __init__(self):
        Element.__init__(self)
        self.initial_axial_force = 0.0
        self.EA = 0.0

    def get_stiffness_and_force(self):
        # Stiffness matrices in local coordinates
        KL = (
            np.array([[1, 0, -1, 0], [0, 0, 0, 0], [-1, 0, 1, 0], [0, 0, 0, 0]])
            * self.EA
            / self.length
        )

        KNL = (
            np.array([[1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1]])
            * self.axial_force
            / self.length
        )

        # Force vectors in local coordinates
        FL = np.array([[self.qs[0]], [self.qp[0]], [self.qs[1]], [self.qp[1]]])

        FNL = np.array([[1], [0], [-1], [0]]) * self.axial_force

        # Add linear and nonlinear components
        K = KL + KNL
        F = FL + FNL

        # Rotate stiffness and force matrices into global coordinates
        C = kp.cosd(self.gamma)
        S = kp.sind(self.gamma)

        TM = np.array([[C, S, 0, 0], [-S, C, 0, 0], [0, 0, C, S], [0, 0, -S, C]])

        K = np.dot(np.dot(TM.T, K), TM)
        F = np.dot(TM.T, F)

        return K, F

    def update_geometry(self):
        Element.update_geometry(self)
        self.axial_force = (1 - config.ramp) * self.initial_axial_force + self.EA * (
            self.length - self.initial_length
        ) / self.initial_length


class RigidElement(Element):
    pass
    # def __init__(self):
    # super().__init__(self)
