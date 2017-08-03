import numpy as np
from scipy.interpolate import interp1d

import planingfsi.config as config
import planingfsi.krampy as kp


class Node():
    obj = []
    
    @classmethod
    def getInd(cls, ind):
        return cls.obj[ind]

    @classmethod
    def All(cls):
        return [o for o in cls.obj]

    @classmethod
    def count(cls):
        return len(cls.All())

    @classmethod
    def findNearest(cls):
        return [o for o in cls.obj]

    def __init__(self):
        self.nodeNum = len(Node.obj)
        Node.obj.append(self)

        self.x = 0.0
        self.y = 0.0
        self.dof = [self.nodeNum * config.flow.num_dim + i for i in [0, 1]]
        self.fixedDOF = [True] * config.flow.num_dim
        self.fixedLoad = np.zeros(config.flow.num_dim)
        self.lineXY = None

    def setCoordinates(self, x, y):
        self.x = x
        self.y = y

    def moveCoordinates(self, dx, dy):
        self.x += dx
        self.y += dy

    def get_coordinates(self):
        return self.x, self.y

    def plot(self, sty=None):
        if self.lineXY is not None:
            self.lineXY.set_data(self.x, self.y)


class Element():
    obj = []

    def __init__(self):
        self.elNum = len(Element.obj)
        Element.obj.append(self)

        self.fluidS = []
        self.fluidP = []
        self.node   = [None] * 2
        self.dof    = [0] * config.flow.num_dim
        self.length = 0.0
        self.initial_length = None
        self.qp     = np.zeros((2))
        self.qs     = np.zeros((2))

        self.lineEl = None
        self.lineEl0 = None
        self.plotOn = True

    def setProperties(self, **kwargs):
        length     = kwargs.get('length', None)
        axialForce = kwargs.get('axialForce', None)
        EA         = kwargs.get('EA', None)

        if not length == None:
            self.length = length
            self.initial_length = length

        if not axialForce == None:
            self.axialForce = axialForce
            self.initialAxialForce = axialForce

        if not EA == None:
            self.EA = EA

    def get_length(self):
        return self.length

    def setNodes(self, nodeList):
        self.node = nodeList
        self.dof = [dof for nd in self.node for dof in nd.dof]
        self.update_geometry()
        self.initPos = [np.array(nd.get_coordinates()) for nd in self.node]
    
    def set_parent(self, parent):
        self.parent = parent

    def update_geometry(self):
        x = [nd.x for nd in self.node]
        y = [nd.y for nd in self.node]

        self.length = ((x[1] - x[0])**2 + (y[1] - y[0])**2)**0.5
        if self.initial_length is None:
            self.initial_length = self.length
        self.gamma = kp.atand2(y[1] - y[0], x[1] - x[0])

    def setPressureAndShear(self, qp, qs):
        self.qp = qp
        self.qs = qs

    def plot(self):
        if self.lineEl is not None and self.plotOn:
            self.lineEl.set_data([nd.x for nd in self.node], [nd.y for nd in self.node])
        
        if self.lineEl0 is not None and self.plotOn:
            basePt = [self.parent.parent.xCofR0, self.parent.parent.yCofR0]
            pos = [kp.rotatePt(pos, basePt, self.parent.parent.trim) - np.array([0, self.parent.parent.draft]) for pos in self.initPos]
            x, y = list(zip(*[[posi[i] for i in range(2)] for posi in pos]))
            self.lineEl0.set_data(x, y)


class TrussElement(Element):

    def __init__(self):
        Element.__init__(self)
        self.initialAxialForce = 0.0
        self.EA = 0.0

    def getStiffnessAndForce(self):
        # Stiffness matrices in local coordinates
        KL  = np.array([[ 1, 0,-1, 0],
                        [ 0, 0, 0, 0],
                        [-1, 0, 1, 0],
                        [ 0, 0, 0, 0]]) * self.EA / self.length

        KNL = np.array([[ 1, 0,-1, 0],
                        [ 0, 1, 0,-1],
                        [-1, 0, 1, 0],
                        [ 0,-1, 0, 1]]) * self.axialForce / self.length

        # Force vectors in local coordinates
        FL  = np.array([[self.qs[0]],
                        [self.qp[0]],
                        [self.qs[1]],
                        [self.qp[1]]])

        FNL = np.array([[ 1],
                        [ 0],
                        [-1],
                        [ 0]]) * self.axialForce 

        # Add linear and nonlinear components
        K = KL + KNL
        F = FL + FNL

        # Rotate stiffness and force matrices into global coordinates
        C = kp.cosd(self.gamma)
        S = kp.sind(self.gamma)
        
        TM = np.array([[ C, S,  0, 0],
                       [-S, C,  0, 0],
                       [ 0, 0,  C, S],
                       [ 0, 0, -S, C]])

        K = np.dot(np.dot(TM.T, K), TM)
        F = np.dot(TM.T, F)
        
        return K, F

    def update_geometry(self):
        Element.update_geometry(self)
        self.axialForce = (1 - config.ramp) * self.initialAxialForce + self.EA * (self.length - self.initial_length) / self.initial_length


class RigidElement(Element):

    def __init__(self):
        Element.__init__(self)
