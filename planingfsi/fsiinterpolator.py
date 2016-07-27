import numpy as np
from scipy.optimize import fmin

import config
import krampy as kp


class Interpolator():

    def __init__(self, solid, fluid, Dict=''):
        self.solid = solid
        self.fluid = fluid

        if isinstance(Dict, str):
          Dict = kp.Dictionary(Dict)
        self.Dict = Dict

        self.solid.setInterpolator(self)
        self.fluid.setInterpolator(self)

        self.solidPositionFunction = None
        self.fluidPressureFunction = None
        self.getBodyHeight = self.getSurfaceHeightFixedX

        self.sSep = []
        self.sImm = []

        self.sSepPctStart = self.Dict.readOrDefault('sSepPctStart', 0.5)
        self.sImmPctStart = self.Dict.readOrDefault('sImmPctStart', 0.9)

    def setBodyHeightFunction(self, funcName):
        exec('self.getBodyHeight = self.{0}'.format(funcName))

    def setSolidPositionFunction(self, func):
        self.solidPositionFunction = func

    def setFluidPressureFunction(self, func):
        self.fluidPressureFunction = func

    def getSurfaceHeightFixedX(self, x):
        s = np.max([self.getSFixedX(x, 0.5), 0.0])
        return self.getCoordinates(s)[1]

    def getCoordinates(self, s):
        return self.solidPositionFunction(s)

    def getMinMaxS(self):
        pts = self.fluid.getPts()
        return [self.getSFixedX(x) for x in [pts[0], pts[-1]]]

    def getSFixedX(self, fixedX, soPct=0.5):
        return kp.fzero(lambda s: self.getCoordinates(s)[0] - fixedX, soPct * self.solid.getArcLength())

    def getSFixedY(self, fixedY, soPct):
        return kp.fzero(lambda s: self.getCoordinates(s)[1] - fixedY, soPct * self.solid.getArcLength())

    def getImmersedLength(self):
        if self.sImm == []:
            self.sImm = self.sImmPctStart * self.solid.getArcLength()

        self.sImm = kp.fzero(lambda s: self.getCoordinates(s)[1] - config.hWL, self.sImm)
        
        return self.getCoordinates(self.sImm)[0]

    def getSeparationPoint(self):
        def yCoord(s):
            return self.getCoordinates(s[0])[1]

        if self.sSep == []:
            self.sSep = self.sSepPctStart * self.solid.getArcLength()

        self.sSep = fmin(yCoord, np.array([self.sSep]), disp=False, xtol=1e-6)[0]
        self.sSep = np.max([self.sSep, 0.0])
        return self.getCoordinates(self.sSep)

    def getLoadsInRange(self, s0, s1):
        x = [self.solidPositionFunction(s)[0] for s in [s0, s1]]
        x, p, tau = self.fluidPressureFunction(x[0], x[1])
        s = np.array([self.getSFixedX(xx, 0.5) for xx in x])
        return s, p, tau
