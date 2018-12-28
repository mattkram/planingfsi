import numpy as np
from scipy.optimize import fmin

import planingfsi.config as config

# import planingfsi.krampy as kp


class Interpolator:
    def __init__(self, solid, fluid, dict_=""):
        self.solid = solid
        self.fluid = fluid

        self.solid.set_interpolator(self)
        self.fluid.interpolator = self

        self.solidPositionFunction = None
        self.fluidPressureFunction = None
        self.getBodyHeight = self.getSurfaceHeightFixedX

        self.sSep = []
        self.sImm = []

        if isinstance(dict_, str):
            dict_ = kp.Dictionary(dict_)
        #         self.Dict = dict_

        self.sSepPctStart = dict_.read("sSepPctStart", 0.5)
        self.sImmPctStart = dict_.read("sImmPctStart", 0.9)

    def setBodyHeightFunction(self, funcName):
        exec("self.getBodyHeight = self.{0}".format(funcName))

    def set_solid_position_function(self, func):
        self.solidPositionFunction = func

    def set_fluid_pressure_function(self, func):
        self.fluidPressureFunction = func

    def getSurfaceHeightFixedX(self, x):
        s = np.max([self.getSFixedX(x, 0.5), 0.0])
        return self.get_coordinates(s)[1]

    def get_coordinates(self, s):
        return self.solidPositionFunction(s)

    def get_min_max_s(self):
        pts = self.fluid._get_element_coords()
        return [self.getSFixedX(x) for x in [pts[0], pts[-1]]]

    def getSFixedX(self, fixedX, soPct=0.5):
        return kp.fzero(
            lambda s: self.get_coordinates(s)[0] - fixedX,
            soPct * self.solid.get_arc_length(),
        )

    def getSFixedY(self, fixedY, soPct):
        return kp.fzero(
            lambda s: self.get_coordinates(s)[1] - fixedY,
            soPct * self.solid.get_arc_length(),
        )

    @property
    def immersed_length(self):
        if self.sImm == []:
            self.sImm = self.sImmPctStart * self.solid.get_arc_length()

        self.sImm = kp.fzero(
            lambda s: self.get_coordinates(s)[1] - config.flow.waterline_height,
            self.sImm,
        )

        return self.get_coordinates(self.sImm)[0]

    def get_separation_point(self):
        def yCoord(s):
            return self.get_coordinates(s[0])[1]

        if self.sSep == []:
            self.sSep = self.sSepPctStart * self.solid.get_arc_length()

        self.sSep = fmin(yCoord, np.array([self.sSep]), disp=False, xtol=1e-6)[0]
        self.sSep = np.max([self.sSep, 0.0])
        return self.get_coordinates(self.sSep)

    def get_loads_in_range(self, s0, s1):
        x = [self.solidPositionFunction(s)[0] for s in [s0, s1]]
        x, p, tau = self.fluidPressureFunction(x[0], x[1])
        s = np.array([self.getSFixedX(xx, 0.5) for xx in x])
        return s, p, tau
