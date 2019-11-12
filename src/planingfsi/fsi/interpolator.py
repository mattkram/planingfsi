import numpy as np
from scipy.optimize import fmin

from .. import config
from ..solver import fzero


class Interpolator:
    def __init__(self, solid, fluid, dict_=""):
        self.solid = solid
        self.fluid = fluid

        self.solid.interpolator = self
        self.fluid.interpolator = self

        self.solid_position_function = None
        self.fluid_pressure_function = None
        self.get_body_height = self.get_surface_height_fixed_x

        self.sSep = []
        self.sImm = []

        if isinstance(dict_, str):
            dict_ = {}

        self.sSepPctStart = dict_.get("sSepPctStart", 0.5)
        self.sImmPctStart = dict_.get("sImmPctStart", 0.9)

    def set_body_height_function(self, func_name):
        self.get_body_height = getattr(self, func_name)

    def get_surface_height_fixed_x(self, x):
        s = np.max([self.get_s_fixed_x(x, 0.5), 0.0])
        return self.get_coordinates(s)[1]

    def get_coordinates(self, s):
        return self.solid_position_function(s)

    def get_min_max_s(self):
        pts = self.fluid.get_element_coords()
        return [self.get_s_fixed_x(x) for x in [pts[0], pts[-1]]]

    def get_s_fixed_x(self, x, so_pct=0.5):
        return fzero(lambda s: self.get_coordinates(s)[0] - x, so_pct * self.solid.arc_length,)

    def get_s_fixed_y(self, y, so_pct):
        return fzero(lambda s: self.get_coordinates(s)[1] - y, so_pct * self.solid.arc_length,)

    @property
    def immersed_length(self):
        if not self.sImm:
            self.sImm = self.sImmPctStart * self.solid.arc_length

        self.sImm = fzero(
            lambda s: self.get_coordinates(s)[1] - config.flow.waterline_height, self.sImm,
        )

        return self.get_coordinates(self.sImm)[0]

    def get_separation_point(self):
        def get_y_coords(s):
            return self.get_coordinates(s[0])[1]

        if not self.sSep:
            self.sSep = self.sSepPctStart * self.solid.arc_length

        self.sSep = fmin(get_y_coords, np.array([self.sSep]), disp=False, xtol=1e-6)[0]
        self.sSep = np.max([self.sSep, 0.0])
        return self.get_coordinates(self.sSep)

    def get_loads_in_range(self, s0, s1):
        x = [self.solid_position_function(s)[0] for s in [s0, s1]]
        x, p, tau = self.fluid_pressure_function(x[0], x[1])
        s = np.array([self.get_s_fixed_x(xx, 0.5) for xx in x])
        return s, p, tau
