from typing import Any, Dict, Optional, List, Tuple, Callable

import numpy as np
from scipy.optimize import fmin

from .. import config
from ..fe import substructure
from ..potentialflow import pressurepatch
from ..solver import fzero


class Interpolator:
    def __init__(
        self,
        solid: "substructure.Substructure",
        fluid: "pressurepatch.PlaningSurface",
        dict_: Dict[str, Any] = None,
    ):
        self.solid = solid
        self.fluid = fluid

        self.solid.interpolator = self
        self.fluid.interpolator = self

        self.solid_position_function: Optional[Callable[[float], np.ndarray]] = None
        self.fluid_pressure_function: Optional[Callable[[float, float], np.ndarray]] = None
        self.get_body_height = self.get_surface_height_fixed_x

        self._separation_arclength: Optional[float] = None
        self._immersed_arclength: Optional[float] = None

        if dict_ is None:
            dict_ = {}

        self.sSepPctStart = dict_.get("sSepPctStart", 0.5)
        self.sImmPctStart = dict_.get("sImmPctStart", 0.9)

    def set_body_height_function(self, func_name: str) -> None:
        self.get_body_height = getattr(self, func_name)

    def get_surface_height_fixed_x(self, x: float) -> float:
        s = np.max([self.get_s_fixed_x(x, 0.5), 0.0])
        return self.get_coordinates(s)[1]

    def get_coordinates(self, s: float) -> np.ndarray:
        assert self.solid_position_function is not None
        return self.solid_position_function(s)

    def get_min_max_s(self) -> List[float]:
        pts = self.fluid.get_element_coords()
        return [self.get_s_fixed_x(x) for x in [pts[0], pts[-1]]]

    def get_s_fixed_x(self, x: float, so_pct: float = 0.5) -> float:
        return fzero(lambda s: self.get_coordinates(s)[0] - x, so_pct * self.solid.arc_length)

    def get_s_fixed_y(self, y: float, so_pct: float) -> float:
        return fzero(lambda s: self.get_coordinates(s)[1] - y, so_pct * self.solid.arc_length)

    @property
    def immersed_length(self) -> float:
        if self._immersed_arclength is None:
            self._immersed_arclength = self.sImmPctStart * self.solid.arc_length

        self._immersed_arclength = fzero(
            lambda s: self.get_coordinates(s)[1] - config.flow.waterline_height,
            self._immersed_arclength,
        )

        return self.get_coordinates(self._immersed_arclength)[0]

    def get_separation_point(self) -> np.ndarray:
        def get_y_coords(s: float) -> float:
            return self.get_coordinates(s)[1]

        if self._separation_arclength is None:
            self._separation_arclength = self.sSepPctStart * self.solid.arc_length

        self._separation_arclength = fmin(
            get_y_coords, self._separation_arclength, disp=False, xtol=1e-6
        )[0]
        self._separation_arclength = float(np.max([self._separation_arclength, 0.0]))
        return self.get_coordinates(self._separation_arclength)

    def get_loads_in_range(self, s0: float, s1: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.solid_position_function is not None
        assert self.fluid_pressure_function is not None
        x = [self.solid_position_function(s)[0] for s in [s0, s1]]
        x, p, tau = self.fluid_pressure_function(x[0], x[1])
        s = np.array([self.get_s_fixed_x(xx, 0.5) for xx in x])
        return s, p, tau
