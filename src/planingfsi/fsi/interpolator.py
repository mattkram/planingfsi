from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import fmin

from planingfsi.fe import substructure
from planingfsi.potentialflow import pressurepatch
from planingfsi.solver import fzero


class Interpolator:
    def __init__(
        self,
        solid: "substructure.Substructure",
        fluid: "pressurepatch.PlaningSurface",
        dict_: dict[str, Any] = None,
    ):
        self.solid = solid
        self.fluid = fluid

        self.solid.interpolator = self
        self.fluid.interpolator = self

        self.solid_position_function: Callable[[float], np.ndarray] | None = solid.get_coordinates
        self.fluid_pressure_function: Callable[
            [float, float], np.ndarray
        ] | None = fluid.get_loads_in_range
        self.get_body_height = self.get_surface_height_fixed_x

        self._separation_arclength: float | None = None
        self._immersed_arclength: float | None = None

        if dict_ is None:
            dict_ = {}

        self._waterline_height = dict_.get("waterline_height", 0.0)
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

    def get_min_max_s(self) -> list[float]:
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
            lambda s: self.get_coordinates(s)[1] - self._waterline_height,
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

    def get_loads_in_range(self, s0: float, s1: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.solid_position_function is not None
        assert self.fluid_pressure_function is not None
        x = [self.solid_position_function(s)[0] for s in [s0, s1]]
        x, p, tau = self.fluid_pressure_function(x[0], x[1])
        s = np.array([self.get_s_fixed_x(xx, 0.5) for xx in x])
        return s, p, tau
