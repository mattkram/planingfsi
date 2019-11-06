"""Classes representing a pressure patch on the free surface."""
import abc
from pathlib import Path
from typing import List, Optional, Any, Dict

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fmin

from . import pressureelement as pe, solver
from .. import config, trig, general
from ..dictionary import load_dict_from_file
from ..fsi.interpolator import Interpolator


class PressurePatch(abc.ABC):
    """Abstract base class representing a patch of pressure elements on the free surface.

    Attributes:
        pressure_elements: List of pressure elements.
        is_kutta_unknown: True of trailing edge pressure unknown.
        interpolator: Object to get interpolated body position if a `PlaningSurface`
        parent: Parent solver this patch belongs to.

    """

    def __init__(self) -> None:
        self.patch_name = "AbstractPressurePatch"
        self.pressure_elements: List[pe.PressureElement] = []
        self._end_pts = np.zeros(2)
        self.is_kutta_unknown = False
        self._neighbor_up: Optional["PressurePatch"] = None
        self._neighbor_down: Optional["PressurePatch"] = None
        self.interpolator: Optional[Interpolator] = None
        self.parent: Optional[solver.PotentialPlaningSolver] = None

        self.drag_total = np.nan
        self.drag_pressure = np.nan
        self.drag_friction = np.nan
        self.drag_wave = np.nan
        self.lift_total = np.nan
        self.lift_pressure = np.nan
        self.lift_friction = np.nan
        self.moment_total = np.nan

    @property
    def base_pt(self) -> float:
        """The x-location of the base point."""
        return self._end_pts[0]

    @base_pt.setter
    def base_pt(self, x: float) -> None:
        self._end_pts[0] = x

    @property
    def length(self) -> float:
        """Length of pressure patch, which is the distance between end points."""
        return self._end_pts[1] - self._end_pts[0]

    @length.setter
    def length(self, value: float) -> None:
        raise NotImplementedError

    @property
    def neighbor_up(self) -> Optional["PressurePatch"]:
        """PressurePatch instance upstream of this one.

        When setting, this patch is set as the other's downstream neighbor.

        """
        return self._neighbor_up

    @neighbor_up.setter
    def neighbor_up(self, obj: Optional["PressurePatch"]) -> None:
        self._neighbor_up = obj
        if self._neighbor_up is not None:
            self._neighbor_up.neighbor_down = self

    @property
    def neighbor_down(self) -> Optional["PressurePatch"]:
        """PressurePatch instance downstream of this one.

        When setting, this patch is set as the other's upstream neighbor.

        """
        return self._neighbor_down

    @neighbor_down.setter
    def neighbor_down(self, obj: Optional["PressurePatch"]) -> None:
        self._neighbor_down = obj
        if self._neighbor_down is not None:
            self._neighbor_down.neighbor_up = self

    @abc.abstractmethod
    def get_element_coords(self) -> np.ndarray:
        """Get the x-position of pressure elements."""
        return NotImplemented

    def _reset_element_coords(self) -> None:
        """Re-distribute pressure element positions along the patch."""
        x = self.get_element_coords()
        for i, el in enumerate(self.pressure_elements):
            el.x_coord = x[i]
            if not el.is_source and self.interpolator is not None:
                el.z_coord = self.interpolator.get_surface_height_fixed_x(x[i])

            if isinstance(el, pe.CompleteTriangularPressureElement):
                el.width = [x[i] - x[i - 1], x[i + 1] - x[i]]
            elif isinstance(el, pe.ForwardHalfTriangularPressureElement):
                el.width = x[i + 1] - x[i]
            elif isinstance(el, pe.AftHalfTriangularPressureElement):
                el.width = x[i] - x[i - 1]
            elif isinstance(el, pe.AftSemiInfinitePressureBand):
                el.width = np.inf
            else:
                raise ValueError("Invalid Element Type!")

    def get_free_surface_height(self, x: float) -> float:
        """Get free surface height at a position x due to the elements on this patch.

        Args:
            x: x-position at which to calculate free-surface height.

        """
        return sum([el.get_influence(x) for el in self.pressure_elements])

    @abc.abstractmethod
    def calculate_forces(self) -> None:
        """Calculate the force components for this pressure patch."""
        return NotImplemented

    def _calculate_wave_drag(self) -> float:
        """Calculate wave drag of patch."""
        xo = -10 * config.flow.lam
        (xTrough,) = fmin(self.get_free_surface_height, xo, disp=False)
        (xCrest,) = fmin(lambda x: -self.get_free_surface_height(x), xo, disp=False)
        return (
            0.0625
            * config.flow.density
            * config.flow.gravity
            * (self.get_free_surface_height(xCrest) - self.get_free_surface_height(xTrough)) ** 2
        )

    @property
    def _force_file_save_path(self) -> Path:
        """A path to the force file."""
        assert self.parent is not None
        return self.parent.simulation.it_dir / f"forces_{self.patch_name}.{config.io.data_format}"

    def write_forces(self) -> None:
        """Write forces to file."""
        general.writeasdict(
            self._force_file_save_path,
            ["Drag", self.drag_total],
            ["WaveDrag", self.drag_wave],
            ["PressDrag", self.drag_pressure],
            ["FricDrag", self.drag_friction],
            ["Lift", self.lift_total],
            ["PressLift", self.lift_pressure],
            ["FricLift", self.lift_friction],
            ["Moment", self.moment_total],
            ["BasePt", self.base_pt],
            ["Length", self.length],
        )

    def load_forces(self) -> None:
        """Load forces from file."""
        dict_ = load_dict_from_file(self._force_file_save_path)
        self.drag_total = dict_.get("Drag", 0.0)
        self.drag_pressure = dict_.get("PressDrag", 0.0)
        self.drag_friction = dict_.get("FricDrag", 0.0)
        self.lift_total = dict_.get("Lift", 0.0)
        self.lift_pressure = dict_.get("PressLift", 0.0)
        self.lift_friction = dict_.get("FricLift", 0.0)
        self.moment_total = dict_.get("Moment", 0.0)
        self.base_pt = dict_.get("BasePt", 0.0)
        self.length = dict_.get("Length", 0.0)


class PressureCushion(PressurePatch):
    """Pressure Cushion consisting solely of source elements.

    Args:
        dict_: Dictionary containing patch definitions.

    Attributes:
        patch_name (str): Name of patch
        cushion_type (str): The type of pressure cushion.
        cushion_pressure (float): The value of the pressure in the cushion.

    """

    count = 0

    def __init__(self, dict_: Dict[str, Any]) -> None:
        super().__init__()
        PressureCushion.count += 1
        self.patch_name = dict_.get(
            "pressureCushionName", f"pressureCushion{PressureCushion.count}"
        )

        self.cushion_pressure = dict_.get("cushionPressure")
        if self.cushion_pressure is None:
            self.cushion_pressure = getattr(config, "cushionPressure", 0.0)

        self.neighbor_up = PlaningSurface.find_by_name(dict_.get("upstreamPlaningSurface"))
        self.neighbor_down = PlaningSurface.find_by_name(dict_.get("downstreamPlaningSurface"))

        if self.neighbor_down is not None:
            self.neighbor_down.upstream_pressure = self.cushion_pressure
        if self.neighbor_up is not None:
            self.neighbor_up.kutta_pressure = self.cushion_pressure

        # TODO: cushion_type variability should be managed by sub-classing PressurePatch
        self.cushion_type = dict_.get("cushionType", "")
        if self.cushion_type == "infinite":
            # Dummy element, will have 0 pressure
            self.pressure_elements.append(
                pe.AftSemiInfinitePressureBand(is_source=True, is_on_body=False)
            )
            self.pressure_elements.append(
                pe.AftSemiInfinitePressureBand(is_source=True, is_on_body=False)
            )
            self._end_pts[0] = -1000.0  # doesn't matter where
        else:
            self.pressure_elements.append(
                pe.ForwardHalfTriangularPressureElement(is_source=True, is_on_body=False)
            )

            self.smoothing_factor = dict_.get("smoothingFactor", np.nan)
            # TODO: Are we adding these elements twice?
            for n in [self.neighbor_down, self.neighbor_up]:
                if n is None and ~np.isnan(self.smoothing_factor):
                    self.pressure_elements.extend(
                        [
                            pe.CompleteTriangularPressureElement(is_source=True, is_on_body=False)
                            for _ in range(dict_.get("numElements", 10))
                        ]
                    )

            for i, key in enumerate(["downstreamLoc", "upstreamLoc"]):
                value = dict_.get(key)
                if value is None:
                    value = getattr(config, key, 0.0)
                self._end_pts[i] = value

            self.pressure_elements.append(
                pe.AftHalfTriangularPressureElement(is_source=True, is_on_body=False)
            )
        self.update_end_pts()

    @property
    def base_pt(self) -> float:
        """The x-coordinate of the base point at the upstream edge of the pressure cushion."""
        if self.neighbor_up is None:
            return self._end_pts[1]
        else:
            return self.neighbor_up.base_pt

    @base_pt.setter
    def base_pt(self, x: float) -> None:
        self._end_pts[1] = x

    @PressurePatch.length.setter
    def length(self, length):
        """Set the length by moving the left end point relative to the right one."""
        self._end_pts[0] = self.base_pt - length

    def get_element_coords(self) -> np.ndarray:
        """Return x-locations of all elements."""
        if self.cushion_type != "smoothed" or np.isnan(self.smoothing_factor):
            return self._end_pts
        else:
            add_width = np.arctanh(0.99) * self.length / (2 * self.smoothing_factor)
            dx = np.linspace(-add_width, add_width, len(self.pressure_elements) // 2 + 1)
            return np.hstack((self._end_pts[0] + dx, self._end_pts[1] + dx))

    def update_end_pts(self) -> None:
        """Update the end coordinates."""
        if self.neighbor_down is not None:
            self._end_pts[0] = self.neighbor_down._end_pts[1]
        if self.neighbor_up is not None:
            self._end_pts[1] = self.neighbor_up._end_pts[0]

        self._reset_element_coords()

        # Set source pressure of elements
        for el_num, el in enumerate(self.pressure_elements):
            el.is_source = True
            if self.cushion_type == "smoothed" and not np.isnan(self.smoothing_factor):
                alf = 2 * self.smoothing_factor / self.length
                el.pressure = (
                    0.5
                    * self.cushion_pressure
                    * (np.tanh(alf * el.x_coord) - np.tanh(alf * (el.x_coord - self.length)))
                )
            # for infinite pressure cushion, first element is dummy, set to
            # zero, second is semiInfinitePressureBand and set to cushion
            # pressure
            elif self.cushion_type == "infinite":
                el.pressure = 0.0 if el_num == 0 else self.cushion_pressure
            else:
                el.pressure = self.cushion_pressure

    def calculate_forces(self) -> None:
        """Calculate the wave drag of the pressure cushion."""
        self.drag_wave = self._calculate_wave_drag()


class PlaningSurface(PressurePatch):
    """Planing Surface consisting of unknown elements."""

    count = 0
    all = []

    @classmethod
    def find_by_name(cls, name: Optional[str]) -> Optional["PlaningSurface"]:
        """Return first planing surface matching provided name.

        Returns
        -------
        PlaningSurface instance or None of no match found
        """
        if name:
            for o in cls.all:
                if o.patch_name == name:
                    return o
        return None

    def __init__(self, dict_, **kwargs):
        PressurePatch.__init__(self)
        self.index = PlaningSurface.count
        PlaningSurface.count += 1
        PlaningSurface.all.append(self)

        self.patch_name = dict_.get("substructureName", "")
        Nfl = dict_.get("Nfl", 0)
        self.point_spacing = dict_.get("pointSpacing", "linear")
        self.initial_length = dict_.get("initialLength", None)
        self.minimum_length = dict_.get("minimumLength", 0.0)
        self.maximum_length = dict_.get("maximum_length", float("Inf"))

        self.spring_constant = dict_.get("springConstant", 1e4)
        self.kutta_pressure = kwargs.get("kuttaPressure", dict_.get("kuttaPressure", 0.0))
        if isinstance(self.kutta_pressure, str):
            self.kutta_pressure = getattr(config, self.kutta_pressure)
        self._upstream_pressure = kwargs.get("upstreamPressure", dict_.get("upstreamPressure", 0.0))
        if isinstance(self._upstream_pressure, str):
            self._upstream_pressure = getattr(config, self._upstream_pressure)

        self.is_initialized = False
        self.is_active = True
        self.is_kutta_unknown = True
        self.is_sprung = dict_.get("isSprung", False)
        if self.is_sprung:
            self.initial_length = 0.0
            self.minimum_length = 0.0
            self.maximum_length = 0.0

        self.pressure_elements += [pe.ForwardHalfTriangularPressureElement(is_on_body=False)]

        self.pressure_elements += [
            pe.ForwardHalfTriangularPressureElement(is_source=True, pressure=self.kutta_pressure)
        ]
        self.pressure_elements += [pe.CompleteTriangularPressureElement() for __ in range(Nfl - 1)]
        self.pressure_elements += [
            pe.AftHalfTriangularPressureElement(is_source=True, pressure=self.upstream_pressure)
        ]

        for el in self.pressure_elements:
            el.parent = self

        # Define point spacing
        if self.point_spacing == "cosine":
            self.pct = 0.5 * (1 - trig.cosd(np.linspace(0, 180, Nfl + 1)))
        else:
            self.pct = np.linspace(0.0, 1.0, Nfl + 1)
        self.pct /= self.pct[-2]
        self.pct = np.hstack((0.0, self.pct))

    @property
    def upstream_pressure(self):
        if self.neighbor_up is not None:
            return self.neighbor_up.cushion_pressure
        else:
            return 0.0

    @property
    def downstream_pressure(self):
        if self.neighbor_down is not None:
            return self.neighbor_down.cushion_pressure
        else:
            return 0.0

    @upstream_pressure.setter
    def upstream_pressure(self, pressure):
        self._upstream_pressure = pressure
        el = self.pressure_elements[-1]
        el.pressure = pressure
        el.is_source = True
        el.z_coord = np.nan

    @PressurePatch.length.setter
    def length(self, length):
        length = np.min([np.max([length, 0.0]), self.maximum_length])

        x0 = self.interpolator.get_separation_point()[0]
        self._end_pts = [x0, x0 + length]
        self._reset_element_coords()

    def initialize_end_pts(self):
        """Initialize end points to be at separation point and wetted length
        root.
        """
        self.base_pt = self.interpolator.get_separation_point()[0]

        if not self.is_initialized:
            if self.initial_length is None:
                self.length = self.interpolator.immersed_length - self.base_pt
            else:
                self.length = self.initial_length
            self.is_initialized = True

    def _reset_element_coords(self) -> None:
        """Set width of first element to be twice as wide."""
        PressurePatch._reset_element_coords(self)
        x = self.get_element_coords()
        self.pressure_elements[0].width = x[2] - x[0]

    def get_element_coords(self) -> np.ndarray:
        """Get position of pressure elements."""
        return self.pct * self.length + self._end_pts[0]

    def get_residual(self) -> float:
        """Get residual to drive first element pressure to zero."""
        if self.length <= 0.0:
            return 0.0
        else:
            return self.pressure_elements[0].pressure / config.flow.stagnation_pressure

    def calculate_forces(self):
        # TODO: Re-factor, this is messy."
        if self.length > 0.0:
            el = [el for el in self.pressure_elements if el.is_on_body]
            self.x = np.array([eli.x_coord for eli in el])
            self.p = np.array([eli.pressure for eli in el])
            self.p += self.pressure_elements[0].pressure
            self.shear_stress = np.array([eli.shear_stress for eli in el])
            self.s = np.array([self.interpolator.get_s_fixed_x(xx) for xx in self.x])

            self.fP = interp1d(self.x, self.p, bounds_error=False, fill_value=0.0)
            self.fTau = interp1d(self.x, self.shear_stress, bounds_error=False, fill_value=0.0)

            AOA = trig.atand(self._get_body_derivative(self.x))

            self.drag_pressure = general.integrate(self.s, self.p * trig.sind(AOA))
            self.drag_friction = general.integrate(self.s, self.shear_stress * trig.cosd(AOA))
            self.lift_pressure = general.integrate(self.s, self.p * trig.cosd(AOA))
            self.lift_friction = -general.integrate(self.s, self.shear_stress * trig.sind(AOA))
            self.drag_total = self.drag_pressure + self.drag_friction
            self.lift_total = self.lift_pressure + self.lift_friction
            self.moment_total = general.integrate(
                self.x, self.p * trig.cosd(AOA) * (self.x - config.body.xCofR)
            )
        else:
            self.drag_pressure = 0.0
            self.drag_friction = 0.0
            self.lift_pressure = 0.0
            self.lift_friction = 0.0
            self.drag_total = 0.0
            self.lift_total = 0.0
            self.moment_total = 0.0
            self.x = []
        if self.is_sprung:
            self._apply_spring()

        if config.flow.include_friction:
            self._calculate_shear_stress()

        self.drag_wave = self._calculate_wave_drag()

    def get_loads_in_range(self, x0, x1):
        """Return pressure and shear stress values at points between two
        arguments.
        """
        # Get indices within range unless length is zero
        if self.length > 0.0:
            ind = np.nonzero((self.x > x0) * (self.x < x1))[0]
        else:
            ind = []

        # Output all values within range
        if not ind == []:
            x = np.hstack((x0, self.x[ind], x1))
            p = np.hstack((self.fP(x0), self.p[ind], self.fP(x1)))
            tau = np.hstack((self.fTau(x0), self.shear_stress[ind], self.fTau(x1)))
        else:
            x = np.array([x0, x1])
            p = np.zeros_like(x)
            tau = np.zeros_like(x)
        return x, p, tau

    def _apply_spring(self):
        xs = self.pressure_elements[0].x_coord
        zs = self.pressure_elements[0].z_coord
        disp = zs - self.parent.get_free_surface_height(xs)
        Fs = -self.spring_constant * disp
        self.lift_total += Fs
        self.moment_total += Fs * (xs - config.body.xCofR)

    def _calculate_shear_stress(self):
        def shear_stress_func(xx):
            if xx == 0.0:
                return 0.0
            else:
                Rex = config.flow.flow_speed * xx / config.flow.kinematic_viscosity
                return 0.332 * config.flow.density * config.flow.flow_speed ** 2 * Rex ** -0.5

        x = self.get_element_coords()[0:-1]
        s = np.array([self.interpolator.get_s_fixed_x(xx) for xx in x])
        s = s[-1] - s

        for si, el in zip(s, self.pressure_elements):
            el.shear_stress = shear_stress_func(si)

    def _get_body_derivative(self, x, direction="r"):
        if isinstance(x, float):
            x = [x]
        return np.array(
            [general.getDerivative(self.interpolator.get_body_height, xx, direction) for xx in x]
        )
