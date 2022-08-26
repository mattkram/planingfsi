"""Classes representing a pressure patch on the free surface."""
from __future__ import annotations

import abc
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fmin

from planingfsi import math_helpers
from planingfsi import trig
from planingfsi import writers
from planingfsi.config import Config
from planingfsi.dictionary import load_dict_from_file
from planingfsi.potentialflow import pressureelement as pe

if TYPE_CHECKING:
    from planingfsi.fe.substructure import Interpolator
    from planingfsi.potentialflow.solver import PotentialPlaningSolver


class PressurePatch(abc.ABC):
    """Abstract base class representing a patch of pressure elements on the free surface.

    Attributes:
        pressure_elements: List of pressure elements.
        is_kutta_unknown: True of trailing edge pressure unknown.
        interpolator: Object to get interpolated body position if a `PlaningSurface`
        parent: Parent solver this patch belongs to.

    """

    name: str

    def __init__(self, *, parent: PotentialPlaningSolver | None) -> None:
        self.parent = parent
        self.pressure_elements: list[pe.PressureElement] = []
        self._end_pts = np.zeros(2)
        self.is_kutta_unknown = False
        self._neighbor_up: PressurePatch | None = None
        self._neighbor_down: PressurePatch | None = None
        self.interpolator: Interpolator | None = None

        self.drag_total = np.nan
        self.drag_pressure = np.nan
        self.drag_friction = np.nan
        self.lift_total = np.nan
        self.lift_pressure = np.nan
        self.lift_friction = np.nan
        self.moment_total = np.nan

    @property
    def config(self) -> Config:
        """A reference to the simulation configuration."""
        if self.parent is None:
            raise ValueError(
                "The parent must be set to access its configuration. "
                "This can be done either with the 'parent=parent' keyword argument, by setting it explicitly, "
                "or by using the `.add_planing_surface()` method."
            )
        return self.parent.config

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
    def neighbor_up(self) -> PressurePatch | None:
        """PressurePatch instance upstream of this one.

        When setting, this patch is set as the other's downstream neighbor.

        """
        return self._neighbor_up

    @neighbor_up.setter
    def neighbor_up(self, obj: PressurePatch | None) -> None:
        self._neighbor_up = obj
        if self._neighbor_up is not None:
            self._neighbor_up._neighbor_down = self

    @property
    def neighbor_down(self) -> PressurePatch | None:
        """PressurePatch instance downstream of this one.

        When setting, this patch is set as the other's upstream neighbor.

        """
        return self._neighbor_down

    @neighbor_down.setter
    def neighbor_down(self, obj: PressurePatch | None) -> None:
        self._neighbor_down = obj
        if self._neighbor_down is not None:
            self._neighbor_down._neighbor_up = self

    @property
    @abc.abstractmethod
    def element_coords(self) -> np.ndarray:
        """Get the x-position of pressure elements."""
        return NotImplemented

    def _reset_element_coords(self) -> None:
        """Re-distribute pressure element positions along the patch."""
        x = self.element_coords
        for i, el in enumerate(self.pressure_elements):
            el.x_coord = x[i]
            if not el.is_source and self.interpolator is not None:
                el.z_coord = self.interpolator.get_body_height(x[i])

            if isinstance(el, pe.CompleteTriangularPressureElement):
                # TODO: Remove ignore
                el.width = np.array([x[i] - x[i - 1], x[i + 1] - x[i]])  # type: ignore
            elif isinstance(el, pe.ForwardHalfTriangularPressureElement):
                el.width = x[i + 1] - x[i]  # type: ignore
            elif isinstance(el, pe.AftHalfTriangularPressureElement):
                el.width = x[i] - x[i - 1]  # type: ignore
            elif isinstance(el, pe.AftSemiInfinitePressureBand):
                el.width = np.inf  # type: ignore
            else:
                raise ValueError("Invalid Element Type!")

    def get_free_surface_height(self, x: float) -> float:
        """Get free surface height at a position x due to the elements on this patch.

        Args:
            x: x-position at which to calculate free-surface height.

        """
        return sum(el.get_influence(x) for el in self.pressure_elements)

    def calculate_forces(self) -> None:
        """Calculate the force components for this pressure patch."""

    @property
    def drag_wave(self) -> float:
        """Calculate wave drag of patch."""
        f = self.get_free_surface_height
        x_init = -10 * self.config.flow.lam
        (x_trough,) = fmin(f, x_init, disp=False)
        (x_crest,) = fmin(lambda x: -f(x), x_init, disp=False)
        return (
            0.0625
            * self.config.flow.density
            * self.config.flow.gravity
            * (f(x_crest) - f(x_trough)) ** 2
        )

    @property
    def _force_file_save_path(self) -> Path:
        """A path to the force file."""
        assert self.parent is not None
        return self.parent.simulation.it_dir / f"forces_{self.name}.{self.config.io.data_format}"

    def write_forces(self) -> None:
        """Write forces to file."""
        writers.write_as_dict(
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
        name: Name of patch.
        cushion_pressure: The value of the pressure in the cushion. If "Pc" is provided, the value will be
            loaded from the simulation-global `config.body.Pc` variable.
        cushion_type: The type of pressure cushion, either "finite" or "infinite".
        smoothing_factor: Used for standalone finite pressure cushions without upstream and downstream surfaces.
        num_elements: The number of pressure elements used to represent the pressure cushion.
        upstream_planing_surface: The `PlaningSurface` upstream of the cushion, used to determine bounds.
        downstream_planing_surface: The `PlaningSurface` downstream of the cushion, used to determine bounds.
        upstream_loc: If there is no upstream `PlaningSurface` instance, this is the fixed upstream bound.
        downstream_loc: If there is no downstream `PlaningSurface` instance, this is the fixed downstream bound.
        parent: The `PotentialPlaningSolver` to which this `PressureCushion` belongs.

    """

    _count = 0

    def __init__(
        self,
        *,
        name: str = "",
        cushion_pressure: float | Literal["Pc"] | None = None,
        cushion_type: Literal["finite"] | Literal["infinite"] = "finite",
        smoothing_factor: float = np.nan,
        num_elements: int = 10,
        upstream_planing_surface: PlaningSurface | str | None = None,
        downstream_planing_surface: PlaningSurface | str | None = None,
        upstream_loc: float | None = None,
        downstream_loc: float | None = None,
        parent: PotentialPlaningSolver | None = None,
        **_: Any,
    ) -> None:
        super().__init__(parent=parent)
        PressureCushion._count += 1
        self.name = name or f"pressureCushion{PressureCushion._count}"

        if cushion_pressure == "Pc" or cushion_pressure is None:
            self.cushion_pressure = getattr(self.config.body, "Pc", 0.0)
        else:
            self.cushion_pressure = cushion_pressure

        if isinstance(upstream_planing_surface, PlaningSurface):
            self.neighbor_up = upstream_planing_surface
        else:
            self.neighbor_up = [
                surf
                for surf in self.parent.planing_surfaces
                if surf.name == upstream_planing_surface
            ][0]

        if isinstance(downstream_planing_surface, PlaningSurface):
            self.neighbor_down = downstream_planing_surface
        else:
            self.neighbor_down = [
                surf
                for surf in self.parent.planing_surfaces
                if surf.name == downstream_planing_surface
            ][0]

        if self.neighbor_down is not None:
            self.neighbor_down.upstream_pressure = self.cushion_pressure
        if self.neighbor_up is not None:
            self.neighbor_up.kutta_pressure = self.cushion_pressure

        # TODO: cushion_type variability should be managed by sub-classing PressurePatch
        self.cushion_type = cushion_type
        if self.cushion_type == "infinite":
            # Dummy element, will have 0 pressure
            self.pressure_elements.append(
                pe.AftSemiInfinitePressureBand(is_source=True, is_on_body=False, parent=self)
            )
            self.pressure_elements.append(
                pe.AftSemiInfinitePressureBand(is_source=True, is_on_body=False, parent=self)
            )
            self._end_pts[0] = -1000.0  # doesn't matter where
        else:
            self.pressure_elements.append(
                pe.ForwardHalfTriangularPressureElement(
                    is_source=True, is_on_body=False, parent=self
                )
            )

            self.smoothing_factor = smoothing_factor
            # TODO: Are we adding these elements twice?
            for n in [self.neighbor_down, self.neighbor_up]:
                if n is None and ~np.isnan(self.smoothing_factor):
                    self.pressure_elements.extend(
                        [
                            pe.CompleteTriangularPressureElement(
                                is_source=True, is_on_body=False, parent=self
                            )
                            for _ in range(num_elements)
                        ]
                    )

            if downstream_planing_surface is None:
                self._end_pts[0] = (
                    downstream_loc
                    if downstream_loc is not None
                    else getattr(self.config, "downstream_loc", 0.0)
                )

            if upstream_planing_surface is None:
                self._end_pts[1] = (
                    upstream_loc
                    if upstream_loc is not None
                    else getattr(self.config, "upstream_loc", 0.0)
                )

            self.pressure_elements.append(
                pe.AftHalfTriangularPressureElement(is_source=True, is_on_body=False, parent=self)
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

    @property
    def length(self) -> float:
        """Length of pressure cushion."""
        return super().length

    @length.setter
    def length(self, length: float) -> None:
        """Set the length by moving the left end point relative to the right one."""
        self._end_pts[0] = self.base_pt - length

    @property
    def element_coords(self) -> np.ndarray:
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


class PlaningSurface(PressurePatch):
    """Planing Surface consisting of unknown elements.

    Attributes:
        name: The name of the planing surface (should be unique, but that is unenforced).
        initial_length: An initial guessed wetted length of the planing surface.
        minimum_length: The minimum length of the surface allowed to be wetted.
        maximum_length: The maximum length of the surface allowed to be wetted.
        num_fluid_elements: The number of elements to discretize surface into.
        point_spacing: Method to use for distributing pressure elements. Either "linear" or "cosine".
        kutta_pressure: The target pressure at the trailing edge.
        upstream_pressure: The target pressure upstream of the stagnation point.
        is_sprung: If true, contribution due to numerical spring is added to the lift force in order to ensure
            the tip of the surface lies on the free surface.
        spring_constant: The spring constant.
        parent: The solver to which this planing surface belongs.

    """

    def __init__(
        self,
        *,
        name: str = "",
        initial_length: float = 0.1,
        minimum_length: float = 0.0,
        maximum_length: float = float("Inf"),
        num_fluid_elements: int = 1,
        point_spacing: Literal["linear"] | Literal["cosine"] = "linear",
        kutta_pressure: float | str = 0.0,
        upstream_pressure: float = 0.0,
        is_sprung: bool = False,
        spring_constant: float = 1e4,
        parent: PotentialPlaningSolver | None = None,
        **_: Any,
    ) -> None:
        super().__init__(parent=parent)

        self.name = name
        self.initial_length = initial_length
        self.minimum_length = minimum_length
        self.maximum_length = maximum_length

        self.kutta_pressure = kutta_pressure
        if isinstance(self.kutta_pressure, str):
            self.kutta_pressure = getattr(self.config.body, self.kutta_pressure)

        self._upstream_pressure = upstream_pressure
        if isinstance(self._upstream_pressure, str):
            self._upstream_pressure = getattr(self.config, self._upstream_pressure)

        self.is_initialized = False
        self.is_active = True
        self.is_kutta_unknown = True

        self.is_sprung = is_sprung
        self.spring_constant = spring_constant
        if self.is_sprung:
            self.initial_length = 0.0
            self.minimum_length = 0.0
            self.maximum_length = 0.0

        self.pressure_elements.append(
            pe.ForwardHalfTriangularPressureElement(parent=self, is_on_body=False)
        )
        self.pressure_elements.append(
            pe.ForwardHalfTriangularPressureElement(
                parent=self, is_source=True, pressure=self.kutta_pressure
            )
        )
        self.pressure_elements.extend(
            [
                pe.CompleteTriangularPressureElement(parent=self)
                for _ in range(num_fluid_elements - 1)
            ]
        )
        self.pressure_elements.append(
            pe.AftHalfTriangularPressureElement(
                parent=self, is_source=True, pressure=self.upstream_pressure
            )
        )

        # Define point spacing
        self.relative_position: np.ndarray
        if point_spacing == "cosine":
            # TODO: Remove ignore
            self.relative_position = 0.5 * (  # type: ignore
                1 - trig.cosd(np.linspace(0.0, 180.0, num_fluid_elements + 1))
            )
        elif point_spacing == "linear":
            self.relative_position = np.linspace(0.0, 1.0, num_fluid_elements + 1)
        else:
            raise ValueError("point_spacing must be 'cosine' or 'linear'")

        self.relative_position /= self.relative_position[-2]
        self.relative_position = np.hstack((0.0, self.relative_position))

        self.x_coords = np.array([])
        self.pressure = np.array([])
        self.shear_stress = np.array([])
        self.s_coords = np.array([])

    @property
    def upstream_pressure(self) -> float:
        """The pressure at the upstream end of the planing surface."""
        if isinstance(self.neighbor_up, PressureCushion):
            return self.neighbor_up.cushion_pressure
        else:
            return 0.0

    @upstream_pressure.setter
    def upstream_pressure(self, pressure: float) -> None:
        """When setting the upstream pressure, that element is now a source."""
        self._upstream_pressure = pressure
        el = self.pressure_elements[-1]
        el.pressure = pressure
        el.is_source = True
        el.z_coord = np.nan

    @property
    def downstream_pressure(self) -> float:
        """The pressure at the downstream end of the planing surface."""
        if isinstance(self.neighbor_down, PressureCushion):
            return self.neighbor_down.cushion_pressure
        else:
            return 0.0

    @property
    def length(self) -> float:
        """Length of planing surface."""
        return super().length

    @length.setter
    def length(self, length: float) -> None:
        """Set the length and re-distribute the elements with the base at the separation point."""
        length = np.min([np.max([length, 0.0]), self.maximum_length])
        assert self.interpolator is not None
        x0 = self.interpolator.get_separation_point()[0]
        self._end_pts[:] = [x0, x0 + length]
        self._reset_element_coords()

    @property
    def residual(self) -> float:
        """The residual to drive first element pressure to zero."""
        if self.length > 0.0:
            return self.pressure_elements[0].pressure / self.config.flow.stagnation_pressure
        else:
            return 0.0

    def initialize_end_pts(self) -> None:
        """Initialize end points to be at separation point and wetted length root."""
        assert self.interpolator is not None
        self.base_pt = self.interpolator.get_separation_point()[0]

        if not self.is_initialized:
            if self.initial_length is None:
                self.length = self.interpolator.immersed_length - self.base_pt
            else:
                self.length = self.initial_length
            self.is_initialized = True

    def _reset_element_coords(self) -> None:
        """Set width of first element to be twice as wide."""
        super()._reset_element_coords()
        x = self.element_coords
        self.pressure_elements[0].width = x[2] - x[0]

    @property
    def element_coords(self) -> np.ndarray:
        """Get position of pressure elements."""
        return self.base_pt + self.relative_position * self.length

    def calculate_forces(self) -> None:
        """Calculate the forces by integrating pressure and shear stress."""
        if self.config.flow.include_friction:
            self._calculate_shear_stress()

        if self.length > 0.0:
            el = [el for el in self.pressure_elements if el.is_on_body]
            self.x_coords = np.array([eli.x_coord for eli in el])
            self.pressure = np.array([eli.pressure for eli in el])
            self.pressure += self.pressure_elements[0].pressure
            self.shear_stress = np.array([eli.shear_stress for eli in el])
            assert self.interpolator is not None
            self.s_coords = np.array([self.interpolator.get_s_fixed_x(xx) for xx in self.x_coords])

            tangent_angle = trig.atand(self._get_body_derivative(self.x_coords))
            self.drag_pressure = math_helpers.integrate(
                self.s_coords, self.pressure * trig.sind(tangent_angle)
            )
            self.drag_friction = math_helpers.integrate(
                self.s_coords, self.shear_stress * trig.cosd(tangent_angle)
            )
            self.drag_total = self.drag_pressure + self.drag_friction
            self.lift_pressure = math_helpers.integrate(
                self.s_coords, self.pressure * trig.cosd(tangent_angle)
            )
            self.lift_friction = -math_helpers.integrate(
                self.s_coords, self.shear_stress * trig.sind(tangent_angle)
            )
            self.lift_total = self.lift_pressure + self.lift_friction
            self.moment_total = math_helpers.integrate(
                self.x_coords,
                # TODO: This shouldn't reference body config
                self.pressure * trig.cosd(tangent_angle) * (self.x_coords - self.config.body.xCofR),
            )
            if self.is_sprung:
                self._apply_spring()
        else:
            self.drag_pressure = 0.0
            self.drag_friction = 0.0
            self.drag_total = 0.0
            self.lift_pressure = 0.0
            self.lift_friction = 0.0
            self.lift_total = 0.0
            self.moment_total = 0.0
            self.x_coords = np.array([])

    def get_loads_in_range(
        self, x0: float, x1: float, /, *, pressure_limit: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return pressure and shear stress values at points between two
        arguments.

        Args:
            x0: The starting x-coordinate.
            x1: The ending x-coordinate.
            pressure_limit: Optionally limit the pressure to be below a specified value (e.g. stagnation pressure).
        """
        if self.length > 0.0:
            ind = np.nonzero((self.x_coords > x0) * (self.x_coords < x1))[0]
            interp_p = interp1d(self.x_coords, self.pressure, bounds_error=False, fill_value=0.0)
            interp_tau = interp1d(
                self.x_coords, self.shear_stress, bounds_error=False, fill_value=0.0
            )

            x = np.hstack((x0, self.x_coords[ind], x1))
            p = np.hstack((interp_p(x0), self.pressure[ind], interp_p(x1)))
            tau = np.hstack((interp_tau(x0), self.shear_stress[ind], interp_tau(x1)))
        else:
            x = np.array([x0, x1])
            p = np.zeros_like(x)
            tau = np.zeros_like(x)

        if pressure_limit is not None:
            # Limit pressure to be below stagnation pressure
            p = np.min(
                np.vstack(
                    (
                        p,
                        np.ones_like(p) * pressure_limit,
                    )
                ),
                axis=0,
            )
        return x, p, tau

    def _apply_spring(self) -> None:
        """Apply the spring force to the total lift and moment forces."""
        x_s = self.pressure_elements[0].x_coord
        z_s = self.pressure_elements[0].z_coord
        assert self.parent is not None
        spring_displacement = z_s - self.parent.get_free_surface_height(x_s)
        spring_force = -self.spring_constant * spring_displacement
        self.lift_total += spring_force
        # TODO: This shouldn't reference body config
        self.moment_total += spring_force * (x_s - self.config.body.xCofR)

    def _calculate_shear_stress(self) -> None:
        """Calculate the shear stress and apply to each element."""

        def get_shear_stress(xx: float) -> float:
            """Calculate the shear stress at a given location."""
            re_x = self.config.flow.flow_speed * xx / self.config.flow.kinematic_viscosity
            return (
                0.332 * self.config.flow.density * self.config.flow.flow_speed**2 * re_x**-0.5
            )

        x = self.element_coords[:-1]
        assert self.interpolator is not None
        s = np.array([self.interpolator.get_s_fixed_x(xx) for xx in x])
        s = s[-1] - s

        for s_i, el in zip(s, self.pressure_elements):
            el.shear_stress = get_shear_stress(s_i) if s_i > 0.0 else 0.0

    def _get_body_derivative(self, x: np.ndarray, direction: str = "r") -> np.ndarray:
        """Calculate the derivative of the body surface at a point."""
        assert self.interpolator is not None
        return np.array(
            [math_helpers.deriv(self.interpolator.get_body_height, xx, direction) for xx in x]
        )
