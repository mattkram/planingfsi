"""Module containing definitions of different types of pressure element."""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from scipy.special import sici

from planingfsi import math_helpers
from planingfsi.config import Config

if TYPE_CHECKING:
    from planingfsi.potentialflow.pressurepatch import PressurePatch


def _get_aux_fg(lam: float) -> tuple[float, float]:
    """Return f and g functions, which are dependent on the auxiliary
    sine and cosine functions. Combined into one function to reduce number of
    calls to scipy.special.sici().
    """
    lam = abs(lam)
    (aux_sine, aux_cosine) = sici(lam)
    (sine, cosine) = np.sin(lam), np.cos(lam)
    return (
        cosine * (0.5 * np.pi - aux_sine) + sine * aux_cosine,
        sine * (0.5 * np.pi - aux_sine) - cosine * aux_cosine,
    )


def _get_gamma1(lam: float, aux_g: float) -> float:
    """Return result of first integral (Gamma_1 in my thesis).

    Args:
        lam: Non-dimensional x-position.
        aux_g: Second result of `_get_aux_fg`.

    """
    if lam > 0:
        return (aux_g + np.log(lam)) / np.pi
    else:
        return (aux_g + np.log(-lam)) / np.pi + (2 * np.sin(lam) - lam)


def _get_gamma2(lam: float, aux_f: float) -> float:
    """Return result of second integral (Gamma_2 in my thesis).

    Args
    ----
    lam : float
        Non-dimensional x-position.
    aux_f : float
        First result of `_get_aux_fg`

    Returns
    -------
    float

    """
    return math_helpers.sign(lam) * aux_f / np.pi + math_helpers.heaviside(-lam) * (
        2 * np.cos(lam) - 1
    )


def _get_gamma3(lam: float, aux_f: float) -> float:
    """Return result of third integral (Gamma_3 in my thesis).

    Args
    ----
    lam : float
        Non-dimensional x-position.
    aux_f : float
        First result of `_get_aux_fg`

    Returns
    -------
    float
    """
    return (
        -math_helpers.sign(lam) * aux_f / np.pi
        - 2 * math_helpers.heaviside(-lam) * np.cos(lam)
        - math_helpers.heaviside(lam)
    )


class PressureElement(abc.ABC):
    """Abstract base class to represent all different types of pressure elements.

    Attributes:
        x_coord: x-location of element.
        z_coord: z-coordinate of element, if on body.
        pressure: The pressure/strength of the element.
        shear_stress: Shear stress at the element.
        width: Width of element.
        is_source: True of element is a source.
        is_on_body: True if on body. Used for force calculation.
        parent: Pressure patch that this element belongs to.

    """

    plot_color = "b"

    def __init__(
        self,
        x_coord: float = np.nan,
        z_coord: float = np.nan,
        pressure: float = np.nan,
        shear_stress: float = 0.0,
        width: np.ndarray | float = np.nan,
        is_source: bool = False,
        is_on_body: bool = False,
        parent: PressurePatch | None = None,
    ):
        # TODO: Replace x_coord with coord pointer to Coordinate object, (e.g. self.coords.x)
        self.x_coord = x_coord
        self.z_coord = z_coord
        self.pressure = pressure
        self.shear_stress = shear_stress
        self._width = np.zeros(2)
        # TODO: Remove ignore
        self.width = width  # type: ignore
        self.is_source: bool = is_source
        self.is_on_body: bool = is_on_body
        self.parent = parent

    @property
    def config(self) -> Config:
        """A reference to the simulation configuration."""
        if self.parent is None:
            raise ValueError("Must set parent pressure patch to access config.")
        return self.parent.config

    @property
    def width(self) -> float:
        """The total width of the pressure element."""
        return self._width.sum()

    @width.setter
    def width(self, width: float) -> None:
        self._width[0] = width

    def get_influence_coefficient(self, x_coord: float) -> float:
        """Return _get_local_influence_coefficient coefficient of element.

        Args:
            x_coord: Dimensional x-coordinate.

        """
        x_rel = x_coord - self.x_coord
        if self.width == 0.0:
            return 0.0
        elif x_rel in [-self._width[0], 0.0, self._width[1]]:
            dx = 1e-6
            influence = 0.5 * (
                self._get_local_influence_coefficient(x_rel + dx)
                + self._get_local_influence_coefficient(x_rel - dx)
            )
        else:
            influence = self._get_local_influence_coefficient(x_rel)
        return influence / (self.config.flow.density * self.config.flow.gravity)

    def get_influence(self, x_coord: float) -> float:
        """Return _get_local_influence_coefficient for actual pressure.

        Args:
            x_coord: Dimensional x-coordinate.

        """
        return self.get_influence_coefficient(x_coord) * self.pressure

    @abc.abstractmethod
    def _get_local_influence_coefficient(self, x_coord: float) -> float:
        """Return influence coefficient in iso-geometric coordinates.

        Args:
            x_coord: x-coordinate, centered at element location.

        """
        pass

    def __repr__(self) -> str:
        """Print element attributes."""
        return ", ".join(
            [
                f"{self.__class__.__name__}: (x,z) = ({self.x_coord}, {self.z_coord})",
                f"width = {self.width}",
                f"is_source = {self.is_source}",
                f"p = {self.pressure}",
                f"is_on_body = {self.is_on_body}",
            ]
        )

    @property
    def plot_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates for pressure plot."""
        return np.array([]), np.array([])


class AftHalfTriangularPressureElement(PressureElement):
    """Pressure element that is triangular in profile but towards aft direction."""

    plot_color = "g"

    def __init__(self, is_on_body: bool = True, **kwargs: Any) -> None:
        # By default, this element is on the body.
        super().__init__(is_on_body=is_on_body, **kwargs)

    @property
    def width(self) -> float:
        """The total width of the pressure element."""
        return super().width

    @width.setter
    def width(self, width: float) -> None:
        self._width[0] = width

    def _get_local_influence_coefficient(self, x_rel: float) -> float:
        """Get influence coefficient for element relative to the element position."""
        lambda_0 = self.config.flow.k0 * x_rel
        aux_f, aux_g = _get_aux_fg(lambda_0)
        influence = _get_gamma1(lambda_0, aux_g) / (self.width * self.config.flow.k0)
        influence += _get_gamma2(lambda_0, aux_f)

        lambda_2 = self.config.flow.k0 * (x_rel + self.width)
        _, aux_g = _get_aux_fg(lambda_2)
        influence -= _get_gamma1(lambda_2, aux_g) / (self.width * self.config.flow.k0)
        return influence

    @property
    def plot_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates for pressure plot."""
        return self.x_coord - np.array([self.width, 0]), np.array([0.0, self.pressure])


class ForwardHalfTriangularPressureElement(PressureElement):
    """Pressure element that is triangular in profile but towards forward direction."""

    def __init__(self, is_on_body: bool = True, **kwargs: Any) -> None:
        # By default, this element is on the body.
        super().__init__(is_on_body=is_on_body, **kwargs)

    @property
    def width(self) -> float:
        """The total width of the pressure element."""
        return super().width

    @width.setter
    def width(self, width: float) -> None:
        self._width[1] = width

    def _get_local_influence_coefficient(self, x_coord: float) -> float:
        """Return _get_local_influence_coefficient coefficient in iso-geometric coordinates.

        Args:
            x_coord: x-coordinate, centered at element location.

        """
        lambda_0 = self.config.flow.k0 * x_coord
        aux_f, aux_g = _get_aux_fg(lambda_0)
        influence = _get_gamma1(lambda_0, aux_g) / (self.width * self.config.flow.k0)
        influence -= _get_gamma2(lambda_0, aux_f)

        lambda_1 = self.config.flow.k0 * (x_coord - self.width)
        _, aux_g = _get_aux_fg(lambda_1)
        influence -= _get_gamma1(lambda_1, aux_g) / (self.width * self.config.flow.k0)
        return influence

    @property
    def plot_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates for pressure plot."""
        return self.x_coord + np.array([0.0, self.width]), np.array([self.pressure, 0.0])


class CompleteTriangularPressureElement(PressureElement):
    """Pressure element that is triangular in profile towards both directions."""

    plot_color = "r"

    def __init__(
        self, is_on_body: bool = True, width: np.ndarray = np.zeros(2), **kwargs: Any
    ) -> None:
        # By default, this element is on the body.
        super().__init__(is_on_body=is_on_body, width=width, **kwargs)

    @property
    def width(self) -> float:
        """The total width of the pressure element."""
        return super().width

    @width.setter
    def width(self, width: np.ndarray) -> None:
        if not len(width) == 2:
            raise ValueError("Width must be length-two array")
        else:
            self._width = np.array(width)

    def _get_local_influence_coefficient(self, x_coord: float) -> float:
        """Return _get_local_influence_coefficient coefficient in iso-geometric
        coordinates.

        Args
        ----
        x_coord : float
            x-coordinate, centered at element location.
        """
        lambda_0 = self.config.flow.k0 * x_coord
        _, aux_g = _get_aux_fg(lambda_0)
        influence = _get_gamma1(lambda_0, aux_g) * self.width / (self._width[1] * self._width[0])

        lambda_1 = self.config.flow.k0 * (x_coord - self._width[1])
        _, aux_g = _get_aux_fg(lambda_1)
        influence -= _get_gamma1(lambda_1, aux_g) / self._width[1]

        lambda_2 = self.config.flow.k0 * (x_coord + self._width[0])
        _, aux_g = _get_aux_fg(lambda_2)
        influence -= _get_gamma1(lambda_2, aux_g) / self._width[0]
        return influence / self.config.flow.k0

    @property
    def plot_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates for pressure plot."""
        return (
            self.x_coord + np.array([-self._width[0], 0.0, self._width[1]]),
            np.array([0.0, self.pressure, 0.0]),
        )


class AftSemiInfinitePressureBand(PressureElement):
    """Semi-infinite pressure band in aft direction."""

    def _get_local_influence_coefficient(self, x_coord: float) -> float:
        """Return _get_local_influence_coefficient coefficient in
        iso-geometric coordinates.

        Args
        ----
        x_coord : float
            x-coordinate, centered at element location.
        """
        lambda_0 = self.config.flow.k0 * x_coord
        aux_f, _ = _get_aux_fg(lambda_0)
        influence = _get_gamma2(lambda_0, aux_f)
        return influence

    @property
    def plot_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates for pressure plot."""
        return (
            np.array([self.config.plotting.x_fs_min, self.x_coord]),
            np.array([self.pressure, self.pressure]),
        )


class ForwardSemiInfinitePressureBand(PressureElement):
    """Semi-infinite pressure band in forward direction."""

    plot_color = "r"

    def _get_local_influence_coefficient(self, x_coord: float) -> float:
        """Return _get_local_influence_coefficient coefficient in iso-geometric coordinates.

        Args:
            x_coord: x-coordinate, centered at element location.

        """
        lambda_0 = self.config.flow.k0 * x_coord
        aux_f, _ = _get_aux_fg(lambda_0)
        influence = _get_gamma3(lambda_0, aux_f)
        return influence

    @property
    def plot_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates for pressure plot."""
        return (
            np.array([self.x_coord, self.config.plotting.x_fs_max]),
            np.array([self.pressure, self.pressure]),
        )
