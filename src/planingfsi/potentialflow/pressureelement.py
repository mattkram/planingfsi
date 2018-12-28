"""Module containing definitions of different types of pressure element."""

import numpy as np
from scipy.special import sici

# TODO: Move all plotting commands to its own module
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    pass

import planingfsi.config as config

# import planingfsi.krampy as kp


_PLOT_INFINITY = 50.0


def _get_aux_fg(lam):
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


def _get_gamma1(lam, aux_g):
    """Return result of first integral (Gamma_1 in my thesis).

    Args
    ----
    lam : float
        Non-dimensional x-position.
    aux_g : float
        Second result of `_get_aux_fg`

    Returns
    -------
    float

    """
    if lam > 0:
        return (aux_g + np.log(lam)) / np.pi
    else:
        return (aux_g + np.log(-lam)) / np.pi + (2 * np.sin(lam) - lam)


def _get_gamma2(lam, aux_f):
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
    return kp.sign(lam) * aux_f / np.pi + kp.heaviside(-lam) * (2 * np.cos(lam) - 1)


def _get_gamma3(lam, aux_f):
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
        -kp.sign(lam) * aux_f / np.pi
        - 2 * kp.heaviside(-lam) * np.cos(lam)
        - kp.heaviside(lam)
    )


def _eval_left_right(f, x, dx=1e-6):
    """Evaluate a function by taking the average of the values just above and
    below the x value. Used to avoid singularity/Inf/NaN.

    Args
    ----
    f : function
        Function handle.
    x : float
        Argument of function.
    dx : float, optional
        Step size.

    Returns
    ------
    float : Function value.

    """
    return (f(x + dx) + f(x - dx)) / 2


class PressureElement(object):
    """Abstract base class to represent all different types of pressure elements.

    Attributes
    ----------
    x_coord : float
        x-location of element.

    z_coord : float
        z-coordinate of element, if on body.

    pressure : float
        The pressure/strength of the element.

    shear_stress : float
        Shear stress at the element.

    width : float
        Width of element.

    is_source : bool
        True of element is a source.

    is_on_body : bool
        True if on body. Used for force calculation.

    parent : PressurePatch
        Pressure patch that this element belongs to.

    """

    def __init__(
        self,
        x_coord=np.nan,
        z_coord=np.nan,
        pressure=np.nan,
        shear_stress=0.0,
        width=np.nan,
        is_source=False,
        is_on_body=False,
        parent=None,
    ):

        # TODO: Replace x_coord with coord pointer to Coordinate object,
        # (e.g. self.coords.x).
        self.x_coord = x_coord
        self.z_coord = z_coord
        self._pressure = pressure
        self.shear_stress = shear_stress
        self._width = np.zeros(2)
        self.width = width
        self.is_source = is_source
        self.is_on_body = is_on_body
        self.parent = parent

    @property
    def width(self):
        return sum(self._width)

    @property
    def pressure(self):
        return self._pressure

    @pressure.setter
    def pressure(self, value):
        self._pressure = value

    def get_influence_coefficient(self, x_coord):
        """Return _get_local_influence_coefficient coefficient of element.

        Args
        ----
        x_coord : float
            Dimensional x-coordinate.
        """
        x_rel = x_coord - self.x_coord
        if self.width == 0.0:
            return 0.0
        elif x_rel == 0.0 or x_rel == self._width[1] or x_rel == -self._width[0]:
            influence = _eval_left_right(self._get_local_influence_coefficient, x_rel)
        else:
            influence = self._get_local_influence_coefficient(x_rel)
        return influence / (config.flow.density * config.flow.gravity)

    def get_influence(self, x_coord):
        """Return _get_local_influence_coefficient for actual pressure.

        Args
        ----
        x_coord : float
            Dimensional x-coordinate.
        """
        return self.get_influence_coefficient(x_coord) * self.pressure

    def _get_local_influence_coefficient(self, x_coord):
        """Return influence coefficient in iso-geometric coordinates. Each
        subclass must implement its own method.

        Args
        ----
        x_coord : float
            x-coordinate, centered at element location.
        """
        pass

    def __repr__(self):
        """Print element attributes."""
        string = (
            "{0}: (x,z) = ({1}, {2}), width = {3},"
            "is_source = {4}, p = {5}, is_on_body = {6}"
        )
        return string.format(
            self.__class__.__name__,
            self.x_coord,
            self.z_coord,
            self.width,
            self.is_source,
            self.pressure,
            self.is_on_body,
        )

    def plot(self, x_coords, color="b"):
        """Plot pressure element shape."""
        _pressure = np.array([self.pressure(xi) for xi in x_coords])

        plt.plot(x_coords, _pressure, color=color, linetype="-")
        plt.plot(
            self.x_coord * np.ones(2), [0.0, self.pressure], color=color, linetype="--"
        )


class AftHalfTriangularPressureElement(PressureElement):
    """Pressure element that is triangular in profile but towards aft
    direction.

    Args
    ----
    **kwargs
        Same as PressureElement arguments, with the following defaults
        overridden:
            is_on_body : True
    """

    def __init__(self, **kwargs):
        kwargs["is_on_body"] = kwargs.get("is_on_body", True)
        super().__init__(**kwargs)

    @PressureElement.width.setter
    def width(self, width):
        self._width[0] = width

    @PressureElement.pressure.getter
    def pressure(self, x_coord=None):
        """Return shape of pressure profile."""
        if x_coord is None:
            return self._pressure

        _x_rel = x_coord - self.x_coord
        if _x_rel < -self.width or _x_rel > 0.0:
            return 0.0
        else:
            return self._pressure * (1 + _x_rel / self.width)

    def _get_local_influence_coefficient(self, x_rel):
        lambda_0 = config.flow.k0 * x_rel
        aux_f, aux_g = _get_aux_fg(lambda_0)
        _influence = _get_gamma1(lambda_0, aux_g) / (self.width * config.flow.k0)
        _influence += _get_gamma2(lambda_0, aux_f)

        lambda_2 = config.flow.k0 * (x_rel + self.width)
        __, aux_g = _get_aux_fg(lambda_2)
        _influence -= _get_gamma1(lambda_2, aux_g) / (self.width * config.flow.k0)

        return _influence

    def plot(self):
        """Plot pressure element shape."""
        x_coords = self.x_coord - np.array([self.width, 0])
        super().plot(x_coords, color="g")


class ForwardHalfTriangularPressureElement(PressureElement):
    """Pressure element that is triangular in profile but towards forward
    direction.

    Args
    ----
    **kwargs
        Same as PressureElement arguments, with the following defaults
        overridden:
            is_on_body : True
    """

    def __init__(self, **kwargs):
        kwargs["is_on_body"] = kwargs.get("is_on_body", True)
        super().__init__(**kwargs)

    @PressureElement.width.setter
    def width(self, width):
        self._width[1] = width

    @PressureElement.pressure.getter
    def pressure(self, x_coord=None):
        """Return shape of pressure profile."""
        if x_coord is None:
            return self._pressure

        _x_rel = x_coord - self.x_coord
        if _x_rel > self.width or _x_rel < 0.0:
            return 0.0
        else:
            return self._pressure * (1 - _x_rel / self.width)

    def _get_local_influence_coefficient(self, x_coord):
        """Return _get_local_influence_coefficient coefficient in iso-geometric
        coordinates.

        Args
        ----
        x_coord : float
            x-coordinate, centered at element location.
        """
        lambda_0 = config.flow.k0 * x_coord
        aux_f, aux_g = _get_aux_fg(lambda_0)
        influence = _get_gamma1(lambda_0, aux_g) / (self.width * config.flow.k0)
        influence -= _get_gamma2(lambda_0, aux_f)

        lambda_1 = config.flow.k0 * (x_coord - self.width)
        _, aux_g = _get_aux_fg(lambda_1)
        influence -= _get_gamma1(lambda_1, aux_g) / (self.width * config.flow.k0)
        return influence

    def plot(self):
        """Plot pressure element shape."""
        x_coords = self.x_coord + np.array([0, self.width])
        super().plot(x_coords, color="b")


class CompleteTriangularPressureElement(PressureElement):
    """Pressure element that is triangular in profile towards both
    directions.

    Args
    ----
    **kwargs
        Same as PressureElement arguments, with the following defaults
        overridden:
            is_on_body : True
    """

    def __init__(self, **kwargs):
        kwargs["is_on_body"] = kwargs.get("is_on_body", True)
        kwargs["width"] = kwargs.get("width", np.zeros(2))
        super().__init__(**kwargs)

    @PressureElement.width.setter
    def width(self, width):
        if not len(width) == 2:
            raise ValueError("Width must be length-two array")
        else:
            self._width = np.array(width)

    @PressureElement.pressure.getter
    def pressure(self, x_coord=None):
        """Return shape of pressure profile."""
        if x_coord is None:
            return self._pressure

        _x_rel = x_coord - self.x_coord
        if _x_rel > self._width[1] or _x_rel < -self._width[0]:
            return 0.0
        elif _x_rel < 0.0:
            return self._pressure * (1 + _x_rel / self._width[0])
        else:
            return self._pressure * (1 - _x_rel / self._width[1])

    def _get_local_influence_coefficient(self, x_coord):
        """Return _get_local_influence_coefficient coefficient in iso-geometric
        coordinates.

        Args
        ----
        x_coord : float
            x-coordinate, centered at element location.
        """
        lambda_0 = config.flow.k0 * x_coord
        _, aux_g = _get_aux_fg(lambda_0)
        influence = (
            _get_gamma1(lambda_0, aux_g)
            * self.width
            / (self._width[1] * self._width[0])
        )

        lambda_1 = config.flow.k0 * (x_coord - self._width[1])
        _, aux_g = _get_aux_fg(lambda_1)
        influence -= _get_gamma1(lambda_1, aux_g) / self._width[1]

        lambda_2 = config.flow.k0 * (x_coord + self._width[0])
        _, aux_g = _get_aux_fg(lambda_2)
        influence -= _get_gamma1(lambda_2, aux_g) / self._width[0]
        return influence / config.flow.k0

    def plot(self):
        """Plot pressure element shape."""
        x_coords = self.x_coord + np.array([-self._width[0], 0.0, self._width[1]])
        super().plot(x_coords, color="r")


class AftSemiInfinitePressureBand(PressureElement):
    """Semi-infinite pressure band in aft direction.

    Args
    ----
    **kwargs
        Same as PressureElement arguments
    """

    @PressureElement.pressure.getter
    def pressure(self, x_coord=0.0):
        """Return shape of pressure profile."""
        x_rel = x_coord - self.x_coord
        if x_rel > 0.0:
            return 0.0
        else:
            return self._pressure

    def _get_local_influence_coefficient(self, x_coord):
        """Return _get_local_influence_coefficient coefficient in
        iso-geometric coordinates.

        Args
        ----
        x_coord : float
            x-coordinate, centered at element location.
        """
        lambda_0 = config.flow.k0 * x_coord
        aux_f, _ = _get_aux_fg(lambda_0)
        influence = _get_gamma2(lambda_0, aux_f)
        return influence

    def plot(self):
        """Plot pressure element shape."""
        x_coords = self.x_coord + np.array([-_PLOT_INFINITY, 0])
        super().plot(x_coords, color="r")


class ForwardSemiInfinitePressureBand(PressureElement):
    """Semi-infinite pressure band in forward direction.

    Args
    ----
    **kwargs
        Same as PressureElement arguments
    """

    @PressureElement.pressure.getter
    def pressure(self, x_coord=None):
        """Return shape of pressure profile."""
        if x_coord is None:
            return self._pressure

        _x_rel = x_coord - self.x_coord
        if _x_rel < 0.0:
            return 0.0
        else:
            return self._pressure

    def _get_local_influence_coefficient(self, x_coord):
        """Return _get_local_influence_coefficient coefficient in iso-geometric
        coordinates.

        Args
        ----
        x_coord : float
            x-coordinate, centered at element location.
        """
        lambda_0 = config.flow.k0 * x_coord
        aux_f, _ = _get_aux_fg(lambda_0)
        influence = _get_gamma3(lambda_0, aux_f)
        return influence

    def plot(self):
        """Plot pressure element shape."""
        x_coords = self.x_coord + np.array([0, _PLOT_INFINITY])
        super().plot(x_coords, color="r")
