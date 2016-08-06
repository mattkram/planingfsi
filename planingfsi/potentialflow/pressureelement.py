"""Module containing definitions of different types of pressure element."""

import numpy as np
from scipy.special import sici

import planingfsi.config as config
import planingfsi.krampy as kp

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


def _get_aux_fg(lam):
    """Return f and g functions, which are dependent on the auxiliary
    sine and cosine functions. Combined in one function to reduce number of
    calls to scipy.special.sici().
    """
    lam = abs(lam)
    (aux_sine, aux_cosine) = sici(lam)
    (sine, cosine) = np.sin(lam), np.cos(lam)
    return (cosine * (0.5 * np.pi - aux_sine) + sine * aux_cosine,
            sine * (0.5 * np.pi - aux_sine) - cosine * aux_cosine)


def _get_gamma1(lam, aux_g):
    """Return result of first integral (Gamma_1 in my thesis).

    Args
    ----
    lam : float
        Lambda.
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
        Lambda.
    aux_f : float
        First result of `_get_aux_fg`

    Returns
    -------
    float
    """
    return (kp.sign(lam) * aux_f / np.pi +
            kp.heaviside(-lam) * (2 * np.cos(lam) - 1))


def _get_gamma3(lam, aux_f):
    """Return result of third integral (Gamma_3 in my thesis).

    Args
    ----
    lam : float
        Lambda.
    aux_f : float
        First result of `_get_aux_fg`

    Returns
    -------
    float
    """
    return (-kp.sign(lam) * aux_f / np.pi -
            2 * kp.heaviside(-lam) * np.cos(lam) -
            kp.heaviside(lam))


def _eval_left_right(f, x, dx=1e-6):
    """Evaluate a function by taking the average of the values just above and
    below the x value. Used to avoid singularity/Inf.

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
    float
    """
    return (f(x + dx) + f(x - dx)) / 2


class PressureElement(object):
    """Abstract base class to represent all different types of pressure elements.

    Attributes
    ----------
    x_coord : float
        x-location of element.

    z_coord : float
        z-coordinate of element.

    pressure : float
        The pressure/strength of the element.

    shear_stress : float
        Shear stress at the element.

    width : float
        Width of element.

    is_source : bool
        True of element is a source.

    is_on_body : bool
        True of on body.

    parent : PressurePatch
        Pressure patch that this element belongs to.
    """
    all = []

    def __init__(self, xLoc=np.nan, zLoc=np.nan, pressure=np.nan,
                 shear_stress=0.0, width=np.nan, is_source=False,
                 is_on_body=False, parent=None):
        PressureElement.all.append(self)

        self.x_coord = xLoc
        self.z_coord = zLoc
        self.pressure = pressure
        self.shear_stress = shear_stress
        self._width_total = width
        self._width_right = width
        self._width_left = width
        self.is_source = is_source
        self.is_on_body = is_on_body
        self.parent = parent

    @property
    def width(self):
        return self._width_total

    @width.setter
    def width(self, width):
        self._width_total = width
        self._width_right = width
        self._width_left = width

    def get_influence_coefficient(self, x):
        """Return influence coefficient of element."""
        if not self.width == 0:
            return self.getK(x - self.x_coord) / (config.rho * config.g)
        else:
            return 0.0

    def get_influence(self, x):
        """Return influence for actual pressure."""
        return self.get_influence_coefficient(x) * self.pressure

#    def getPressureFun(self, x):
#        """Return pressure function."""
#        if not self.get_width() == 0:
#            return self.pressureFun(x - self.x_coord)
#        else:
#            return 0.0

    def influence(self, x):
        """Placeholder, overwritten by each subclass's own method."""
        pass

    def getK(self, x):
        """Evaluate and return influence coefficient in isometric coordinate."""
        if x == 0.0 or x == self._width_right or x == -self._width_left:
            return _eval_left_right(self.influence, x)
        else:
            return self.influence(x)

    def __repr__(self):
        """Print element attributes."""
        string = ('{0}: (x,z) = ({1}, {2}), width = {3},',
                  'is_source = {4}, p = {5}, is_on_body = {6}')
        return string.format(self.__class__.__name__,
                             self.x_coord,
                             self.z_coord,
                             self.width,
                             self.is_source,
                             self.pressure,
                             self.is_on_body)


class AftHalfTriangularPressureElement(PressureElement):
    """Pressure element that is triangular in profile but towards aft
    direction."""

    def __init__(self, **kwargs):
        PressureElement.__init__(self, **kwargs)
        self.is_on_body = kwargs.get('is_on_body', True)

    def influence(self, x_coord):
        """Return influence coefficient in iso-geometric coordinates.

        Args
        ----
        x_coord : float
            x-coordinate, centered at element location.
        """
        Lambda0 = config.k0 * x_coord
        aux_f, aux_g = _get_aux_fg(Lambda0)
        K = _get_gamma1(Lambda0, aux_g) / (self.width * config.k0)
        K += _get_gamma2(Lambda0, aux_f)

        Lambda2 = config.k0 * (x_coord + self.width)
        __, aux_g = _get_aux_fg(Lambda2)
        K -= _get_gamma1(Lambda2, aux_g) / (self.width * config.k0)

        return K

    def pressureFun(self, xx):
        """Return shape of pressure profile."""
        if xx < -self.get_width() or xx > 0.0:
            return 0.0
        else:
            return self.pressure * (1 + xx / self.get_width())

    def plot(self):
        """Plot pressure element shape."""
        x = self.get_xloc() - np.array([self.get_width(), 0])
        p = np.array([0, self.p])

        col = 'g'
        plt.plot(x, p, col + '-')
        plt.plot([x[1], x[1]], [0.0, p[1]], col + '--')


class ForwardHalfTriangularPressureElement(PressureElement):
    """Pressure element that is triangular in profile but towards forward
    direction."""

    def __init__(self, **kwargs):
        PressureElement.__init__(self, **kwargs)
        self.is_on_body = kwargs.get('is_on_body', True)

    def influence(self, xx):
        """Return influence coefficient in iso-geometric coordinates.

        Args
        ----
        x_coord : float
            x-coordinate, centered at element location.
        """
        Lambda0 = config.k0 * xx
        aux_f, aux_g = _get_aux_fg(Lambda0)
        K = _get_gamma1(Lambda0, aux_g) / (self.width * config.k0)
        K -= _get_gamma2(Lambda0, aux_f)

        Lambda1 = config.k0 * (xx - self.width)
        __, aux_g = _get_aux_fg(Lambda1)
        K -= _get_gamma1(Lambda1, aux_g) / (self.width * config.k0)
        return K

    def pressureFun(self, xx):
        """Return shape of pressure profile."""
        if xx > self.get_width() or xx < 0.0:
            return 0.0
        else:
            return self.pressure * (1 - xx / self.get_width())

    def plot(self):
        """Plot pressure element shape."""
        x = self.get_xloc() - np.array([self.get_width(), 0])
        x = np.ones(2) * self.get_xloc() + np.array([0, self.get_width()])
        p = np.array([self.p, 0])

        col = 'b'
        plt.plot(x, p, col + '-')
        plt.plot([x[0], x[0]], [0.0, p[0]], col + '--')


class CompleteTriangularPressureElement(PressureElement):
    """Pressure element that is triangular in profile towards both
    directions. See `PressureElement` for allowable kwargs."""

    def __init__(self, **kwargs):
        kwargs['is_on_body'] = True
        super(self.__class__, self).__init__(**kwargs)

    @PressureElement.width.setter
    def width(self, width):
        """Set width.

        Args
        ----
        width : list(float)
            length-two list or tuple defining left and right widths of elements.
        """
        self._width_total = np.sum(width)
        self._width_left = width[0]
        self._width_right = width[1]

    def influence(self, xx):
        """Return influence coefficient in iso-geometric coordinates.

        Args
        ----
        x_coord : float
            x-coordinate, centered at element location.
        """
        Lambda0 = config.k0 * xx
        __, aux_g = _get_aux_fg(Lambda0)
        K = _get_gamma1(Lambda0, aux_g) * self.width / \
            (self._width_right * self._width_left)

        Lambda1 = config.k0 * (xx - self._width_right)
        __, aux_g = _get_aux_fg(Lambda1)
        K -= _get_gamma1(Lambda1, aux_g) / self._width_right

        Lambda2 = config.k0 * (xx + self._width_left)
        __, aux_g = _get_aux_fg(Lambda2)
        K -= _get_gamma1(Lambda2, aux_g) / self._width_left
        return K / config.k0

    def pressureFun(self, xx):
        """Return shape of pressure profile."""
        if xx > self._width_right or xx < -self._width_left:
            return 0.0
        elif xx < 0.0:
            return self.pressure * (1 + xx / self._width_left)
        else:
            return self.pressure * (1 - xx / self._width_right)

    def plot(self):
        """Plot pressure element shape."""
        x = np.ones(3) * self.get_xloc() + \
            np.array([-self._width_left, 0.0, self._width_right])
        p = np.array([0, self.p, 0])

        col = 'r'
        plt.plot(x, p, col + '-')
        plt.plot([x[1], x[1]], [0.0, p[1]], col + '--')


class AftSemiInfinitePressureBand(PressureElement):
    """Pressure element that is triangular in profile but towards aft
    direction."""

    def __init__(self, **kwargs):
        PressureElement.__init__(self, **kwargs)

    def influence(self, xx):
        """Return influence coefficient in iso-geometric coordinates.

        Args
        ----
        x_coord : float
            x-coordinate, centered at element location.
        """
        Lambda0 = config.k0 * xx
        aux_f, __ = _get_aux_fg(Lambda0)
        K = _get_gamma2(Lambda0, aux_f)
        return K

    def pressureFun(self, xx):
        """Return shape of pressure profile."""
        if xx > 0.0:
            return 0.0
        else:
            return self.pressure

    def plot(self):
        """Plot pressure element shape."""
        infinity = 50.0

        x = np.ones(2) * self.get_xloc()
        x += np.array([-infinity, 0])

        p = np.zeros(len(x))
        p += np.array([self.p, self.p])

        col = 'r'
        plt.plot(x, p, col + '-')
        plt.plot([x[1], x[1]], [p[0], 0.0], col + '--')


class ForwardSemiInfinitePressureBand(PressureElement):
    """Pressure element that is triangular in profile but towards aft
    direction."""

    def __init__(self, **kwargs):
        PressureElement.__init__(self, **kwargs)

    def influence(self, xx):
        """Return influence coefficient in iso-geometric coordinates.

        Args
        ----
        x_coord : float
            x-coordinate, centered at element location.
        """
        Lambda0 = config.k0 * xx
        aux_f, __ = _get_aux_fg(Lambda0)
        K = _get_gamma3(Lambda0, aux_f)
        return K

    def pressureFun(self, xx):
        """Return shape of pressure profile."""
        if xx < 0.0:
            return 0.0
        else:
            return self.pressure

    def plot(self):
        """Plot pressure element shape."""
        infinity = 50.0
        x = np.ones(2) * self.get_xloc()
        x += np.array([0, infinity])

        p = np.zeros(len(x))
        p += np.array([self.p, self.p])

        col = 'r'
        plt.plot(x, p, col + '-')
        plt.plot([x[0], x[0]], [p[0], 0.0], col + '--')
