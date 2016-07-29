"""Module containing definitions of different pressure element classes."""
#pylint: disable=C0103

import numpy as np
from scipy.special import sici

import planingfsi.config as config
import planingfsi.krampy as kp

if config.plot:
    import matplotlib.pyplot as plt

def _get_aux_fg(lam):
    """Return f and g functions, which are dependent on solving the auxiliary
    sine and cosine functions.
    """
    lam = abs(lam)
    (aux_sine, aux_cosine) = sici(lam)
    (sine, cosine) = np.sin(lam), np.cos(lam)
    return (cosine * (0.5 * np.pi - aux_sine) + sine * aux_cosine,
            sine * (0.5 * np.pi - aux_sine) - cosine * aux_cosine)

def _get_func1(lam, aux_g):
    """Return result of first integral (Lambda_1 in my thesis).

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
        return (aux_g + np.log(-lam)) / np.pi + (2*np.sin(lam) - lam)

def _get_func2(lam, aux_f):
    """Return result of second integral (Lambda_2 in my thesis).

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
    return kp.sign(lam) * aux_f / np.pi + kp.heaviside(-lam) * (2*np.cos(lam) - 1)

def _get_func3(lam, aux_f):
    """Return result of third integral (Lambda_3 in my thesis).

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
    return -kp.sign(lam) * aux_f / np.pi - 2 * kp.heaviside(-lam) * np.cos(lam) - kp.heaviside(lam)

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
    return (f(x+dx) + f(x-dx)) / 2


class PressureElement(object):
    """Abstract base class to represent all different type of pressure elements.
    """
    obj = []

    # TODO: Is this class method necessary? Can the objects be stored in a set
    # instead, or returned directly?
    @classmethod
    def All(cls):
        """Return all objects of class PressureElement."""
        return [o for o in cls.obj]

    def __init__(self, **kwargs):
        PressureElement.obj.append(self)

        self.p = kwargs.get('pressure', np.nan)
        self.tau = kwargs.get('shear', 0.0)
        self.xLoc = kwargs.get('xLoc', np.nan)
        self.zLoc = kwargs.get('zLoc', np.nan)
        self.width = kwargs.get('width', np.nan)
        self.widthR = self.width
        self.widthL = self.width
        self.source = kwargs.get('source', False)
        self.onBody = False
        self.parent = None

    def setParent(self, parent):
        """Set parent PressurePatch instance.

        Args
        ----
        parent : PressurePatch
            PressurePatch object that this element belongs to.
        """
        self.parent = parent

    def isSource(self):
        """Return True of element is a source term.

        Returns
        -------
        bool
            True of element is source.
        """
        return self.source

    def setSource(self, source):
        """Set whether element is a source.

        Args
        ----
        source : bool
            True if element should be a source
        """
        self.source = source

    def setWidth(self, width):
        """Set width of element.
        By default, widthL and widthR are equal to provided width.
        """
        self.width = width
        self.widthR = self.width
        self.widthL = self.width

    def getWidth(self):
        """Return element width."""
        return self.width

    def setXLoc(self, xLoc):
        """Set element x-location."""
        self.xLoc = xLoc

    def getXLoc(self):
        """Get element x-location."""
        return self.xLoc

    def setZLoc(self, zLoc):
        """Set element z-location."""
        self.zLoc = zLoc

    def getZLoc(self):
        """Get element z-location."""
        return self.zLoc

    def setPressure(self, p):
        """Set element pressure."""
        self.p = p

    def setSourcePressure(self, p):
        """Set element pressure and set as source."""
        self.setPressure(p)
        self.setSource(True)
        self.setZLoc(np.nan)

    def setShearStress(self, tau):
        """Set element shear stress."""
        self.tau = tau

    def getPressure(self):
        """Return element pressure."""
        return self.p

    def getShearStress(self):
        """Return element shear stress."""
        return self.tau

    def isOnBody(self):
        """Return True if element is on a body, as opposed to a source term."""
        return self.onBody

    def getInfluenceCoefficient(self, x):
        """Return influence coefficient of element."""
        if not self.getWidth() == 0:
            return self.getK(x - self.xLoc) / (config.rho * config.g)
        else:
            return 0.0

    def getInfluence(self, x):
        """Return influence for actual pressure."""
        return self.getInfluenceCoefficient(x) * self.getPressure()

#    def getPressureFun(self, x):
#        """Return pressure function."""
#        if not self.getWidth() == 0:
#            return self.pressureFun(x - self.xLoc)
#        else:
#            return 0.0

    def influence(self, x):
        """Placeholder, overwritten by each subclass's own method."""
        return 0.0

    def getK(self, x):
        """Evaluate and return influence coefficient in element coordinate."""
        if x == 0.0 or x == self.widthR or x == -self.widthL:
            return _eval_left_right(self.influence, x)
        else:
            return self.influence(x)

#    def pressureFun(self, xx):
#        return 0.0

    def printElement(self):
        """Print element values."""
        # TODO: Use __repr__ and __str__ instead
        string = '{0}: (x,z) = ({1}, {2}), width = {3}, source = {4}, p = {5}, onBody = {6}'
        return string.format(self.__class__.__name__,
                             self.getXLoc(),
                             self.getZLoc(),
                             self.getWidth(),
                             self.isSource(),
                             self.getPressure(),
                             self.isOnBody())


class AftHalfTriangularPressureElement(PressureElement):
    """Pressure element that is triangular in profile but towards aft
    direction."""
    def __init__(self, **kwargs):
        PressureElement.__init__(self, **kwargs)
        self.onBody = kwargs.get('onBody', True)

    def influence(self, x_coord):
        """Return influence coefficient in iso-geometric coordinates.

        Args
        ----
        x_coord : float
            x-coordinate, centered at element location.
        """
        Lambda0 = config.k0 *  x_coord
        aux_f, aux_g = _get_aux_fg(Lambda0)
        K = _get_func1(Lambda0, aux_g) / (self.width * config.k0)
        K += _get_func2(Lambda0, aux_f)

        Lambda2 = config.k0 * (x_coord + self.width)
        __, aux_g = _get_aux_fg(Lambda2)
        K -= _get_func1(Lambda2, aux_g) / (self.width * config.k0)

        return K

    def pressureFun(self, xx):
        """Return shape of pressure profile."""
        if xx < -self.getWidth() or xx > 0.0:
            return 0.0
        else:
            return self.getPressure() * (1 + xx / self.getWidth())

    def plot(self):
        """Plot pressure element shape."""
        x = self.getXLoc() - np.array([self.getWidth(), 0])
        p = np.array([0, self.p])

        col = 'g'
        plt.plot(x, p, col + '-')
        plt.plot([x[1], x[1]], [0.0, p[1]], col + '--')


class ForwardHalfTriangularPressureElement(PressureElement):
    """Pressure element that is triangular in profile but towards forward
    direction."""

    def __init__(self, **kwargs):
        PressureElement.__init__(self, **kwargs)
        self.onBody = kwargs.get('onBody', True)

    def influence(self, xx):
        """Return influence coefficient in iso-geometric coordinates.

        Args
        ----
        x_coord : float
            x-coordinate, centered at element location.
        """
        Lambda0 = config.k0 *  xx
        aux_f, aux_g = _get_aux_fg(Lambda0)
        K = _get_func1(Lambda0, aux_g) / (self.width * config.k0)
        K -= _get_func2(Lambda0, aux_f)

        Lambda1 = config.k0 * (xx - self.width)
        __, aux_g = _get_aux_fg(Lambda1)
        K -= _get_func1(Lambda1, aux_g) / (self.width * config.k0)
        return K

    def pressureFun(self, xx):
        """Return shape of pressure profile."""
        if xx > self.getWidth() or xx < 0.0:
            return 0.0
        else:
            return self.getPressure() * (1 - xx / self.getWidth())

    def plot(self):
        """Plot pressure element shape."""
        x = self.getXLoc() - np.array([self.getWidth(), 0])
        x = np.ones(2) * self.getXLoc() + np.array([0, self.getWidth()])
        p = np.array([self.p, 0])

        col = 'b'
        plt.plot(x, p, col + '-')
        plt.plot([x[0], x[0]], [0.0, p[0]], col + '--')


class NegativeForwardHalfTriangularPressureElement(PressureElement):
    """Pressure element that is triangular in profile but towards aft
    direction."""

    def __init__(self, **kwargs):
        PressureElement.__init__(self, **kwargs)
        self.onBody = kwargs.get('onBody', True)

    def influence(self, xx):
        """Return influence coefficient in iso-geometric coordinates.

        Args
        ----
        x_coord : float
            x-coordinate, centered at element location.
        """
        Lambda0 = config.k0 *  xx
        aux_f, aux_g = _get_aux_fg(Lambda0)
        K = _get_func1(Lambda0, aux_g) / (self.width * config.k0)
        K -= _get_func2(Lambda0, aux_f)

        Lambda1 = config.k0 * (xx - self.width)
        __, aux_g = _get_aux_fg(Lambda1)
        K -= _get_func1(Lambda1, aux_g) / (self.width * config.k0)
        return -K

    def plot(self):
        """Plot pressure element shape."""
        x = np.ones(2) * self.getXLoc() + np.array([0, self.getWidth()])
        p = np.array([self.p, 0])

        col = 'b'
        plt.plot(x, p, col + '-')
        plt.plot([x[0], x[0]], [0.0, p[0]], col + '--')


class TransomPressureElement(PressureElement):
    """Pressure element that is triangular in profile but towards aft
    direction."""

    def getInfluenceCoefficient(self, x):
        if kp.inRange(x, self.parent.getEndPts()):
            return -1.0 / (config.rho * config.g)
        else:
            return 0.0

    def plot(self):
        """Plot pressure element shape."""
        return None


class CompleteTriangularPressureElement(PressureElement):
    """Pressure element that is triangular in profile but towards aft
    direction."""

    def __init__(self, **kwargs):
        PressureElement.__init__(self, **kwargs)
        self.onBody = kwargs.get('onBody', True)

    def setWidth(self, width):
        PressureElement.setWidth(self, np.sum(width))
        self.widthL = width[0]
        self.widthR = width[1]

    def influence(self, xx):
        """Return influence coefficient in iso-geometric coordinates.

        Args
        ----
        x_coord : float
            x-coordinate, centered at element location.
        """
        Lambda0 = config.k0 *  xx
        __, aux_g = _get_aux_fg(Lambda0)
        K = _get_func1(Lambda0, aux_g) * self.width / (self.widthR * self.widthL)

        Lambda1 = config.k0 * (xx - self.widthR)
        __, aux_g = _get_aux_fg(Lambda1)
        K -= _get_func1(Lambda1, aux_g) / self.widthR

        Lambda2 = config.k0 * (xx + self.widthL)
        __, aux_g = _get_aux_fg(Lambda2)
        K -= _get_func1(Lambda2, aux_g) / self.widthL
        return K / config.k0

    def pressureFun(self, xx):
        """Return shape of pressure profile."""
        if xx > self.widthR or xx < -self.widthL:
            return 0.0
        elif xx < 0.0:
            return self.getPressure() * (1 + xx / self.widthL)
        else:
            return self.getPressure() * (1 - xx / self.widthR)

    def plot(self):
        """Plot pressure element shape."""
        x = np.ones(3) * self.getXLoc() + np.array([-self.widthL, 0.0, self.widthR])
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
        K = _get_func2(Lambda0, aux_f)
        return K

    def pressureFun(self, xx):
        """Return shape of pressure profile."""
        if xx > 0.0:
            return 0.0
        else:
            return self.getPressure()

    def plot(self):
        """Plot pressure element shape."""
        infinity = 50.0

        x = np.ones(2) * self.getXLoc()
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
        K = _get_func3(Lambda0, aux_f)
        return K

    def pressureFun(self, xx):
        """Return shape of pressure profile."""
        if xx < 0.0:
            return 0.0
        else:
            return self.getPressure()

    def plot(self):
        """Plot pressure element shape."""
        infinity = 50.0
        x = np.ones(2) * self.getXLoc()
        x += np.array([0, infinity])

        p = np.zeros(len(x))
        p += np.array([self.p, self.p])

        col = 'r'
        plt.plot(x, p, col + '-')
        plt.plot([x[0], x[0]], [p[0], 0.0], col + '--')


class CompoundPressureElement(PressureElement):
    """Pressure element that is triangular in profile but towards aft
    direction."""

    def __init__(self, **kwargs):
        PressureElement.__init__(self, **kwargs)
        self.element = [self]

    def setWidth(self, width):
        PressureElement.setWidth(self, np.sum(width))
        for i, el in enumerate(self.element):
            el.setWidth(width[i])

    def setXLoc(self, xLoc):
        PressureElement.setXLoc(self, xLoc)
        for el in self.element:
            el.setXLoc(xLoc)

    def setPressure(self, p):
        PressureElement.setPressure(self, p)
        for el in self.element:
            el.setPressure(p)

    def getInfluence(self, x):
        """Return influence coefficient in iso-geometric coordinates.

        Args
        ----
        x_coord : float
            x-coordinate, centered at element location.
        """
        return np.sum([el.getInfluenceCoefficient(x) * el.getPressure() for el in self.element])

    def getInfluenceCoefficient(self, x):
        return self.getInfluence(x) / self.getPressure()

    def plot(self):
        """Plot pressure element shape."""
        for el in self.element:
            el.plot()


class CompleteTriangularPressureElementNew(CompoundPressureElement):
    """Pressure element that is triangular in profile but towards aft
    direction."""

    def __init__(self, **kwargs):
        CompoundPressureElement.__init__(self, **kwargs)
        self.onBody = kwargs.get('onBody', True)
        self.element = [CompleteTriangularPressureElement()]


class AftFinitePressureBand(CompoundPressureElement):
    """Pressure element that is triangular in profile but towards aft
    direction."""

    def __init__(self, **kwargs):
        CompoundPressureElement.__init__(self, **kwargs)
        self.element = [AftSemiInfinitePressureBand() for _ in range(2)]
        self.setWidth(self.width)
        self.setXLoc(self.xLoc)
        self.setPressure(self.p)

    def setWidth(self, width):
        PressureElement.setWidth(self, width)
        self.element[1].setXLoc(self.getXLoc() - self.getWidth())

    def setXLoc(self, xLoc):
        PressureElement.setXLoc(self, xLoc)
        self.element[0].setXLoc(xLoc)
        self.element[1].setXLoc(xLoc - self.getWidth())

    def setPressure(self, p):
        PressureElement.setPressure(self, p)
        self.element[0].setPressure(p)
        self.element[1].setPressure(-p)

    def pressureFun(self, xx):
        """Return shape of pressure profile."""
        if xx > 0.0 or xx < -self.getWidth():
            return 0.0
        else:
            return self.getPressure()

    def plot(self):
        """Plot pressure element shape."""
        x = np.array([self.element[0].getXLoc(), self.element[1].getXLoc()])
        p = np.array([0.0, self.p])

        col = 'm'
        plt.plot(x, np.ones(2)*p[1], col + '-')
        plt.plot(np.ones(2)*x[0], p, col + '--')
        plt.plot(np.ones(2)*x[1], p, col + '--')


class AftFinitePressureBandNew(CompoundPressureElement):
    """Pressure element that is triangular in profile but towards aft
    direction."""

    def __init__(self, **kwargs):
        CompoundPressureElement.__init__(self, **kwargs)
        self.element = [AftHalfTriangularPressureElement(), \
                        ForwardHalfTriangularPressureElement()]
        self.setWidth(self.width)
        self.setXLoc(self.xLoc)
        self.setPressure(self.p)

    def setWidth(self, width):
        PressureElement.setWidth(self, width)
        self.element[1].setXLoc(self.getXLoc() - self.getWidth())
        for el in self.element:
            el.setWidth(width)

    def setXLoc(self, xLoc):
        PressureElement.setXLoc(self, xLoc)
        self.element[0].setXLoc(xLoc)
        self.element[1].setXLoc(xLoc - self.getWidth())

    def setPressure(self, p):
        PressureElement.setPressure(self, p)
        self.element[0].setPressure(p)
        self.element[1].setPressure(p)

    def plot(self):
        """Plot pressure element shape."""
        x = np.array([self.element[0].getXLoc(), self.element[1].getXLoc()])
        p = np.array([0.0, self.p])

        col = 'm'
        plt.plot(x, np.ones(2)*p[1], col + '-')
        plt.plot(np.ones(2)*x[0], p, col + '--')
        plt.plot(np.ones(2)*x[1], p, col + '--')


class AftFinitePressureBandWithFwdHalfTriangle(CompoundPressureElement):
    """Pressure element that is triangular in profile but towards aft
    direction."""

    def __init__(self, **kwargs):
        CompoundPressureElement.__init__(self, **kwargs)
        self.element = [AftFinitePressureBand(), ForwardHalfTriangularPressureElement()]


class AftSemiInfinitePressureBandWithFwdHalfTriangle(CompoundPressureElement):
    """Pressure element that is triangular in profile but towards aft
    direction."""

    def __init__(self, **kwargs):
        CompoundPressureElement.__init__(self, **kwargs)
        self.element = [AftSemiInfinitePressureBand(), ForwardHalfTriangularPressureElement()]

    def setWidth(self, width):
        PressureElement.setWidth(self, width)
        self.element[1].setWidth(width)

    def setXLoc(self, xLoc):
        PressureElement.setXLoc(self, xLoc)
        self.element[0].setXLoc(xLoc)
        self.element[1].setXLoc(xLoc)
