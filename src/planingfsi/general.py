"""General utilities."""
from pathlib import Path
from typing import Union, Any, Callable

import numpy

from . import trig


def sign(x: float) -> float:
    """Return the sign of the argument. Zero returns zero."""
    if x > 0:
        return 1.0
    elif x < 0:
        return -1.0
    else:
        return 0.0


def heaviside(x: float) -> float:
    """The Heaviside step function returns one if argument is positive, zero if negative, and 0.5 if 0."""
    if x > 0.0:
        return 1.0
    elif x < 0.0:
        return 0.0
    else:
        return 0.5


def integrate(x: numpy.ndarray, f: numpy.ndarray) -> float:
    """Integrate a function using Trapezoidal integration."""
    ind = numpy.argsort(x)
    x = x[ind]
    f = f[ind]
    f[numpy.nonzero(numpy.abs(f) == float("Inf"))] = 0.0

    return 0.5 * numpy.sum((x[1:] - x[:-1]) * (f[1:] + f[:-1]))


def grow_points(x0, x1, x_max, rate=1.1):
    """Grow points exponentially from two starting points assuming a growth rate.

    Args:
        x0: The first point.
        x1: The second point.
        x_max: The maximum distance.
        rate: The growth rate of spacing between subsequent points.

    """
    # TODO: Check this function, is first point included?
    dx = x1 - x0
    x = [x1]

    if dx > 0:
        def done(xt): return xt > x_max
    elif dx < 0:
        def done(xt): return xt < x_max
    else:
        def done(_): return True

    while not done(x[-1]):
        x.append(x[-1] + dx)
        dx *= rate

    return numpy.array(x[1:])


def deriv(f: Callable[[float], float], x: float, direction: str = "c") -> float:
    """Calculate the derivative of a function at a specific point."""
    dx = 1e-6
    fr = f(x + dx)
    fl = f(x - dx)

    if direction[0].lower() == "r" or numpy.isnan(fl):
        return (fr - f(x)) / dx
    elif direction[0].lower() == "l" or numpy.isnan(fr):
        return (f(x) - fl) / dx
    else:
        return (f(x + dx) - f(x - dx)) / (2 * dx)


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def cumdiff(x):
    return numpy.sum(numpy.diff(x))


def rotatePt(oldPt, basePt, ang):
    relPos = numpy.array(oldPt) - numpy.array(basePt)
    newPos = trig.rotate_vec_2d(relPos, ang)
    newPt = basePt + newPos

    return newPt


def writeasdict(filename, *args, **kwargs):
    dataFormat = kwargs.get("dataFormat", ">+10.8e")
    ff = open(filename, "w")
    for name, value in args:
        ff.write("{2:{0}} : {3:{1}}\n".format("<14", dataFormat, name, value))
    ff.close()


def writeaslist(filename: Union[Path, str], *args: Any, **kwargs: Any) -> None:
    headerFormat = kwargs.get("headerFormat", "<15")
    dataFormat = kwargs.get("dataFormat", ">+10.8e")
    with Path(filename).open("w") as ff:
        write(ff, headerFormat, [item for item in [arg[0] for arg in args]])
        for value in zip(*[arg[1] for arg in args]):
            write(ff, dataFormat, value)


def write(ff, writeFormat, items):
    if isinstance(items[0], str):
        ff.write("# ")
    else:
        ff.write("  ")
    ff.write("".join("{1:{0}} ".format(writeFormat, item) for item in items) + "\n")
