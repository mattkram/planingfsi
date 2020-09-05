"""General utilities."""
from pathlib import Path
from typing import Union, Any, Callable, TextIO

import numpy

from planingfsi import trig


def sign(x: float) -> float:
    """Return the sign of the argument. Zero returns zero."""
    if x > 0:
        return 1.0
    elif x < 0:
        return -1.0
    else:
        return 0.0


def heaviside(x: float) -> float:
    """The Heaviside step function.

    Returns one if argument is positive, zero if negative, and 0.5 if 0.

    """
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


def grow_points(x0: float, x1: float, x_max: float, rate: float = 1.1) -> numpy.ndarray:
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

        def done(xt: float) -> bool:
            return xt > x_max

    elif dx < 0:

        def done(xt: float) -> bool:
            return xt < x_max

    else:

        def done(xt: float) -> bool:
            _ = xt
            return True

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


def cross2(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Calculate the cross product of two two-dimensional vectors."""
    return a[0] * b[1] - a[1] * b[0]


def cumdiff(x: numpy.ndarray) -> float:
    """Calculate the cumulative difference of an array."""
    return float(numpy.sum(numpy.diff(x)))


def rotate_point(point: numpy.ndarray, about: numpy.ndarray, angle: float) -> numpy.ndarray:
    """Rotate a point about another point by a specific angle in degrees."""
    relative_pos = numpy.array(point) - numpy.array(about)
    new_pos = trig.rotate_vec_2d(relative_pos, angle)
    return about + new_pos


def write_as_dict(filename: Union[Path, str], *args: Any, data_format: str = ">+10.8e") -> None:
    """Write arguments to a file as a dictionary."""
    with Path(filename).open("w") as ff:
        for name, value in args:
            ff.write(f"{name:<14} : {value:{data_format}}\n")


def write_as_list(
    filename: Union[Path, str], *args: Any, header_format: str = "<15", data_format: str = ">+10.8e"
) -> None:
    """Write the arguments to a file as a list."""
    with Path(filename).open("w") as ff:
        _write(ff, header_format, [item for item in [arg[0] for arg in args]])
        for value in zip(*[arg[1] for arg in args]):
            _write(ff, data_format, value)


def _write(ff: TextIO, write_format: str, items: Any) -> None:
    """Write items to a file with a specific format."""
    if isinstance(items[0], str):
        ff.write("# ")
    else:
        ff.write("  ")
    ff.write("".join("{1:{0}} ".format(write_format, item) for item in items) + "\n")
