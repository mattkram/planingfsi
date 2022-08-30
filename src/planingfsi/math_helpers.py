"""General math helpers."""
from __future__ import annotations

from collections.abc import Callable

import numpy


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
