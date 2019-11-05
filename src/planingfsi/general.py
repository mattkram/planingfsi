"""General utilities."""
import os
from fnmatch import fnmatch

import numpy

from . import trig


def find_files(path=".", pattern="*"):
    """Return list of files in path matching the pattern.

    Args
    ----
    path : str, optional
        Path to search. Default is current directory.

    pattern : str
        Wildcard pattern to match.
    """
    return [d for d in os.listdir(path) if fnmatch(d, pattern)]


def sign(x):
    if x > 0:
        return 1.0
    elif x < 0:
        return -1.0
    else:
        return 0.0


def heaviside(x):
    if x > 0:
        return 1.0
    elif x < 0:
        return 0.0
    else:
        return 0.5


def trapz(x, f):
    # Trapezoidal integration
    I = numpy.argsort(x)
    x = x[I]
    f = f[I]
    f[numpy.nonzero(numpy.abs(f) == float("Inf"))] = 0.0

    return 0.5 * numpy.sum((x[1:] - x[0:-1]) * (f[1:] + f[0:-1]))


def integrate(x, f):
    return trapz(x, f)


def growPoints(x0, x1, xMax, rate=1.1):
    dx = x1 - x0
    x = [x1]

    if dx > 0:
        done = lambda xt: xt > xMax
    elif dx < 0:
        done = lambda xt: xt < xMax
    else:
        done = lambda xt: True

    while not done(x[-1]):
        x.append(x[-1] + dx)
        dx *= rate

    return numpy.array(x[1:])


def getDerivative(f, x, direction="c"):
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


def writeaslist(filename, *args, **kwargs):
    headerFormat = kwargs.get("headerFormat", "<15")
    dataFormat = kwargs.get("dataFormat", ">+10.8e")
    ff = open(filename, "w")
    write(ff, headerFormat, [item for item in [arg[0] for arg in args]])
    for value in zip(*[arg[1] for arg in args]):
        write(ff, dataFormat, value)
    ff.close()


def write(ff, writeFormat, items):
    if isinstance(items[0], str):
        ff.write("# ")
    else:
        ff.write("  ")
    ff.write("".join("{1:{0}} ".format(writeFormat, item) for item in items) + "\n")


def sortDirByNum(dirStr, direction="forward"):
    num = numpy.array(
        [
            float("".join(i for i in dir.lower() if i.isdigit() or i == ".").strip("."))
            for dir in dirStr
        ]
    )
    if direction == "reverse":
        ind = numpy.argsort(num)[::-1]
    else:
        ind = numpy.argsort(num)
    return [dirStr[i] for i in ind], num[ind]
