import os
from fnmatch import fnmatch

import numpy as np


class RootFinder:
    def __init__(self, func, xo, method, **kwargs):
        self.func = func
        self.dim = len(xo)

        self.xMin = kwargs.get("xMin", -np.ones_like(xo) * float("Inf"))
        self.xMax = kwargs.get("xMax", np.ones_like(xo) * float("Inf"))
        self.maxIt = kwargs.get("maxIt", 100)
        self.dx0 = kwargs.get("firstStep", 1e-6)
        self.errLim = kwargs.get("errLim", 1e-6)
        self.relax = kwargs.get("relax", 1.0)
        self.dxMax = kwargs.get("dxMax", np.ones_like(xo) * float("Inf"))
        self.dxMaxInc = kwargs.get("dxMaxInc", self.dxMax)
        self.dxMaxDec = kwargs.get("dxMaxDec", self.dxMax)

        self.relax = kwargs.get("relax", 1.0)
        self.derivativeMethod = kwargs.get("derivativeMethod", "right")
        self.err = 1.0
        self.it = 0
        self.J = None
        self.xOld = None
        self.fOld = None
        self.maxJacobianResetStep = kwargs.get("maxJacobianResetStep", 5)
        self.converged = False

        # Calculate function value at initial point
        self.x = xo

        # Convert to numpy arrays
        for v in ["x", "xMin", "xMax", "dxMax", "dxMaxInc", "dxMaxDec"]:
            exec("self.{0} = np.array([a for a in self.{0}])".format(v))

        self.evalF()

        if method.lower() == "broyden":
            self.getStep = self.getStepBroyden
        else:
            self.getStep = self.getStepSecant

    def reinitialize(self, xo):
        self.err = 1.0
        self.it = 0
        self.J = None

        # Calculate function value at initial point
        self.x = xo
        self.evalF()

    def setMaxStep(self, *args):
        if len(args) == 1:
            self.dxMaxInc = args[0]
            self.dxMaxDec = args[0]
        else:
            self.dxMaxInc = args[0]
            self.dxMaxDec = args[1]

    def limitStep(self, dx=None):
        if dx is None:
            dx = self.dx
        dx *= self.relax

        x = self.x + dx

        x = np.max(np.vstack((x, self.xMin)), axis=0)
        x = np.min(np.vstack((x, self.xMax)), axis=0)

        dx = x - self.x

        dxLimPct = np.ones_like(dx)
        for i in range(len(dxLimPct)):
            if dx[i] > 0:
                dxLimPct[i] = np.min([dx[i], self.dxMaxInc[i]]) / dx[i]
            elif dx[i] < 0:
                dxLimPct[i] = np.max([dx[i], -self.dxMaxDec[i]]) / dx[i]

        dx *= np.min(dxLimPct)
        self.dx = dx

        return dx

    def storePrevStep(self):
        self.xOld = self.x * 1.0
        self.fOld = self.f * 1.0

    def evalErr(self):
        self.err = np.max(np.abs(self.dx + 1e-8))
        if self.df is not None:
            self.err += np.max(np.abs(self.df + 1e-8))
        else:
            self.err += np.max(np.abs(self.f))

    def evalF(self):
        self.f = self.func(self.x)
        if self.fOld is not None:
            self.df = self.f - self.fOld
        else:
            self.df = None

    def takeStep(self, dx=None):
        if dx is not None:
            self.dx = dx
        self.storePrevStep()
        self.x += self.dx
        self.evalF()
        self.it += 1

    def getStepSecant(self):
        if self.it == 0:
            self.dx = np.ones_like(self.x) * self.dx0
        else:
            self.dx = -self.f * (self.x - self.xOld) / (self.f - self.fOld + 1e-8)
        return self.dx

    def reset_jacobian(self):
        fo = self.f * 1.0
        xo = self.x * 1.0

        self.J = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            self.dx = np.zeros_like(self.x)
            self.dx[i] = self.dx0
            self.takeStep()
            self.J[:, i] = self.df / self.dx[i]
            self.f = fo
            self.x = xo

        self.step = 0

    def getStepBroyden(self):
        if self.it == 0 or self.J is None:
            self.reset_jacobian()

        dx = np.reshape(self.dx, (self.dim, 1))
        df = np.reshape(self.df, (self.dim, 1))

        self.J += np.dot(df - np.dot(self.J, dx), dx.T) / np.linalg.norm(dx) ** 2

        dx *= 0.0
        dof = [
            not x <= xMin and not x >= xMax
            for x, xMin, xMax in zip(self.x, self.xMin, self.xMax)
        ]
        if any(dof):
            A = -self.J
            b = self.f.reshape(self.dim, 1)
            dx[np.ix_(dof)] = np.linalg.solve(A[np.ix_(dof, dof)], b[np.ix_(dof)])

        if any(np.abs(self.f) - np.abs(self.fOld) > 0.0):
            self.step += 1

        if self.step >= self.maxJacobianResetStep:
            dx = np.ones_like(dx) * self.dx0
            self.J = None

        self.dx = dx.reshape(self.dim)

        return self.dx

    def solve(self):
        while self.err >= self.errLim and self.it < self.maxIt:
            self.getStep()
            self.limitStep()
            self.takeStep()
            self.evalErr()

        self.converged = self.err < self.errLim and not any(self.x <= self.xMin)

        return self.x


def mag(vec):
    return np.sum(np.array([veci ** 2 for veci in vec])) ** 0.5


def ang2vec(ang):
    return np.array([np.cos(ang), np.sin(ang)])


def ang2vecd(ang):
    return np.array([cosd(ang), sind(ang)])


def deg2rad(ang):
    return ang * np.pi / 180


def rad2deg(ang):
    return ang * 180 / np.pi


def cosd(ang):
    return np.cos(deg2rad(ang))


def sind(ang):
    return np.sin(deg2rad(ang))


def tand(ang):
    return np.tan(deg2rad(ang))


def acosd(slope):
    return rad2deg(np.arccos(slope))


def asind(slope):
    return rad2deg(np.arcsin(slope))


def atand(slope):
    return rad2deg(np.arctan(slope))


def atand2(y, x):
    return rad2deg(np.arctan2(y, x))


def cumdiff(x):
    return np.sum(np.diff(x))


def fzero(f, xo, **kwargs):
    maxIt = kwargs.get("maxIt", 100)
    dx0 = kwargs.get("firstStep", 1e-6)
    errLim = kwargs.get("errLim", 1e-6)
    printout = kwargs.get("printout", False)
    xname = kwargs.get("xname", "x")
    xscale = kwargs.get("xscale", 1.0)
    xmin = kwargs.get("xmin", -float("Inf"))
    xmax = kwargs.get("xmax", float("Inf"))
    nspace = kwargs.get("nspace", 0.0)
    method = kwargs.get("method", "Secant")

    if nspace == 0:
        space = " "
    else:
        space = " " * nspace

    if method == "Secant" or method == "Newton":

        def grad(x):
            dx = 1e-6
            return (f(x + dx) - f(x - dx)) / (2 * dx)

        err = 1
        it = 0
        xOld = xo
        fOld = f(xOld)
        xNew = xOld + dx0
        while err >= errLim and it < maxIt:
            fNew = f(xNew)

            if printout:
                string = "{5}Iteration {0}: {1} = {2:f}, {3} = {4:5.3e}"
                print((string.format(it, xname, xNew * xscale, "residual", err, space)))

            if method == "Newton":
                dx = -fNew / grad(xOld)
            elif method == "Secant":
                dx = -fNew * (xNew - xOld) / (fNew - fOld)

            xOld = xNew
            fOld = fNew
            xNew += dx

            xNew = max(min(xNew, xmax), xmin)
            dx = xNew - xOld

            err = np.abs(dx)
            it += 1
        return xNew

    elif method == "Golden":
        GR = (np.sqrt(5) - 1) / 2
        X = np.array([xmin, xmax - GR * (xmax - xmin), xmax])
        F = np.zeros_like(X)
        for i in range(len(X)):
            F[i] = np.abs(f(X[i]))

        err = 1
        it = 0
        while err > errLim:
            xNew = X[0] + GR * (X[2] - X[0])
            fNew = np.abs(f(xNew))

            if fNew < F[1]:
                X = np.array([X[1], xNew, X[2]])
                F = np.array([F[1], fNew, X[2]])
            else:
                X = np.array([xNew, X[1], X[0]])
                F = np.array([fNew, F[1], F[0]])
            err = np.abs(X[2] - X[0])
            it += 1

        return f((X[2] + X[0]) / 2)


def integrate(x, f):
    # Trapezoidal integration
    I = np.argsort(x)
    x = x[I]
    f = f[I]
    f[np.nonzero(np.abs(f) == float("Inf"))] = 0.0

    return 0.5 * np.sum((x[1:] - x[0:-1]) * (f[1:] + f[0:-1]))


def growPoints(x0, x1, xMax, rate=1.1):
    dx = x1 - x0
    x = [x1]

    def done(xt):
        if dx > 0:
            return xt > xMax
        elif dx < 0:
            return xt < xMax
        else:
            return True

    while not done(x[-1]):
        x.append(x[-1] + dx)
        dx *= rate

    return np.array(x[1:])


def fillPoints(x0, x1, L, pctLast, targetRate=1.1):
    dxLast = pctLast * L
    x2 = x1 + np.sign(x1 - x0) * (L - 0.5 * dxLast)

    def func(rate):
        return growPoints(x0, x1, x2, rate)

    def res1(rate):
        return np.abs(np.diff(func(rate)[-2:])[0]) - dxLast

    def res2(rate):
        return func(rate)[-1] - x2

    return func(fzero(res2, fzero(res1, targetRate)))


def getDerivative(f, x, direction="c"):
    dx = 1e-6
    fr = f(x + dx)
    fl = f(x - dx)

    if direction[0].lower() == "r" or np.isnan(fl):
        return (fr - f(x)) / dx
    elif direction[0].lower() == "l" or np.isnan(fr):
        return (f(x) - fl) / dx
    else:
        return (f(x + dx) - f(x - dx)) / (2 * dx)


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def rotateVec(v, ang):
    C = cosd(ang)
    S = sind(ang)

    return np.array([C * v[0] - S * v[1], S * v[0] + C * v[1]])


def rotatePt(oldPt, basePt, ang):
    relPos = np.array(oldPt) - np.array(basePt)
    newPos = rotateVec(relPos, ang)
    newPt = basePt + newPos

    return newPt


def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
        elif x > xs[-1]:
            return ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        else:
            return interpolator(x)

    return pointwise


def minMax(x):
    return min(x), max(x)


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


def inRange(x, lims):
    if x >= lims[0] and x <= lims[1]:
        return True
    else:
        return False


def createIfNotExist(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)  # , 0755)


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
    num = np.array(
        [
            float("".join(i for i in d.lower() if i.isdigit() or i == ".").strip("."))
            for d in dirStr
        ]
    )
    if direction == "reverse":
        ind = np.argsort(num)[::-1]
    else:
        ind = np.argsort(num)
    return [dirStr[i] for i in ind], num[ind]


def getFG(x):
    txMax = 5.0
    N = 20

    pt = np.array([-1.0, 0.0, 1.0]) * np.sqrt(3.0 / 5.0)
    w = np.array([5.0, 8.0, 5.0]) / 9

    t = np.linspace(0.0, txMax / x, N + 1)
    dt = 0.5 * (t[1] - t[0])

    f = 0.0
    g = 0.0
    for i in range(N):
        ti = t[i] + dt * (pt + 1)
        F = w * dt * np.exp(-ti * x) / (ti ** 2 + 1)

        f += np.sum(F)
        g += np.sum(F * ti)

    return f, g


def rm_rf(dList):
    dList = list(dList)
    for d in dList:
        for path in (os.path.join(d, f) for f in os.listdir(d)):
            if os.path.isdir(path):
                rm_rf(path)
            else:
                os.unlink(path)
        os.rmdir(d)


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


def checkDir(f):
    if not os.path.isdir(f):
        os.makedirs(f)


def listdir_nohidden(d):
    return [di for di in os.listdir(d) if not fnmatch(di, ".*")]


listdirNoHidden = listdir_nohidden
