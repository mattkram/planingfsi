import numpy

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

    def getStepSecant(self):
        if self.it == 0:
            self.dx = np.ones_like(self.x) * self.dx0
        else:
            self.dx = -self.f * (self.x - self.xOld) / (self.f - self.fOld + 1e-8)
        return self.dx

    def takeStep(self, dx=None):
        if not dx is None:
            self.dx = dx
        self.storePrevStep()
        self.x += self.dx
        self.evalF()
        self.it += 1

    def resetJacobian(self):
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
            self.resetJacobian()

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


class RootFinderNew(RootFinder):
    def takeStep(self, dx=None):
        if not dx is None:
            self.dx = dx
        self.storePrevStep()
        self.x += self.dx
        self.evalF()
        self.it += 1

    def resetJacobian(self):
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
            self.resetJacobian()

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


def fzero(func, x_init, **kwargs):
    """Find the root of a function func with an initial guess x_init.

    Parameters
    ----------
    func : function
        The function for which to find the zero-crossing point.
    x_init : float
        The initial guess of the function's solution.

    Keyword Parameters
    ------------------
    max_it : int
        Maximum number of iterations (default=100)
    first_step : float
        Length of first step (default=1e-6)
    err_lim : float
        Tolerance for iteration residual (default=1e-6)
    xmin : float
        Minimum value for x-variable (default=-Inf)
    xmax : float
        Maximum value for x-variable (default=+Inf)

    """
    max_it = kwargs.get("maxIt", 100)
    first_step = kwargs.get("first_step", 1e-6)
    err_lim = kwargs.get("err_lim", 1e-6)
    x_min = kwargs.get("xmin", -float("Inf"))
    x_max = kwargs.get("xmax", float("Inf"))

    error = 1.0
    it_num = 0
    x_old = x_init
    f_old = func(x_old)
    x_new = x_old + first_step
    while error >= err_lim and it_num < max_it:
        f_new = func(x_new)
        delta_x = -f_new * (x_new - x_old) / (f_new - f_old)
        x_old, f_old = x_new, f_new
        x_new = max(min(x_new + delta_x, x_max), x_min)
        error = numpy.abs(x_new - x_old)
        it_num += 1
    return x_new


# def fzero(f, xo, **kwargs):
#     '''
#     Find the root of a function f with an initial guess xo.
#
#     Available keyword arguments:
#     maxIt     : maximum number of iterations (default=100)
#     firstStep : length of first step (default=1e-6)
#     errLim    : tolerance for iteration residual (default=1e-6)
#     printOut  : Boolean for whether to print results as solver runs (default=False)
#     xscale    : scale factor for x-variable (default=1.0)
#     xmin      : minimum value for x-variable (default=-Inf)
#     xmax      : maximum value for x-variable (default=+Inf)
#     method    : method to use for solving, options: ['Secant', 'Golden'] (default='Secant')
#     '''
#     maxIt     = kwargs.get('maxIt', 100)
#     dx0       = kwargs.get('firstStep', 1e-6)
#     errLim    = kwargs.get('errLim', 1e-6)
#     printout  = kwargs.get('printout', False)
#     xname     = kwargs.get('xname', 'x')
#     xscale    = kwargs.get('xscale', 1.0)
#     xmin      = kwargs.get('xmin', -float('Inf'))
#     xmax      = kwargs.get('xmax',  float('Inf'))
#     nspace    = kwargs.get('nspace', 0.0)
#     method    = kwargs.get('method', 'Secant')
#
#     if nspace == 0:
#         space = ' '
#     else:
#         space = ' ' * nspace
#
#     if method == 'Secant' or method == 'Newton':
#         def grad(x):
#             dx = 1e-6
#             return (f(x+dx) - f(x-dx)) / (2*dx)
#
#         err  = 1
#         it   = 0
#         xOld = xo
#         fOld = f(xOld)
#         xNew = xOld + dx0
#         while err >= errLim and it < maxIt:
#             fNew  = f(xNew)
#
#             if printout:
#                 print '{5}Iteration {0}: {1} = {2:f}, {3} = {4:5.3e}'.format(it, xname, xNew*xscale, 'residual', err, space)
#
#             if method == 'Newton':
#                 dx = -fNew / grad(xOld)
#             elif method == 'Secant':
#                 dx    = -fNew * (xNew - xOld) / (fNew - fOld)
#
#             xOld  = xNew
#             fOld  = fNew
#             xNew += dx
#
#             xNew = max(min(xNew, xmax), xmin)
#             dx  = xNew - xOld
#
#             err  = np.abs(dx)
#             it  += 1
#         return xNew
#
#     elif method == 'Golden':
#         GR = (np.sqrt(5) - 1) / 2
#         X = np.array([xmin, xmax - GR * (xmax - xmin), xmax])
#         F = np.zeros_like(X)
#         for i in range(len(X)):
#             F[i] = np.abs(f(X[i]))
#
#         err = 1
#         it = 0
#         while err > errLim:
#             xNew = X[0] + GR * (X[2] - X[0])
#             fNew = np.abs(f(xNew))
#
#             if fNew < F[1]:
#                 X = np.array([X[1], xNew, X[2]])
#                 F = np.array([F[1], fNew, X[2]])
#             else:
#                 X = np.array([xNew, X[1], X[0]])
#                 F = np.array([fNew, F[1], F[0]])
#             err = np.abs(X[2] - X[0])
#             it += 1
#
#         return f((X[2] + X[0]) / 2)
