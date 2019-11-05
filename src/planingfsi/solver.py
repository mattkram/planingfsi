from typing import Callable, Any

import numpy

import numpy as np


class RootFinder:
    """A generalized root finder for solving nonlinear equations.

    Args:
        func: A callable which returns an array of residuals the same shape as the argument.
        xo: An array with an initial guess at the solution.
        method: The method to use for solving. Options are "broyden" and "secant".

    """

    def __init__(
        self,
        func: Callable[[numpy.array], numpy.array],
        xo: numpy.array,
        method: str,
        **kwargs: Any
    ):
        self.func = func
        self.x = numpy.array(xo)
        self.f = numpy.zeros_like(xo)
        self.dim = len(xo)

        self.x_min = numpy.array(kwargs.get("xMin", -np.ones_like(xo) * float("Inf")))
        self.x_max = numpy.array(kwargs.get("xMax", np.ones_like(xo) * float("Inf")))
        self.max_it = kwargs.get("maxIt", 100)
        self.dx_init = kwargs.get("firstStep", 1e-6)
        self.error_limit = kwargs.get("errLim", 1e-6)
        self.relax = kwargs.get("relax", 1.0)

        dx_max = kwargs.get("dxMax", np.ones_like(xo) * float("Inf"))
        self.dx_max_increase = numpy.array(kwargs.get("dxMaxInc", dx_max))
        self.dx_max_decrease = numpy.array(kwargs.get("dxMaxDec", dx_max))

        self.derivative_method = kwargs.get("derivativeMethod", "right")
        self.err = 1.0
        self.it = 0
        self.jacobian: numpy.array = None
        self.x_prev = numpy.zeros((self.dim,))
        self.f_prev = numpy.zeros((self.dim,))
        self.max_jacobian_reset_step = kwargs.get("maxJacobianResetStep", 5)
        self.is_converged = False

        # Calculate function value at initial point
        self.dx = numpy.zeros_like(xo)
        self.df = numpy.zeros_like(xo)

        self.evaluate_function()

        self.jacobian_step = 0

        if method.lower() == "broyden":
            self.get_step = self.get_step_broyden
        else:
            self.get_step = self.get_step_secant

    def reinitialize(self, xo: numpy.array) -> None:
        self.err = 1.0
        self.it = 0
        self.jacobian = None

        # Calculate function value at initial point
        self.x = xo
        self.evaluate_function()

    def _limit_step(self, dx: numpy.array = None) -> numpy.array:
        if dx is None:
            dx = self.dx
        dx *= self.relax

        x = self.x + dx

        x = np.max(np.vstack((x, self.x_min)), axis=0)
        x = np.min(np.vstack((x, self.x_max)), axis=0)

        dx = x - self.x

        dx_lim_pct = np.ones_like(dx)
        for i in range(len(dx_lim_pct)):
            if dx[i] > 0:
                dx_lim_pct[i] = np.min([dx[i], self.dx_max_increase[i]]) / dx[i]
            elif dx[i] < 0:
                dx_lim_pct[i] = np.max([dx[i], -self.dx_max_decrease[i]]) / dx[i]

        dx *= np.min(dx_lim_pct)
        self.dx = dx

        return dx

    def store_previous_step(self) -> None:
        self.x_prev = self.x * 1.0
        self.f_prev = self.f * 1.0

    def evaluate_error(self) -> None:
        self.err = np.max(np.abs(self.dx + 1e-8))
        if self.df is not None:
            self.err += np.max(np.abs(self.df + 1e-8))
        else:
            self.err += np.max(np.abs(self.f))

    def evaluate_function(self) -> None:
        self.f = self.func(self.x)
        if self.f_prev is not None:
            self.df = self.f - self.f_prev
        else:
            self.df = None

    def get_step_secant(self) -> numpy.array:
        if self.it == 0:
            self.dx = np.ones_like(self.x) * self.dx_init
        else:
            self.dx = -self.f * (self.x - self.x_prev) / (self.f - self.f_prev + 1e-8)
        return self.dx

    def take_step(self, dx: numpy.array = None) -> None:
        if dx is not None:
            self.dx = dx
        self.store_previous_step()
        self.x += self.dx
        self.evaluate_function()
        self.it += 1

    def reset_jacobian(self) -> None:
        fo = self.f * 1.0
        xo = self.x * 1.0

        self.jacobian = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            self.dx = np.zeros_like(self.x)
            self.dx[i] = self.dx_init
            self.take_step()
            self.jacobian[:, i] = self.df / self.dx[i]
            self.f = fo
            self.x = xo

        self.jacobian_step = 0

    def get_step_broyden(self) -> numpy.array:
        if self.it == 0 or self.jacobian is None:
            self.reset_jacobian()

        dx = np.reshape(self.dx, (self.dim, 1))
        df = np.reshape(self.df, (self.dim, 1))

        self.jacobian += (
            np.dot(df - np.dot(self.jacobian, dx), dx.T) / np.linalg.norm(dx) ** 2
        )
        dx *= 0.0
        dof = [
            not x <= xMin and not x >= xMax
            for x, xMin, xMax in zip(self.x, self.x_min, self.x_max)
        ]
        if any(dof):
            b = self.f.reshape(self.dim, 1)
            dx[np.ix_(dof)] = np.linalg.solve(-self.jacobian[np.ix_(dof, dof)], b[np.ix_(dof)])

        if any(np.abs(self.f) - np.abs(self.f_prev) > 0.0):
            self.jacobian_step += 1

        if self.jacobian_step >= self.max_jacobian_reset_step:
            dx = np.ones_like(dx) * self.dx_init
            self.jacobian = None

        self.dx = dx.reshape(self.dim)

        return self.dx

    def solve(self) -> numpy.array:
        while self.err >= self.error_limit and self.it < self.max_it:
            self.get_step()
            self._limit_step()
            self.take_step()
            self.evaluate_error()

        self.is_converged = self.err < self.error_limit and not any(
            self.x <= self.x_min
        )

        return self.x


def fzero(func: Callable[[float], float], x_init: float, **kwargs: Any) -> float:
    """Find the root of a scalar function func with an initial guess x_init.

    Args:
        func: The function for which to find the zero-crossing point.
        x_init: The initial guess of the function's solution.

    Keyword Args:
        max_it (int): Maximum number of iterations (default=100)
        first_step (float): Length of first step (default=1e-6)
        err_lim (float): Tolerance for iteration residual (default=1e-6)
        x_min (float): Minimum value for x-variable (default=-Inf)
        x_max (float): Maximum value for x-variable (default=+Inf)

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
