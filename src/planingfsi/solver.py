"""Solvers for single and multi-dimensional nonlinear problems."""
from typing import Callable, Any

import numpy


class RootFinder:
    """A generalized root finder for solving nonlinear equations.

    Args:
        func: A callable which returns an array of residuals the same shape as the argument.
        xo: An array with an initial guess at the solution.
        method: The method to use for solving. Options are "broyden" and "secant".

    """

    def __init__(
        self,
        func: Callable[[numpy.ndarray], numpy.ndarray],
        xo: numpy.ndarray,
        method: str,
        **kwargs: Any,
    ):
        self.func = func
        self.x = numpy.array(xo)
        self.f = numpy.zeros_like(xo)
        self.dim = len(xo)

        self.x_min = numpy.array(kwargs.get("xMin", -numpy.ones_like(xo) * float("Inf")))
        self.x_max = numpy.array(kwargs.get("xMax", numpy.ones_like(xo) * float("Inf")))
        self.max_it = kwargs.get("maxIt", 100)
        self.dx_init = kwargs.get("firstStep", 1e-6)
        self.error_limit = kwargs.get("errLim", 1e-6)
        self.relax = kwargs.get("relax", 1.0)

        dx_max = kwargs.get("dxMax", numpy.ones_like(xo) * float("Inf"))
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

        # TODO: Replace with factory method and method overloading
        if method.lower() == "broyden":
            self.get_step = self.get_step_broyden
        else:
            self.get_step = self.get_step_secant

    def reinitialize(self, xo: numpy.ndarray) -> None:
        """Re-initialize the solver for a new solution."""
        self.err = 1.0
        self.it = 0
        self.jacobian = None

        # Calculate function value at initial point
        self.x = xo
        self.evaluate_function()

    def limit_step(self, dx: numpy.ndarray = None) -> numpy.ndarray:
        if dx is None:
            dx = self.dx
        dx *= self.relax

        x = self.x + dx

        x = numpy.max(numpy.vstack((x, self.x_min)), axis=0)
        x = numpy.min(numpy.vstack((x, self.x_max)), axis=0)

        dx = x - self.x

        dx_lim_pct = numpy.ones_like(dx)
        for i in range(len(dx_lim_pct)):
            if dx[i] > 0:
                dx_lim_pct[i] = numpy.min([dx[i], self.dx_max_increase[i]]) / dx[i]
            elif dx[i] < 0:
                dx_lim_pct[i] = numpy.max([dx[i], -self.dx_max_decrease[i]]) / dx[i]

        dx *= numpy.min(dx_lim_pct)
        self.dx = dx

        return dx

    def store_previous_step(self) -> None:
        """Store the results from the previous step."""
        self.x_prev = self.x * 1.0
        self.f_prev = self.f * 1.0

    def evaluate_error(self) -> None:
        """Evaluate the error residual."""
        self.err = numpy.max(numpy.abs(self.dx + 1e-8))
        if self.df is not None:
            self.err += numpy.max(numpy.abs(self.df + 1e-8))
        else:
            self.err += numpy.max(numpy.abs(self.f))

    def evaluate_function(self) -> None:
        """Evaluate the function and calculated the change."""
        self.f = self.func(self.x)
        if self.f_prev is not None:
            self.df = self.f - self.f_prev
        else:
            self.df = None

    def get_step_secant(self) -> numpy.ndarray:
        """Get the step using the Secant method."""
        if self.it == 0:
            self.dx = numpy.ones_like(self.x) * self.dx_init
        else:
            self.dx = -self.f * (self.x - self.x_prev) / (self.f - self.f_prev + 1e-8)
        return self.dx

    def take_step(self, dx: numpy.ndarray = None) -> None:
        """Take the step in the solution."""
        if dx is not None:
            self.dx = dx
        self.store_previous_step()
        self.x += self.dx
        self.evaluate_function()
        self.it += 1

    def reset_jacobian(self) -> None:
        """Reset the Jacobian."""
        fo = self.f * 1.0
        xo = self.x * 1.0

        self.jacobian = numpy.zeros((self.dim, self.dim))
        for i in range(self.dim):
            self.dx = numpy.zeros_like(self.x)
            self.dx[i] = self.dx_init
            self.take_step()
            self.jacobian[:, i] = self.df / self.dx[i]
            self.f = fo
            self.x = xo

        self.jacobian_step = 0

    def get_step_broyden(self) -> numpy.ndarray:
        """Get the next step using Broyden's method."""
        if self.it == 0 or self.jacobian is None:
            self.reset_jacobian()

        dx = numpy.reshape(self.dx, (self.dim, 1))
        df = numpy.reshape(self.df, (self.dim, 1))

        self.jacobian += (
            numpy.dot(df - numpy.dot(self.jacobian, dx), dx.T) / numpy.linalg.norm(dx) ** 2
        )
        dx *= 0.0
        dof = [
            not x <= xMin and not x >= xMax for x, xMin, xMax in zip(self.x, self.x_min, self.x_max)
        ]
        if any(dof):
            b = self.f.reshape(self.dim, 1)
            dx[numpy.ix_(dof)] = numpy.linalg.solve(
                -self.jacobian[numpy.ix_(dof, dof)], b[numpy.ix_(dof)]
            )

        if any(numpy.abs(self.f) - numpy.abs(self.f_prev) > 0.0):
            self.jacobian_step += 1

        if self.jacobian_step >= self.max_jacobian_reset_step:
            dx = numpy.ones_like(dx) * self.dx_init
            self.jacobian = None

        self.dx = dx.reshape(self.dim)

        return self.dx

    def solve(self) -> numpy.ndarray:
        """Solve the nonlinear problem."""
        while self.err >= self.error_limit and self.it < self.max_it:
            self.get_step()
            self.limit_step()
            self.take_step()
            self.evaluate_error()

        self.is_converged = self.err < self.error_limit and not any(self.x <= self.x_min)

        return self.x


def fzero(
    func: Callable[[float], float],
    x_init: float,
    *,
    max_it: int = 100,
    first_step: float = 1e-6,
    err_lim: float = 1e-6,
) -> float:
    """Find the root of a scalar function func with an initial guess x_init.

    Args:
        func: The function for which to find the zero-crossing point.
        x_init: The initial guess of the function's solution.
        max_it: Maximum number of iterations.
        first_step: Length of first step.
        err_lim: Tolerance for iteration residual.

    Returns:
        The value x such that func(x) == 0.0, within a given tolerance.

    """
    error = 1.0
    it_num = 0
    x_old = x_init
    f_old = func(x_old)
    x_new = x_old + first_step
    while error >= err_lim and it_num < max_it:
        f_new = func(x_new)
        delta_x = -f_new * (x_new - x_old) / (f_new - f_old)
        x_old, f_old = x_new, f_new
        x_new += delta_x
        error = numpy.abs(x_new - x_old)
        it_num += 1
    return x_new
