from __future__ import annotations

import os
from collections.abc import Callable
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from planingfsi import trig

if TYPE_CHECKING:
    from planingfsi.config import Config
    from planingfsi.fe.rigid_body import RigidBody
    from planingfsi.fe.structure import StructuralSolver
    from planingfsi.fe.substructure import Substructure
    from planingfsi.potentialflow.solver import PotentialPlaningSolver
    from planingfsi.simulation import Simulation


class FSIFigure:
    """A wrapper around a Matplotlib figure to plot the results/status of a PlaningFSI simulation."""

    def __init__(self, simulation: Simulation):
        self.simulation = simulation

        self.figure = fig = plt.figure(figsize=(16, 12))
        if self.config.plotting.watch:
            plt.ion()
        self.geometry_ax = fig.add_axes([0.05, 0.6, 0.9, 0.35])

        (self.line_nodes,) = plt.plot([], [], "ko")

        (self._handle_fs,) = self.geometry_ax.plot([], [], "b-")
        (self._handle_fs_init,) = self.geometry_ax.plot([], [], "b--")

        # Line handles for element initial and current positions
        self.element_handles_0 = {}
        self.element_handles = {}

        # Line handles for the air and fluid pressure profiles
        self.line_air_pressure = {}
        self.line_fluid_pressure = {}
        for struct in self.solid.substructures:
            (self.line_air_pressure[struct],) = plt.plot([], [], "g-")
            (self.line_fluid_pressure[struct],) = plt.plot([], [], "r-")
            for el in struct.elements:
                (self.element_handles_0[el],) = plt.plot([], [], "k--")
                (self.element_handles[el],) = plt.plot([], [], "k-", linewidth=2)

        self.lineCofR: list["CofRPlot"] = []
        for bodies in self.solid.rigid_bodies:
            CofRPlot(
                self.geometry_ax,
                bodies,
                grid_len=self.config.plotting.CofR_grid_len,
                symbol=False,
                style="k--",
                marker="s",
                fill=False,
            )
            self.lineCofR.append(
                CofRPlot(
                    self.geometry_ax,
                    bodies,
                    grid_len=self.config.plotting.CofR_grid_len,
                    symbol=False,
                    style="k-",
                    marker="o",
                )
            )

        x = [nd.x for struct in self.solid.substructures for nd in struct.nodes]
        y = [nd.y for struct in self.solid.substructures for nd in struct.nodes]
        xMin, xMax = min(x), max(x)
        yMin, yMax = min(y), max(y)

        if self.config.plotting.xmin is not None:
            xMin = self.config.plotting.xmin
            self.config.plotting.ext_w = 0.0

        if self.config.plotting.xmax is not None:
            xMax = self.config.plotting.xmax
            self.config.plotting.ext_e = 0.0

        if self.config.plotting.ymin is not None:
            yMin = self.config.plotting.ymin
            self.config.plotting.ext_s = 0.0

        if self.config.plotting.ymax is not None:
            yMax = self.config.plotting.ymax
            self.config.plotting.ext_n = 0.0

        plt.xlabel(r"$x$ [m]", fontsize=22)
        plt.ylabel(r"$y$ [m]", fontsize=22)

        plt.xlim(
            [
                xMin - (xMax - xMin) * self.config.plotting.ext_w,
                xMax + (xMax - xMin) * self.config.plotting.ext_e,
            ]
        )
        plt.ylim(
            [
                yMin - (yMax - yMin) * self.config.plotting.ext_s,
                yMax + (yMax - yMin) * self.config.plotting.ext_n,
            ]
        )
        plt.gca().set_aspect("equal")
        self.TXT = plt.text(0.05, 0.95, "", ha="left", va="top", transform=plt.gca().transAxes)

        self.subplot: list[TimeHistoryAxes] = []

        if self.solid.rigid_bodies:
            body = self.solid.rigid_bodies[0]
            self.subplot.append(ForceSubplot((0.70, 0.30, 0.25, 0.2), body=body, parent=self))
            self.subplot.append(MotionSubplot((0.70, 0.05, 0.25, 0.2), body=body, parent=self))

        if len(self.solid.rigid_bodies) > 1:
            body = self.solid.rigid_bodies[1]
            self.subplot.append(ForceSubplot((0.05, 0.30, 0.25, 0.2), body=body, parent=self))
            self.subplot.append(MotionSubplot((0.05, 0.05, 0.25, 0.2), body=body, parent=self))

        self.subplot.append(ResidualSubplot((0.40, 0.05, 0.25, 0.45), self.solid, parent=self))

    @property
    def config(self) -> Config:
        return self.simulation.config

    @property
    def solid(self) -> "StructuralSolver":
        return self.simulation.structural_solver

    @property
    def fluid(self) -> "PotentialPlaningSolver":
        return self.simulation.fluid_solver

    def _draw_free_surface(self) -> None:
        """Draw the actual and undisturbed free-surface lines."""
        self._handle_fs.set_data(self.fluid.x_coord_fs, self.fluid.z_coord_fs)
        end_pts = np.array([self.config.plotting.x_fs_min, self.config.plotting.x_fs_max])
        self._handle_fs_init.set_data(
            end_pts, self.config.flow.waterline_height * np.ones_like(end_pts)
        )

    @staticmethod
    def _get_pressure_plot_points(ss, s0: np.ndarray, p0: np.ndarray) -> Iterable[Iterable]:
        """Get coordinates required to plot pressure profile as lines."""
        sp = [(s, p) for s, p in zip(s0, p0) if not np.abs(p) < 1e-4]

        if not sp:
            return [], []

        s0, p0 = list(zip(*sp))
        nVec = list(map(ss.get_normal_vector, s0))
        coords0 = [ss.get_coordinates(s) for s in s0]
        coords1 = [
            c + ss.config.plotting.pressure_scale * p * n for c, p, n in zip(coords0, p0, nVec)
        ]

        return list(
            zip(*[xyi for c0, c1 in zip(coords0, coords1) for xyi in [c0, c1, np.ones(2) * np.nan]])
        )

    def _draw_pressure_profiles(self, ss: Substructure) -> None:
        """Plot the internal and external pressure profiles as lines."""
        if handle := self.line_fluid_pressure.get(ss):
            handle.set_data(self._get_pressure_plot_points(ss, ss.s_hydro, ss.p_hydro))
        if handle := self.line_air_pressure.get(ss):
            handle.set_data(self._get_pressure_plot_points(ss, ss.s_air, -ss.p_air))

    def _draw_substructure(self, ss: Substructure) -> None:
        """Plot the substructure elements and pressure profiles."""
        for el in ss.elements:
            if handle := self.element_handles.get(el):
                handle.set_data([nd.x for nd in el.nodes], [nd.y for nd in el.nodes])

            if handle := self.element_handles_0.get(el):
                base_pt = np.array([el.parent.rigid_body.x_cr_init, el.parent.rigid_body.y_cr_init])
                pos = [
                    trig.rotate_point(pos, base_pt, el.parent.rigid_body.trim)
                    - np.array([0, el.parent.rigid_body.draft])
                    for pos in el._initial_coordinates
                ]
                x, y = list(zip(*[[posi[i] for i in range(2)] for posi in pos]))
                handle.set_data(x, y)

        #    for nd in [self.node[0],self.node[-1]]:
        #      nd.plot()
        self._draw_pressure_profiles(ss)

    def _draw_structures(self):
        """Plot the structure."""
        for body in self.solid.rigid_bodies:
            for struct in body.substructures:
                self._draw_substructure(struct)

    def update(self) -> None:
        self._draw_structures()
        self._draw_free_surface()
        self.TXT.set_text(
            "\n".join(
                [
                    f"Iteration {self.simulation.it}",
                    f"$Fr={self.config.flow.froude_num:>8.3f}$",
                    f"$\\bar{{P_c}}={self.config.body.PcBar:>8.3f}$",
                ]
            )
        )

        # Update each lower subplot
        for s in self.subplot:
            s.update(
                self.solid.residual < self.config.solver.max_residual
                and self.simulation.it > self.config.solver.num_ramp_it
            )

        for line in self.lineCofR:
            line.update()

        plt.draw()

        if self.config.plotting.save:
            self.save()

    def write_time_histories(self) -> None:
        for s in self.subplot:
            if isinstance(s, TimeHistoryAxes):
                s.write()

    def save(self) -> None:
        plt.savefig(
            os.path.join(
                self.config.path.fig_dir_name,
                "frame{1:04d}.{0}".format(self.config.plotting.fig_format, self.simulation.it),
            ),
            format=self.config.plotting.fig_format,
        )  # , dpi=300)

    def show(self) -> None:
        plt.show(block=True)


class Series:
    """A container to represent a series on a figure.

    Args:
        x_func: A callable which returns x-coordinates of the series points.
        y_func: A callable which returns y-coordinates of the series points.
        include_history: If True, the full history will be stored and drawn. Otherwise,
            only the latest point will be drawn.
        ax: An axis handle. Defaults to `plt.gca()`.
        label: An optional label to use for the plot legend.
        ignore_first: If True, ignore the first point of the series.
        style: The style to use for the history line.
        style_current: The style to use for the symbol representing the current value.

    """

    def __init__(
        self,
        x_func: Callable[[], float],
        y_func: Callable[[], float],
        *,
        include_history: bool = True,
        ax: Axes | None = None,
        label: str | None = None,
        ignore_first: bool = False,
        style: str = "k-",
        style_current: str = "ro",
    ) -> None:
        self._points: list[tuple[float, float]] = []

        self._get_x = x_func
        self._get_y = y_func

        self.label = label
        self._ignore_first = ignore_first

        ax = ax or plt.gca()
        (self.line_handle,) = ax.plot([], [], style, label=label)

        self._current_value_series: Series | None = None
        if include_history:
            self._current_value_series = Series(
                x_func,
                y_func,
                ax=ax,
                include_history=False,
                style=style_current,
            )

    def update(self, is_final: bool = False) -> None:
        """Update the line data by calling the assigned callback functions.

        Args:
            is_final: If True, it is the final iteration and the current value will
                be drawn as a yellow star.
        """
        if self._current_value_series is None:
            self._points = [(self._get_x(), self._get_y())]
        elif self._ignore_first and not self._points:
            self._points.append((np.nan, np.nan))
        else:
            self._points.append((self._get_x(), self._get_y()))

        self.line_handle.set_data(*zip(*self._points))
        if self._current_value_series is not None:
            self._current_value_series.update(is_final)

        if is_final:
            self.line_handle.set_marker("*")
            self.line_handle.set_markerfacecolor("y")
            self.line_handle.set_markersize(10)


class TimeHistoryAxes:
    """A collection of axes (subplot) for displaying iteration series'.

    Args:
        pos: The relative position of the axes in the figure.
        parent: A reference to the parent FSIFigure.
        name: A name for the subplot, used when exporting plot data to file.

    """

    def __init__(
        self, pos: tuple[float, float, float, float], *, name: str = "default", parent: FSIFigure
    ) -> None:
        self._parent = parent
        self._name = name
        self._series: list[Series] = []
        self._ax: list[Axes] = []

        self._add_axes(parent.figure.add_axes(pos))

    def _add_axes(self, ax: Axes) -> None:
        """Add a set of child axes."""
        self._ax.append(ax)

    def _add_y_axes(self) -> None:
        """Add a twin y-axis, which shares an x-axis but has a different y-axis scale."""
        self._add_axes(plt.twinx(self._ax[0]))

    def add_series(self, series: Series) -> Series:
        """Add a series to the subplot."""
        self._series.append(series)
        return series

    def set_properties(self, ax_ind: int = 0, **kwargs: Any) -> None:
        """Set the properties of a specific axis by calling matplotlib's `plt.setp()` function.

        Args:
            ax_ind: The index of the axes.
            kwargs: Keyword arguments to pass through to `plt.setp()`.

        """
        plt.setp(self._ax[ax_ind], **kwargs)

    def create_legend(self, ax_ind: int = 0) -> None:
        """Create a legend for a specific set of axes.

        Args:
            ax_ind: The index of the axes.

        """
        self._ax[ax_ind].legend(loc="lower left")

    def update(self, is_final: bool = False) -> None:
        """Update the figure."""
        for s in self._series:
            s.update(is_final)

        for ax in self._ax:
            self._reset_axis_limits(ax)

    def _reset_axis_limits(self, ax: Axes) -> None:
        x_min, x_max = 0, self._parent.simulation.it + 5
        y_min, y_max = np.nan, np.nan
        for i, l in enumerate(ax.get_lines()):
            y = l.get_ydata()
            if len(np.shape(y)) == 0:
                y = np.array([y])

            try:
                y = y[np.ix_(~np.isnan(y))]
            except TypeError:
                y = []

            if not np.shape(y)[0] == 0:
                y_min_l = np.min(y) - 0.02
                y_max_l = np.max(y) + 0.02
                if i == 0 or np.isnan(y_min):
                    y_min, y_max = y_min_l, y_max_l
                else:
                    y_min = np.min([y_min, y_min_l])
                    y_max = np.max([y_max, y_max_l])

        ax.set_xlim(x_min, x_max)
        if ~np.isnan(y_min) and ~np.isnan(y_max):
            ax.set_ylim(y_min, y_max)

    def write(self) -> None:
        """Write the time series results from the axes to a file."""
        with Path(f"{self._name}_timeHistories.txt").open("w") as ff:
            x = [pt[0] for pt in self._series[0]._points]
            for i, x_i in enumerate(x):
                ff.write("{0:4.0f}".format(x_i))
                for s in self._series:
                    ff.write(" {0:8.6e}".format(s._points[i][1]))
                ff.write("\n")


class MotionSubplot(TimeHistoryAxes):
    def __init__(
        self, pos: tuple[float, float, float, float], *, body: RigidBody, parent: FSIFigure
    ):
        super().__init__(pos, name=body.name, parent=parent)

        def itFunc() -> int:
            return self._parent.simulation.it

        def drftFunc() -> float:
            return body.draft

        def trimFunc() -> float:
            return body.trim

        self._add_y_axes()

        # Add plot series to appropriate axis
        self.add_series(
            Series(
                itFunc,
                drftFunc,
                ax=self._ax[0],
                style="b-",
                label="Draft",
            )
        )
        self.add_series(
            Series(
                itFunc,
                trimFunc,
                ax=self._ax[1],
                style="r-",
                label="Trim",
            )
        )

        self.set_properties(
            title=r"Motion History: {0}".format(body.name),
            xlabel=r"Iteration",
            ylabel=r"$d$ [m]",
        )
        self.set_properties(1, ylabel=r"$\theta$ [deg]")
        self.create_legend()


class ForceSubplot(TimeHistoryAxes):
    def __init__(
        self, pos: tuple[float, float, float, float], *, body: RigidBody, parent: FSIFigure
    ):
        super().__init__(pos, name=body.name, parent=parent)

        def itFunc() -> int:
            return self._parent.simulation.it

        def liftFunc() -> float:
            return body.loads.L / body.weight

        def dragFunc() -> float:
            return body.loads.D / body.weight

        def momFunc() -> float:
            return body.loads.M / (body.weight * self._parent.config.body.reference_length)

        self.add_series(
            Series(
                itFunc,
                liftFunc,
                style="r-",
                label="Lift",
                ignore_first=True,
            )
        )
        self.add_series(
            Series(
                itFunc,
                dragFunc,
                style="b-",
                label="Drag",
                ignore_first=True,
            )
        )
        self.add_series(
            Series(
                itFunc,
                momFunc,
                style="g-",
                label="Moment",
                ignore_first=True,
            )
        )

        self.set_properties(
            title=r"Force & Moment History: {0}".format(body.name),
            #                       xlabel=r'Iteration', \
            ylabel=r"$\mathcal{D}/W$, $\mathcal{L}/W$, $\mathcal{M}/WL_c$",
        )
        self.create_legend()


class ResidualSubplot(TimeHistoryAxes):
    def __init__(
        self, pos: tuple[float, float, float, float], solid: "StructuralSolver", parent: FSIFigure
    ):
        super().__init__(pos, name="residuals", parent=parent)

        def itFunc() -> int:
            return self._parent.simulation.it

        col = ["r", "b", "g"]
        for bd, coli in zip(solid.rigid_bodies, col):
            self.add_series(
                Series(
                    itFunc,
                    bd.get_res_lift,
                    style="{0}-".format(coli),
                    label="ResL: {0}".format(bd.name),
                    ignore_first=True,
                )
            )
            self.add_series(
                Series(
                    itFunc,
                    bd.get_res_moment,
                    style="{0}--".format(coli),
                    label="ResM: {0}".format(bd.name),
                    ignore_first=True,
                )
            )

        self.add_series(
            Series(
                itFunc,
                lambda: np.abs(solid.residual),
                style="k-",
                label="Total",
                ignore_first=True,
            )
        )

        self.set_properties(title=r"Residual History", yscale="log")
        self.create_legend()


class CofRPlot:
    def __init__(self, ax: Axes, body: RigidBody, **kwargs: Any):
        self.ax = ax
        self.body = body
        self.symbol = kwargs.get("symbol", True)
        style = kwargs.get("style", "k-")
        fill = kwargs.get("fill", True)
        marker = kwargs.get("marker", "o")
        if not fill:
            fcol = "w"
        else:
            fcol = "k"
        self._grid_len = kwargs.get("grid_len", 0.5)

        (self.line,) = self.ax.plot([], [], style)
        (self.lineCofG,) = self.ax.plot([], [], "k" + marker, markersize=8, markerfacecolor=fcol)
        self.update()

    def update(self) -> None:
        c = np.array([self.body.x_cr, self.body.y_cr])
        hvec = trig.rotate_vec_2d(np.array([0.5 * self._grid_len, 0.0]), self.body.trim)
        vvec = trig.rotate_vec_2d(np.array([0.0, 0.5 * self._grid_len]), self.body.trim)
        pts = np.array([c - hvec, c + hvec, c, c - vvec, c + vvec])
        self.line.set_data(pts.T)
        self.lineCofG.set_data(self.body.x_cg, self.body.y_cg)


def plot_pressure(solver: "PotentialPlaningSolver", fig_format: str = "png") -> None:
    """Create a plot of the pressure and shear stress profiles."""
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 5.0))

    for el in solver.pressure_elements:
        ax.plot(*el.plot_coords, color=el.plot_color, linestyle="-")
        ax.plot(el.x_coord * np.ones(2), [0.0, el.pressure], color=el.plot_color, linestyle="--")

    ax.plot(solver.x_coord, solver.pressure, "k-")
    # ax.plot(solver.x_coord, solver.shear_stress * 1000, "c--")

    # Scale x and y axes
    # for line in ax.lines:
    #     x, y = line.get_data()
    #     line.set_data(
    #         x / (config.body.reference_length * 2), y / config.flow.stagnation_pressure,
    #     )

    ax.set_xlabel(r"$x\,\mathrm{[m]}$")
    ax.set_ylabel(r"$p\,\mathrm{[kPa]}$")
    # ax.set_xlabel(r"$x/L_i$")
    # ax.set_ylabel(r"$p/(1/2\rho U^2)$")
    ax.set_ylim(ymin=0.0)

    fig.savefig(f"pressureElements.{fig_format}", format=fig_format)
