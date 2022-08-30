from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator
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


class Figure:
    """A wrapper around a Matplotlib figure to plot the results/status of a PlaningFSI simulation."""

    def __init__(self, simulation: Simulation, *, figsize=(16, 9)):
        self.simulation = simulation

        self.figure = plt.figure(figsize=figsize)
        if self.config.plotting.watch:
            plt.ion()

        self.subplots: list[Subplot] = list(self.create_subplots())

    @property
    def config(self) -> Config:
        return self.simulation.config

    @property
    def solid(self) -> StructuralSolver:
        return self.simulation.structural_solver

    def create_subplots(self) -> Iterator[Subplot]:
        """A generator method that is used to create all the subplots.

        This method can be overridden in a subclass to create different layouts.

        """
        yield GeometrySubplot((0.05, 0.6, 0.9, 0.35), parent=self)
        yield ResidualSubplot((0.40, 0.05, 0.25, 0.45), parent=self)

        if self.solid.rigid_bodies:
            body = self.solid.rigid_bodies[0]
            yield ForceSubplot((0.70, 0.30, 0.25, 0.2), body=body, parent=self)
            yield MotionSubplot((0.70, 0.05, 0.25, 0.2), body=body, parent=self)

        if len(self.solid.rigid_bodies) > 1:
            body = self.solid.rigid_bodies[1]
            yield ForceSubplot((0.05, 0.30, 0.25, 0.2), body=body, parent=self)
            yield MotionSubplot((0.05, 0.05, 0.25, 0.2), body=body, parent=self)

    def update(self) -> None:
        """Update all subplots and redraw. If configured, also save figure as an image."""
        for s in self.subplots:
            s.update(is_final=self.simulation.is_converged)

        plt.draw()

        if self.config.plotting.save:
            self.save()

    def write_time_histories(self) -> None:
        for s in self.subplots:
            if isinstance(s, TimeHistorySubplot):
                s.write()

    def save(self) -> None:
        """Save the figure to a file, numbered by the iteration.

        If it is the first iteration, clear out any existing figures.

        """
        self.simulation.fig_dir.mkdir(exist_ok=True)

        if self.simulation.it == 0:
            for f in self.simulation.fig_dir.glob("*"):
                f.unlink()

        file_path = Path(
            self.simulation.fig_dir,
            f"frame{self.simulation.it:04d}.{self.config.plotting.fig_format}",
        )
        self.figure.savefig(file_path, format=self.config.plotting.fig_format)

    @staticmethod
    def show() -> None:
        try:
            plt.show(block=True)
        except Exception:
            pass


class Subplot:
    """Base class for all subplots."""

    def update(self, is_final: bool = False) -> None:
        raise NotImplementedError


class GeometrySubplot(Subplot):
    """A subplot containing the geometry of the body as well as the free surface
    and pressure profiles.

    """

    def __init__(self, pos: tuple[float, float, float, float], *, parent: Figure):
        self._parent = parent
        self._ax = parent.figure.add_axes(pos)

        self._init_handles()
        self._init_axes()

    def _init_handles(self):
        """Initialize all line handles."""
        (self._handle_nodes,) = self._ax.plot([], [], "ko")
        (self._handle_fs,) = self._ax.plot([], [], "b-")
        (self._handle_fs_init,) = self._ax.plot([], [], "b--")

        # Line handles for element initial and current positions
        self._handles_elements_init = {}
        self._handles_elements = {}

        # Line handles for the air and fluid pressure profiles
        self._handles_air_pressure = {}
        self._handles_hydro_pressure = {}
        for struct in self._parent.solid.substructures:
            (self._handles_air_pressure[struct],) = self._ax.plot([], [], "g-")
            (self._handles_hydro_pressure[struct],) = self._ax.plot([], [], "r-")
            for el in struct.elements:
                (self._handles_elements_init[el],) = self._ax.plot([], [], "k--")
                (self._handles_elements[el],) = self._ax.plot([], [], "k-", linewidth=2)

        self.lineCofR: list[CofRPlot] = []
        for body in self._parent.solid.rigid_bodies:
            # Initial position
            CofRPlot(
                self._ax,
                body,
                grid_len=self._parent.config.plotting.CofR_grid_len,
                cr_style="k--",
                cg_marker="ks",
                fill=False,
            )
            # Current position (will get updated)
            self.lineCofR.append(
                CofRPlot(
                    self._ax,
                    body,
                    grid_len=self._parent.config.plotting.CofR_grid_len,
                    cr_style="k-",
                    cg_marker="ko",
                )
            )

        self._handle_status_text = self._ax.text(
            0.05, 0.95, "", ha="left", va="top", transform=self._ax.transAxes
        )

    def _init_axes(self) -> None:
        """Initialize the axes with the correct limits and scaling."""
        # Find max/min of all nodal coordinates
        x = [nd.x for nd in self.solid.nodes]
        y = [nd.y for nd in self.solid.nodes]
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)

        # Handle explicit setting of max/min dimensions
        if self.config.plotting.xmin is not None:
            x_min = self.config.plotting.xmin
            self.config.plotting.ext_w = 0.0
        if self.config.plotting.xmax is not None:
            x_max = self.config.plotting.xmax
            self.config.plotting.ext_e = 0.0
        if self.config.plotting.ymin is not None:
            y_min = self.config.plotting.ymin
            self.config.plotting.ext_s = 0.0
        if self.config.plotting.ymax is not None:
            y_max = self.config.plotting.ymax
            self.config.plotting.ext_n = 0.0

        # Set the limits & axis labels
        self._ax.set_xlim(
            x_min - (x_max - x_min) * self.config.plotting.ext_w,
            x_max + (x_max - x_min) * self.config.plotting.ext_e,
        )
        self._ax.set_ylim(
            y_min - (y_max - y_min) * self.config.plotting.ext_s,
            y_max + (y_max - y_min) * self.config.plotting.ext_n,
        )
        self._ax.set_aspect("equal")

        self._ax.set_xlabel(r"$x$ [m]")
        self._ax.set_ylabel(r"$y$ [m]")

    @property
    def simulation(self) -> Simulation:
        return self._parent.simulation

    @property
    def solid(self) -> StructuralSolver:
        return self.simulation.structural_solver

    @property
    def fluid(self) -> PotentialPlaningSolver:
        return self.simulation.fluid_solver

    @property
    def config(self) -> Config:
        return self._parent.config

    def _draw_free_surface(self) -> None:
        """Draw the actual and undisturbed free-surface lines."""
        self._handle_fs.set_data(self.fluid.x_coord_fs, self.fluid.z_coord_fs)
        end_pts = np.array([self.config.plotting.x_fs_min, self.config.plotting.x_fs_max])
        self._handle_fs_init.set_data(
            end_pts, self.config.flow.waterline_height * np.ones_like(end_pts)
        )

    @staticmethod
    def _get_pressure_plot_points(ss: Substructure, s0: np.ndarray, p0: np.ndarray) -> np.ndarray:
        """Get coordinates required to plot pressure profile as lines.

        Args:
            ss: The substructure, from which to load coordinates and normal vectors.
            s0: An array of arclengths.
            p0: An array of pressures at the arclength points.

        Returns:
            An array containing x & y coordinates.

        """
        ind = np.abs(p0) > 1e-4
        s0 = s0[ind]
        p0 = p0[ind]

        if s0.size == 0:
            return np.empty((2, 0))

        normal_vec = np.array([ss.get_normal_vector(s) for s in s0])
        coords0 = np.array([ss.get_coordinates(s) for s in s0])
        coords1 = coords0 + ss.config.plotting.pressure_scale * p0[:, np.newaxis] * normal_vec

        # We start with an array of NaN, and then slice in the start & end points for each line
        data = np.full((s0.size * 3, 2), fill_value=np.nan)
        data[::3, :] = coords0
        data[1::3, :] = coords1
        return data.T

    def _draw_pressure_profiles(self, ss: Substructure) -> None:
        """Plot the internal and external pressure profiles as lines."""
        if handle := self._handles_hydro_pressure.get(ss):
            handle.set_data(self._get_pressure_plot_points(ss, ss.s_hydro, ss.p_hydro))
        if handle := self._handles_air_pressure.get(ss):
            handle.set_data(self._get_pressure_plot_points(ss, ss.s_air, -ss.p_air))

    def _draw_substructure(self, ss: Substructure) -> None:
        """Plot the substructure elements and pressure profiles."""
        for el in ss.elements:
            if handle := self._handles_elements.get(el):
                coords = np.array([nd.coordinates for nd in el.nodes])
                handle.set_data(coords.T)

            if handle := self._handles_elements_init.get(el):
                base_pt = np.array([el.parent.rigid_body.x_cr_init, el.parent.rigid_body.y_cr_init])
                pos = np.array(
                    [
                        trig.rotate_point(pos, base_pt, el.parent.rigid_body.trim)
                        - np.array([0, el.parent.rigid_body.draft])
                        for pos in el._initial_coordinates
                    ]
                )
                handle.set_data(pos.T)

        self._draw_pressure_profiles(ss)

    def _draw_structures(self):
        """Plot the structure."""
        for body in self.solid.rigid_bodies:
            for struct in body.substructures:
                self._draw_substructure(struct)

    def update(self, is_final: bool = False) -> None:
        self._draw_structures()
        self._draw_free_surface()
        self._handle_status_text.set_text(
            "\n".join(
                [
                    f"Iteration {self.simulation.it}",
                    f"$Fr={self.config.flow.froude_num:>8.3f}$",
                    f"$\\bar{{P_c}}={self.config.body.PcBar:>8.3f}$",
                ]
            )
        )
        for line in self.lineCofR:
            line.update()


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
        elif is_final:
            self.line_handle.set_marker("*")
            self.line_handle.set_markerfacecolor("y")
            self.line_handle.set_markersize(10)


class TimeHistorySubplot(Subplot):
    """A collection of axes (subplot) for displaying iteration series'.

    Args:
        pos: The relative position of the axes in the figure.
        parent: A reference to the parent FSIFigure.
        name: A name for the subplot, used when exporting plot data to file.

    """

    _is_ylog = False

    def __init__(
        self, pos: tuple[float, float, float, float], *, name: str = "default", parent: Figure
    ) -> None:
        self._parent = parent
        self._name = name
        self._ax: list[Axes] = []
        self._add_axes(parent.figure.add_axes(pos))
        self._series = list(self.create_series())

    def _add_axes(self, ax: Axes) -> None:
        """Add a set of child axes."""
        self._ax.append(ax)

    def _add_y_axes(self) -> None:
        """Add a twin y-axis, which shares an x-axis but has a different y-axis scale."""
        self._add_axes(plt.twinx(self._ax[0]))

    def create_series(self) -> Iterator[Series]:
        """A generator that must yield at least one series to plot on the axes."""
        raise NotImplementedError

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

        for series in self._series:
            point_data = np.array(series._points)
            x, y = point_data[~np.isnan(point_data[:, 1])].T

            if x.size == 0:
                continue

            x_min = min(x_min, np.min(x))
            x_max = max(x_max, np.max(x))

            y_min_tmp = np.min(y)
            y_max_tmp = np.max(y)

            # If the y-scale is log, we can only have positive limits
            # And we need to handle the case where y_min/max begins at Nan,
            # where min/max functions don't work
            if not self._is_ylog or y_min_tmp > 0:
                if np.isnan(y_min):
                    y_min = y_min_tmp
                else:
                    y_min = min(y_min, y_min_tmp)

            if not self._is_ylog or y_max_tmp > 0:
                if np.isnan(y_max):
                    y_max = y_max_tmp
                else:
                    y_max = max(y_max, y_max_tmp)

        # This prevents a warning for the first iteration where y_min == y_max
        if y_min == y_max:
            y_min, y_max = y_max - 0.1, y_max + 0.1

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


class MotionSubplot(TimeHistorySubplot):
    def __init__(self, pos: tuple[float, float, float, float], *, body: RigidBody, parent: Figure):
        super().__init__(pos, name=body.name, parent=parent)
        self._body = body
        self.set_properties(
            title=r"Motion History: {0}".format(body.name),
            xlabel=r"Iteration",
            ylabel=r"$d$ [m]",
        )
        self.set_properties(1, ylabel=r"$\theta$ [deg]")
        self.create_legend()

    def create_series(self) -> Iterator[Series]:
        yield Series(
            lambda: self._parent.simulation.it,
            lambda: self._body.draft,
            ax=self._ax[0],
            style="b-",
            label="Draft",
        )

        self._add_y_axes()
        yield Series(
            lambda: self._parent.simulation.it,
            lambda: self._body.trim,
            ax=self._ax[1],
            style="r-",
            label="Trim",
        )


class ForceSubplot(TimeHistorySubplot):
    def __init__(self, pos: tuple[float, float, float, float], *, body: RigidBody, parent: Figure):
        super().__init__(pos, name=body.name, parent=parent)
        self._body = body
        self.set_properties(
            title=f"Force & Moment History: {body.name}",
            ylabel=r"$\mathcal{D}/W$, $\mathcal{L}/W$, $\mathcal{M}/WL_c$",
        )
        self.create_legend()

    def create_series(self) -> Iterator[Series]:

        yield Series(
            lambda: self._parent.simulation.it,
            lambda: self._body.loads.L / self._body.weight,
            style="r-",
            label="Lift",
            ignore_first=True,
        )
        yield Series(
            lambda: self._parent.simulation.it,
            lambda: self._body.loads.D / self._body.weight,
            style="b-",
            label="Drag",
            ignore_first=True,
        )
        yield Series(
            lambda: self._parent.simulation.it,
            lambda: self._body.loads.M
            / (self._body.weight * self._parent.config.body.reference_length),
            style="g-",
            label="Moment",
            ignore_first=True,
        )


class ResidualSubplot(TimeHistorySubplot):
    _is_ylog = True

    def __init__(self, pos: tuple[float, float, float, float], *, parent: Figure):
        super().__init__(pos, name="residuals", parent=parent)
        self.set_properties(title="Residual History", xlabel="Iteration", yscale="log")
        self.create_legend()

    def create_series(self) -> Iterator[Series]:
        col = ["r", "b", "g"]
        for body, col_i in zip(self._parent.solid.rigid_bodies, col):
            yield Series(
                lambda: self._parent.simulation.it,
                body.get_res_lift,
                style=f"{col_i}-",
                label=f"Lift: {body.name}",
                ignore_first=True,
            )
            yield Series(
                lambda: self._parent.simulation.it,
                body.get_res_moment,
                style=f"{col_i}--",
                label=f"Moment: {body.name}",
                ignore_first=True,
            )

        yield Series(
            lambda: self._parent.simulation.it,
            lambda: np.abs(self._parent.solid.residual),
            style="k-",
            label="Total",
            ignore_first=True,
        )


class CofRPlot:
    """A set of lines and symbols to represent a body center of gravity and rotation.

    The center of rotation is drawn as two intersecting lines, center of gravity is a black marker.

    Args:
        ax: The axes to which this belongs.
        cr_style: The style to use for the lines.
        grid_len: The length to use for the grid lines.
        cg_marker: The center of gravity marker shape.
        fill: If True, fill the symbols with black.

    """

    def __init__(
        self,
        ax: Axes,
        body: RigidBody,
        *,
        cr_style: str = "k-",
        fill: bool = True,
        cg_marker: str = "o",
        grid_len: float = 0.5,
    ):
        self._ax = ax
        self._body = body
        self._grid_len = grid_len

        (self._cr_handle,) = self._ax.plot([], [], cr_style)
        (self._cg_handle,) = self._ax.plot(
            [], [], cg_marker, markersize=8, markerfacecolor="w" if not fill else "k"
        )
        self.update()

    def update(self) -> None:
        """Update the position and orientation of the handles."""
        c = np.array([self._body.x_cr, self._body.y_cr])
        hvec = trig.rotate_vec_2d(np.array([0.5 * self._grid_len, 0.0]), self._body.trim)
        vvec = trig.rotate_vec_2d(np.array([0.0, 0.5 * self._grid_len]), self._body.trim)
        pts = np.array([c - hvec, c + hvec, c, c - vvec, c + vvec])
        self._cr_handle.set_data(pts.T)
        self._cg_handle.set_data(self._body.x_cg, self._body.y_cg)


def plot_pressure(solver: PotentialPlaningSolver, fig_format: str = "png") -> None:
    """Create a plot of the pressure and shear stress profiles."""
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 5.0))

    for el in solver.pressure_elements:
        ax.plot(*el.plot_coords, color=el.plot_color, linestyle="-")
        ax.plot(el.x_coord * np.ones(2), [0.0, el.pressure], color=el.plot_color, linestyle="--")

    ax.plot(solver.x_coord, solver.pressure, "k-")

    ax.set_xlabel(r"$x\,\mathrm{[m]}$")
    ax.set_ylabel(r"$p\,\mathrm{[kPa]}$")
    ax.set_ylim(ymin=0.0)

    fig.savefig(f"pressureElements.{fig_format}", format=fig_format)
