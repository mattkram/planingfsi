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
from planingfsi.config import Config

if TYPE_CHECKING:
    from planingfsi.fe.rigid_body import RigidBody
    from planingfsi.fe.structure import StructuralSolver
    from planingfsi.fe.substructure import Substructure
    from planingfsi.potentialflow.solver import PotentialPlaningSolver
    from planingfsi.simulation import Simulation


class FSIFigure:
    def __init__(self, simulation: Simulation, config: Config):
        # TODO: We can get the config via the simulation in a property
        self.config = config
        self.simulation = simulation

        fig = plt.figure(figsize=(16, 12))
        if config.plotting.watch:
            plt.ion()
        self.geometry_ax = fig.add_axes([0.05, 0.6, 0.9, 0.35])

        (self.line_nodes,) = plt.plot([], [], "ko")

        (self.lineFS,) = plt.plot([], [], "b-")
        (self.lineFSi,) = plt.plot([], [], "b--")

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

        if config.plotting.xmin is not None:
            xMin = config.plotting.xmin
            config.plotting.ext_w = 0.0

        if config.plotting.xmax is not None:
            xMax = config.plotting.xmax
            config.plotting.ext_e = 0.0

        if config.plotting.ymin is not None:
            yMin = config.plotting.ymin
            config.plotting.ext_s = 0.0

        if config.plotting.ymax is not None:
            yMax = config.plotting.ymax
            config.plotting.ext_n = 0.0

        plt.xlabel(r"$x$ [m]", fontsize=22)
        plt.ylabel(r"$y$ [m]", fontsize=22)

        plt.xlim(
            [
                xMin - (xMax - xMin) * config.plotting.ext_w,
                xMax + (xMax - xMin) * config.plotting.ext_e,
            ]
        )
        plt.ylim(
            [
                yMin - (yMax - yMin) * config.plotting.ext_s,
                yMax + (yMax - yMin) * config.plotting.ext_n,
            ]
        )
        plt.gca().set_aspect("equal")
        self.TXT = plt.text(0.05, 0.95, "", ha="left", va="top", transform=plt.gca().transAxes)

        self.subplot: list["TimeHistory"] = []

        if self.solid.rigid_bodies:
            body = self.solid.rigid_bodies[0]
            self.subplot.append(ForceSubplot([0.70, 0.30, 0.25, 0.2], body, parent=self))
            self.subplot.append(MotionSubplot([0.70, 0.05, 0.25, 0.2], body, parent=self))

        if len(self.solid.rigid_bodies) > 1:
            body = self.solid.rigid_bodies[1]
            self.subplot.append(ForceSubplot([0.05, 0.30, 0.25, 0.2], body, parent=self))
            self.subplot.append(MotionSubplot([0.05, 0.05, 0.25, 0.2], body, parent=self))

        self.subplot.append(ResidualSubplot([0.40, 0.05, 0.25, 0.45], self.solid, parent=self))

    @property
    def solid(self) -> "StructuralSolver":
        return self.simulation.structural_solver

    @property
    def fluid(self) -> "PotentialPlaningSolver":
        return self.simulation.fluid_solver

    def plot_free_surface(self) -> None:
        """Create a plot of the free surface profile."""
        self.lineFS.set_data(self.fluid.x_coord_fs, self.fluid.z_coord_fs)
        end_pts = np.array([self.config.plotting.x_fs_min, self.config.plotting.x_fs_max])
        self.lineFSi.set_data(end_pts, self.config.flow.waterline_height * np.ones_like(end_pts))

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

    def plot_pressure_profiles(self, ss: Substructure) -> None:
        """Plot the internal and external pressure profiles as lines."""
        if handle := self.line_fluid_pressure.get(ss):
            handle.set_data(self._get_pressure_plot_points(ss, ss.s_hydro, ss.p_hydro))
        if handle := self.line_air_pressure.get(ss):
            handle.set_data(self._get_pressure_plot_points(ss, ss.s_air, -ss.p_air))

    def plot_substructure(self, ss: Substructure) -> None:
        """Plot the substructure elements and pressure profiles."""
        for el in ss.elements:
            if handle := self.element_handles.get(el):
                handle.set_data([nd.x for nd in el.nodes], [nd.y for nd in el.nodes])

            if handle := self.element_handles_0.get(el):
                base_pt = np.array([el.parent.parent.xCofR0, el.parent.parent.yCofR0])
                pos = [
                    trig.rotate_point(pos, base_pt, el.parent.parent.trim)
                    - np.array([0, el.parent.parent.draft])
                    for pos in el._initial_coordinates
                ]
                x, y = list(zip(*[[posi[i] for i in range(2)] for posi in pos]))
                handle.set_data(x, y)

        #    for nd in [self.node[0],self.node[-1]]:
        #      nd.plot()
        self.plot_pressure_profiles(ss)

    def plot_structure(self):
        """Plot the structure."""
        for body in self.solid.rigid_bodies:
            for struct in body.substructures:
                self.plot_substructure(struct)

    def update(self) -> None:
        self.plot_structure()
        self.plot_free_surface()
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
            if isinstance(s, TimeHistory):
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


class PlotSeries:
    def __init__(
        self, x_func: Callable[[], float], y_func: Callable[[], float], **kwargs: Any
    ) -> None:
        self.x: list[float] = []
        self.y: list[float] = []
        self.additionalSeries: list["PlotSeries"] = []

        self.get_x = x_func
        self.get_y = y_func

        self.series_type = kwargs.get("type", "point")
        self.ax = kwargs.get("ax", plt.gca())
        self.legEnt = kwargs.get("legEnt", None)
        self.ignore_first = kwargs.get("ignoreFirst", False)

        (self.line,) = self.ax.plot([], [], kwargs.get("sty", "k-"))

        if self.series_type == "history+curr":
            self.additionalSeries.append(
                PlotSeries(
                    x_func,
                    y_func,
                    ax=self.ax,
                    type="point",
                    sty=kwargs.get("currSty", "ro"),
                )
            )

    def update(self, final: bool = False) -> None:
        if self.series_type == "point":
            self.x = [self.get_x()]
            self.y = [self.get_y()]
            if final:
                self.line.set_marker("*")
                self.line.set_markerfacecolor("y")
                self.line.set_markersize(10)
        else:
            if self.ignore_first and self.get_x() == 0:
                self.x.append(np.nan)
                self.y.append(np.nan)
            else:
                self.x.append(self.get_x())
                self.y.append(self.get_y())

        self.line.set_data(self.x, self.y)

        for s in self.additionalSeries:
            s.update(final)


class TimeHistory:
    def __init__(self, pos: list[float], name: str = "default", *, parent: FSIFigure) -> None:
        self.parent = parent
        self.series: list["PlotSeries"] = []
        self.ax: list[Axes] = []
        self.title = ""
        self.xlabel = ""
        self.ylabel = ""

        self.name = name

        self.add_axes(plt.axes(pos))

    def add_axes(self, ax: Axes) -> None:
        self.ax.append(ax)

    def add_y_axes(self) -> None:
        self.add_axes(plt.twinx(self.ax[0]))

    def add_series(self, series: "PlotSeries") -> "PlotSeries":
        self.series.append(series)
        return series

    def set_properties(self, ax_ind: int = 0, **kwargs: Any) -> None:
        plt.setp(self.ax[ax_ind], **kwargs)

    def create_legend(self, ax_ind: int = 0) -> None:
        line, name = list(zip(*[(s.line, s.legEnt) for s in self.series if s.legEnt is not None]))
        self.ax[ax_ind].legend(line, name, loc="lower left")

    def update(self, final: bool = False) -> None:
        """Update the figure."""
        for s in self.series:
            s.update(final)

        for ax in self.ax:
            xMin, xMax = 0.0, 5  # config.it + 5
            yMin = np.nan
            for i, l in enumerate(ax.get_lines()):
                y = l.get_data()[1]
                if len(np.shape(y)) == 0:
                    y = np.array([y])

                try:
                    y = y[np.ix_(~np.isnan(y))]
                except TypeError:
                    y = []

                if not np.shape(y)[0] == 0:
                    yMinL = np.min(y) - 0.02
                    yMaxL = np.max(y) + 0.02
                    if i == 0 or np.isnan(yMin):
                        yMin, yMax = yMinL, yMaxL
                    else:
                        yMin = np.min([yMin, yMinL])
                        yMax = np.max([yMax, yMaxL])

            ax.set_xlim([xMin, xMax])
            if ~np.isnan(yMin) and ~np.isnan(yMax):
                ax.set_ylim([yMin, yMax])

    def write(self) -> None:
        with Path(f"{self.name}_timeHistories.txt").open("w") as ff:
            x = self.series[0].x
            y = list(zip(*[s.y for s in self.series]))

            for xi, yi in zip(x, y):
                ff.write("{0:4.0f}".format(xi))
                for yii in yi:
                    ff.write(" {0:8.6e}".format(yii))
                ff.write("\n")


class MotionSubplot(TimeHistory):
    def __init__(self, pos: list[float], body: RigidBody, parent: FSIFigure):
        TimeHistory.__init__(self, pos, body.name, parent=parent)

        def itFunc() -> float:
            return 0.0  # config.it

        def drftFunc() -> float:
            return body.draft

        def trimFunc() -> float:
            return body.trim

        self.add_y_axes()

        # Add plot series to appropriate axis
        self.add_series(
            PlotSeries(
                itFunc,
                drftFunc,
                ax=self.ax[0],
                sty="b-",
                type="history+curr",
                legEnt="Draft",
            )
        )
        self.add_series(
            PlotSeries(
                itFunc,
                trimFunc,
                ax=self.ax[1],
                sty="r-",
                type="history+curr",
                legEnt="Trim",
            )
        )

        self.set_properties(
            title=r"Motion History: {0}".format(body.name),
            xlabel=r"Iteration",
            ylabel=r"$d$ [m]",
        )
        self.set_properties(1, ylabel=r"$\theta$ [deg]")
        self.create_legend()


class ForceSubplot(TimeHistory):
    def __init__(self, pos: list[float], body: RigidBody, parent: FSIFigure):
        TimeHistory.__init__(self, pos, body.name, parent=parent)

        def itFunc() -> float:
            return 0.0  # config.it

        def liftFunc() -> float:
            return body.L / body.weight

        def dragFunc() -> float:
            return body.D / body.weight

        def momFunc() -> float:
            return body.M / (body.weight * self.parent.config.body.reference_length)

        self.add_series(
            PlotSeries(
                itFunc,
                liftFunc,
                sty="r-",
                type="history+curr",
                legEnt="Lift",
                ignoreFirst=True,
            )
        )
        self.add_series(
            PlotSeries(
                itFunc,
                dragFunc,
                sty="b-",
                type="history+curr",
                legEnt="Drag",
                ignoreFirst=True,
            )
        )
        self.add_series(
            PlotSeries(
                itFunc,
                momFunc,
                sty="g-",
                type="history+curr",
                legEnt="Moment",
                ignoreFirst=True,
            )
        )

        self.set_properties(
            title=r"Force & Moment History: {0}".format(body.name),
            #                       xlabel=r'Iteration', \
            ylabel=r"$\mathcal{D}/W$, $\mathcal{L}/W$, $\mathcal{M}/WL_c$",
        )
        self.create_legend()


class ResidualSubplot(TimeHistory):
    def __init__(self, pos: list[float], solid: "StructuralSolver", parent: FSIFigure):
        TimeHistory.__init__(self, pos, "residuals", parent=parent)

        def itFunc() -> float:
            return 0.0  # config.it

        col = ["r", "b", "g"]
        for bd, coli in zip(solid.rigid_bodies, col):
            self.add_series(
                PlotSeries(
                    itFunc,
                    bd.get_res_lift,
                    sty="{0}-".format(coli),
                    type="history+curr",
                    legEnt="ResL: {0}".format(bd.name),
                    ignoreFirst=True,
                )
            )
            self.add_series(
                PlotSeries(
                    itFunc,
                    bd.get_res_moment,
                    sty="{0}--".format(coli),
                    type="history+curr",
                    legEnt="ResM: {0}".format(bd.name),
                    ignoreFirst=True,
                )
            )

        self.add_series(
            PlotSeries(
                itFunc,
                lambda: np.abs(solid.residual),
                sty="k-",
                type="history+curr",
                legEnt="Total",
                ignoreFirst=True,
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
        c = np.array([self.body.xCofR, self.body.yCofR])
        hvec = trig.rotate_vec_2d(np.array([0.5 * self._grid_len, 0.0]), self.body.trim)
        vvec = trig.rotate_vec_2d(np.array([0.0, 0.5 * self._grid_len]), self.body.trim)
        pts = np.array([c - hvec, c + hvec, c, c - vvec, c + vvec])
        self.line.set_data(pts.T)
        self.lineCofG.set_data(self.body.xCofG, self.body.yCofG)


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
