import os
from pathlib import Path
from typing import List, Callable, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .. import config, trig
from ..fe import rigid_body
from ..fe.structure import StructuralSolver
from ..fsi import simulation as fsi_simulation
from ..potentialflow.solver import PotentialPlaningSolver


class FSIFigure:
    def __init__(self, simulation: "fsi_simulation.Simulation"):
        self.simulation = simulation

        fig = plt.figure(figsize=(16, 12))
        if config.plotting.watch:
            plt.ion()
        self.geometry_ax = fig.add_axes([0.05, 0.6, 0.9, 0.35])

        for nd in self.solid.node:
            (nd.line_xy,) = plt.plot([], [], "ro")

        (self.lineFS,) = plt.plot([], [], "b-")
        (self.lineFSi,) = plt.plot([], [], "b--")

        for struct in self.solid.substructure:
            (struct.line_air_pressure,) = plt.plot([], [], "g-")
            (struct.line_fluid_pressure,) = plt.plot([], [], "r-")
            for el in struct.el:
                (el.lineEl0,) = plt.plot([], [], "k--")
                (el.lineEl,) = plt.plot([], [], "k-", linewidth=2)

        self.lineCofR: List["CofRPlot"] = []
        for bodies in self.solid.rigid_body:
            CofRPlot(self.geometry_ax, bodies, symbol=False, style="k--", marker="s", fill=False)
            self.lineCofR.append(
                CofRPlot(self.geometry_ax, bodies, symbol=False, style="k-", marker="o")
            )

        x = [nd.x for struct in self.solid.substructure for nd in struct.node]
        y = [nd.y for struct in self.solid.substructure for nd in struct.node]
        xMin, xMax = min(x), max(x)
        yMin, yMax = min(y), max(y)

        try:
            xMin = config.plotting.xmin
            config.plotting.ext_w = 0.0
        except AttributeError:
            pass
        try:
            xMax = config.plotting.xmax
            config.plotting.ext_e = 0.0
        except AttributeError:
            pass
        try:
            yMin = config.plotting.ymin
            config.plotting.ext_s = 0.0
        except AttributeError:
            pass
        try:
            yMax = config.plotting.ymax
            config.plotting.ext_n = 0.0
        except AttributeError:
            pass

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

        self.subplot: List["TimeHistory"] = []

        if self.solid.rigid_body:
            body = self.solid.rigid_body[0]
            self.subplot.append(ForceSubplot([0.70, 0.30, 0.25, 0.2], body))
            self.subplot.append(MotionSubplot([0.70, 0.05, 0.25, 0.2], body))

        if len(self.solid.rigid_body) > 1:
            body = self.solid.rigid_body[1]
            self.subplot.append(ForceSubplot([0.05, 0.30, 0.25, 0.2], body))
            self.subplot.append(MotionSubplot([0.05, 0.05, 0.25, 0.2], body))

        self.subplot.append(ResidualSubplot([0.40, 0.05, 0.25, 0.45], self.solid))

    @property
    def solid(self) -> StructuralSolver:
        return self.simulation.solid_solver

    @property
    def fluid(self) -> PotentialPlaningSolver:
        return self.simulation.fluid_solver

    def plot_free_surface(self) -> None:
        """Create a plot of the free surface profile."""
        self.lineFS.set_data(self.fluid.x_coord_fs, self.fluid.z_coord_fs)
        end_pts = np.array([config.plotting.x_fs_min, config.plotting.x_fs_max])
        self.lineFSi.set_data(end_pts, config.flow.waterline_height * np.ones_like(end_pts))

    def update(self) -> None:
        self.solid.plot()
        self.plot_free_surface()
        self.TXT.set_text(
            "\n".join(
                [
                    f"Iteration {self.simulation.it}",
                    f"$Fr={config.flow.froude_num:>8.3f}$",
                    f"$\\bar{{P_c}}={config.body.PcBar:>8.3f}$",
                ]
            )
        )

        # Update each lower subplot
        for s in self.subplot:
            s.update(
                self.solid.res < config.solver.max_residual
                and self.simulation.it > config.solver.num_ramp_it
            )

        for line in self.lineCofR:
            line.update()

        plt.draw()

        if config.plotting.save:
            self.save()

    def write_time_histories(self) -> None:
        for s in self.subplot:
            if isinstance(s, TimeHistory):
                s.write()

    def save(self) -> None:
        plt.savefig(
            os.path.join(
                config.path.fig_dir_name,
                "frame{1:04d}.{0}".format(config.plotting.fig_format, self.simulation.it),
            ),
            format=config.plotting.fig_format,
        )  # , dpi=300)

    def show(self) -> None:
        plt.show(block=True)


class PlotSeries:
    def __init__(
        self, x_func: Callable[[], float], y_func: Callable[[], float], **kwargs: Any
    ) -> None:
        self.x: List[float] = []
        self.y: List[float] = []
        self.additionalSeries: List["PlotSeries"] = []

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
    def __init__(self, pos: List[float], name: str = "default") -> None:
        self.series: List["PlotSeries"] = []
        self.ax: List[Axes] = []
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
    def __init__(self, pos: List[float], body: "rigid_body.RigidBody"):
        TimeHistory.__init__(self, pos, body.name)

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
    def __init__(self, pos: List[float], body: "rigid_body.RigidBody"):
        TimeHistory.__init__(self, pos, body.name)

        def itFunc() -> float:
            return 0.0  # config.it

        def liftFunc() -> float:
            return body.L / body.weight

        def dragFunc() -> float:
            return body.D / body.weight

        def momFunc() -> float:
            return body.M / (body.weight * config.body.reference_length)

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
    def __init__(self, pos: List[float], solid: "StructuralSolver"):
        TimeHistory.__init__(self, pos, "residuals")

        def itFunc() -> float:
            return 0.0  # config.it

        col = ["r", "b", "g"]
        for bd, coli in zip(solid.rigid_body, col):
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
                lambda: np.abs(solid.res),
                sty="k-",
                type="history+curr",
                legEnt="Total",
                ignoreFirst=True,
            )
        )

        self.set_properties(title=r"Residual History", yscale="log")
        self.create_legend()


class CofRPlot:
    def __init__(self, ax: Axes, body: "rigid_body.RigidBody", **kwargs: Any):
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

        (self.line,) = self.ax.plot([], [], style)
        (self.lineCofG,) = self.ax.plot([], [], "k" + marker, markersize=8, markerfacecolor=fcol)
        self.update()

    def update(self) -> None:
        grid_len = config.plotting.CofR_grid_len
        c = np.array([self.body.xCofR, self.body.yCofR])
        hvec = trig.rotate_vec_2d(np.array([0.5 * grid_len, 0.0]), self.body.trim)
        vvec = trig.rotate_vec_2d(np.array([0.0, 0.5 * grid_len]), self.body.trim)
        pts = np.array([c - hvec, c + hvec, c, c - vvec, c + vvec])
        self.line.set_data(pts.T)
        self.lineCofG.set_data(self.body.xCofG, self.body.yCofG)


def plot_pressure(solver: PotentialPlaningSolver) -> None:
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

    fig.savefig(
        f"pressureElements.{config.plotting.fig_format}",
        format=config.plotting.fig_format,
    )
