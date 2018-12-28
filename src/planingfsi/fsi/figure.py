import os

import numpy as np
import matplotlib.pyplot as plt

import planingfsi.config as config

# import planingfsi.krampy as kp


class FSIFigure:
    def __init__(self, solid, fluid):
        self.solid = solid
        self.fluid = fluid

        plt.figure(figsize=(16, 12))
        if config.plotting.watch:
            plt.ion()
        self.geometryAx = plt.axes([0.05, 0.6, 0.9, 0.35])

        for nd in self.solid.node:
            nd.line_x_yline_xy, = plt.plot([], [], "ro")

        self.fluid.lineFS, = plt.plot([], [], "b-")
        self.fluid.lineFSi, = plt.plot([], [], "b--")

        for struct in self.solid.substructure:
            struct.line_air_pressure, = plt.plot([], [], "g-")
            struct.line_fluid_pressure, = plt.plot([], [], "r-")
            for el in struct.el:
                el.lineEl0, = plt.plot([], [], "k--")
                el.lineEl, = plt.plot([], [], "k-", linewidth=2)

        self.lineCofR = []
        for bd in self.solid.rigid_body:
            CofRPlot(
                self.geometryAx, bd, symbol=False, style="k--", marker="s", fill=False
            )
            self.lineCofR.append(
                CofRPlot(self.geometryAx, bd, symbol=False, style="k-", marker="o")
            )

        xMin, xMax = kp.minMax(
            [nd.x for struct in self.solid.substructure for nd in struct.node]
        )
        yMin, yMax = kp.minMax(
            [nd.y for struct in self.solid.substructure for nd in struct.node]
        )

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
        self.TXT = plt.text(
            0.05, 0.95, "", ha="left", va="top", transform=plt.gca().transAxes
        )

        self.subplot = []

        bd = [bd for bd in self.solid.rigid_body]

        if len(bd) > 0:
            self.subplot.append(ForceSubplot([0.70, 0.30, 0.25, 0.2], bd[0]))
            self.subplot.append(MotionSubplot([0.70, 0.05, 0.25, 0.2], bd[0]))

        if len(bd) > 1:
            self.subplot.append(ForceSubplot([0.05, 0.30, 0.25, 0.2], bd[1]))
            self.subplot.append(MotionSubplot([0.05, 0.05, 0.25, 0.2], bd[1]))

        self.subplot.append(ResidualSubplot([0.40, 0.05, 0.25, 0.45], self.solid))

    def update(self):
        self.solid.plot()
        self.fluid.plot_free_surface()
        self.TXT.set_text(
            (
                r"Iteration {0}"
                + "\n"
                + r"$Fr={1:>8.3f}$"
                + "\n"
                + r"$\bar{{P_c}}={2:>8.3f}$"
            ).format(config.it, config.flow.froude_num, config.body.PcBar)
        )

        # Update each lower subplot
        for s in self.subplot:
            s.update(
                self.solid.res < config.solver.max_residual
                and config.it > config.solver.num_ramp_it
            )

        for l in self.lineCofR:
            l.update()

        plt.draw()

        if config.plotting.save:
            self.save()

    def writeTimeHistories(self):
        for s in self.subplot:
            if isinstance(s, TimeHistory):
                s.write()

    def save(self):
        plt.savefig(
            os.path.join(
                config.path.fig_dir_name,
                "frame{1:04d}.{0}".format(config.plotting.fig_format, config.it),
            ),
            format=config.plotting.fig_format,
        )  # , dpi=300)

    def show(self):
        plt.show(block=True)


class PlotSeries:
    def __init__(self, xFunc, yFunc, **kwargs):
        self.x = []
        self.y = []
        self.additionalSeries = []

        self.getX = xFunc
        self.getY = yFunc

        self.seriesType = kwargs.get("type", "point")
        self.ax = kwargs.get("ax", plt.gca())
        self.legEnt = kwargs.get("legEnt", None)
        self.ignoreFirst = kwargs.get("ignoreFirst", False)

        self.line, = self.ax.plot([], [], kwargs.get("sty", "k-"))

        if self.seriesType == "history+curr":
            self.additionalSeries.append(
                PlotSeries(
                    xFunc,
                    yFunc,
                    ax=self.ax,
                    type="point",
                    sty=kwargs.get("currSty", "ro"),
                )
            )

    def update(self, final=False):
        if self.seriesType == "point":
            self.x = self.getX()
            self.y = self.getY()
            if final:
                self.line.set_marker("*")
                self.line.set_markerfacecolor("y")
                self.line.set_markersize(10)
        else:
            if self.ignoreFirst and self.getX() == 0:
                self.x.append(np.nan)
                self.y.append(np.nan)
            else:
                self.x.append(self.getX())
                self.y.append(self.getY())

        self.line.set_data(self.x, self.y)

        for s in self.additionalSeries:
            s.update(final)


class TimeHistory:
    def __init__(self, pos, name="default"):
        self.series = []
        self.ax = []
        self.title = ""
        self.xlabel = ""
        self.ylabel = ""

        self.name = name

        self.addAxes(plt.axes(pos))

    def addAxes(self, ax):
        self.ax.append(ax)

    def addYAxes(self):
        self.addAxes(plt.twinx(self.ax[0]))

    def addSeries(self, series):
        self.series.append(series)
        return series

    def setProperties(self, axInd=0, **kwargs):
        plt.setp(self.ax[axInd], **kwargs)

    def createLegend(self, axInd=0):
        line, name = list(
            zip(*[(s.line, s.legEnt) for s in self.series if s.legEnt is not None])
        )
        self.ax[axInd].legend(line, name, loc="lower left")

    def update(self, final=False):
        for s in self.series:
            s.update(final)

        for ax in self.ax:
            xMin, xMax = 0.0, config.it + 5
            yMin = np.nan
            for i, l in enumerate(ax.get_lines()):
                y = l.get_data()[1]
                if len(np.shape(y)) == 0:
                    y = np.array([y])

                try:
                    y = y[np.ix_(~np.isnan(y))]
                except:
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

    def write(self):
        ff = open("{0}_timeHistories.txt".format(self.name), "w")
        x = self.series[0].x
        y = list(zip(*[s.y for s in self.series]))

        for xi, yi in zip(x, y):
            ff.write("{0:4.0f}".format(xi))
            for yii in yi:
                ff.write(" {0:8.6e}".format(yii))
            ff.write("\n")

        ff.close()


class MotionSubplot(TimeHistory):
    def __init__(self, pos, body):
        TimeHistory.__init__(self, pos, body.name)

        itFunc = lambda: config.it
        drftFunc = lambda: body.draft
        trimFunc = lambda: body.trim

        self.addYAxes()

        # Add plot series to appropriate axis
        self.addSeries(
            PlotSeries(
                itFunc,
                drftFunc,
                ax=self.ax[0],
                sty="b-",
                type="history+curr",
                legEnt="Draft",
            )
        )
        self.addSeries(
            PlotSeries(
                itFunc,
                trimFunc,
                ax=self.ax[1],
                sty="r-",
                type="history+curr",
                legEnt="Trim",
            )
        )

        self.setProperties(
            title=r"Motion History: {0}".format(body.name),
            xlabel=r"Iteration",
            ylabel=r"$d$ [m]",
        )
        self.setProperties(1, ylabel=r"$\theta$ [deg]")
        self.createLegend()


class ForceSubplot(TimeHistory):
    def __init__(self, pos, body):
        TimeHistory.__init__(self, pos, body.name)

        itFunc = lambda: config.it
        liftFunc = lambda: body.L / body.weight
        dragFunc = lambda: body.D / body.weight
        momFunc = lambda: body.M / (body.weight * config.body.reference_length)

        self.addSeries(
            PlotSeries(
                itFunc,
                liftFunc,
                sty="r-",
                type="history+curr",
                legEnt="Lift",
                ignoreFirst=True,
            )
        )
        self.addSeries(
            PlotSeries(
                itFunc,
                dragFunc,
                sty="b-",
                type="history+curr",
                legEnt="Drag",
                ignoreFirst=True,
            )
        )
        self.addSeries(
            PlotSeries(
                itFunc,
                momFunc,
                sty="g-",
                type="history+curr",
                legEnt="Moment",
                ignoreFirst=True,
            )
        )

        self.setProperties(
            title=r"Force & Moment History: {0}".format(body.name),
            #                       xlabel=r'Iteration', \
            ylabel=r"$\mathcal{D}/W$, $\mathcal{L}/W$, $\mathcal{M}/WL_c$",
        )
        self.createLegend()


class ResidualSubplot(TimeHistory):
    def __init__(self, pos, solid):
        TimeHistory.__init__(self, pos, "residuals")

        itFunc = lambda: config.it

        col = ["r", "b", "g"]
        for bd, coli in zip(solid.rigid_body, col):
            self.addSeries(
                PlotSeries(
                    itFunc,
                    bd.get_res_lift,
                    sty="{0}-".format(coli),
                    type="history+curr",
                    legEnt="ResL: {0}".format(bd.name),
                    ignoreFirst=True,
                )
            )
            self.addSeries(
                PlotSeries(
                    itFunc,
                    bd.get_res_moment,
                    sty="{0}--".format(coli),
                    type="history+curr",
                    legEnt="ResM: {0}".format(bd.name),
                    ignoreFirst=True,
                )
            )

        self.addSeries(
            PlotSeries(
                itFunc,
                lambda: np.abs(solid.res),
                sty="k-",
                type="history+curr",
                legEnt="Total",
                ignoreFirst=True,
            )
        )

        self.setProperties(title=r"Residual History", yscale="log")
        self.createLegend()


class CofRPlot:
    def __init__(self, ax, body, **kwargs):
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

        self.line, = self.ax.plot([], [], style)
        self.lineCofG, = self.ax.plot(
            [], [], "k" + marker, markersize=8, markerfacecolor=fcol
        )
        self.update()

    def update(self):
        l = config.plotting.CofR_grid_len
        c = np.array([self.body.xCofR, self.body.yCofR])
        hvec = kp.rotateVec(np.array([0.5 * l, 0.0]), self.body.trim)
        vvec = kp.rotateVec(np.array([0.0, 0.5 * l]), self.body.trim)
        pts = [c - hvec, c + hvec, np.ones(2) * np.nan, c - vvec, c + vvec]
        self.line.set_data(list(zip(*pts)))
        self.lineCofG.set_data(self.body.xCofG, self.body.yCofG)
