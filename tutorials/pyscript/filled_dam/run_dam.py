from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from planingfsi import FlexibleMembraneSubstructure
from planingfsi import Mesh
from planingfsi import Simulation
from planingfsi.figure import Figure
from planingfsi.figure import GeometrySubplot
from planingfsi.figure import Series
from planingfsi.figure import Subplot
from planingfsi.figure import TimeHistorySubplot


class MyGeometrySubplot(GeometrySubplot):
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

        self.lineCofR = []

    def update(self, is_final: bool = False) -> None:
        self._draw_structures()
        self._draw_free_surface()


class MyResidualSubplot(TimeHistorySubplot):
    _is_ylog = True

    def __init__(self, pos: tuple[float, float, float, float], *, parent: Figure):
        super().__init__(pos, name="residuals", parent=parent)
        self.set_properties(title="Residual History", xlabel="Iteration", yscale="log")

    def create_series(self) -> Iterator[Series]:
        yield Series(
            lambda: self._parent.simulation.it,
            lambda: np.abs(self._parent.solid.residual),
            style="k-",
            ignore_first=True,
        )


class MyFigure(Figure):
    def create_subplots(self) -> Iterator[Subplot]:
        yield MyGeometrySubplot((0.1, 0.55, 0.85, 0.4), parent=self)
        yield MyResidualSubplot((0.1, 0.05, 0.85, 0.4), parent=self)


def run_dam_case():
    mesh = Mesh()

    # Create points (ID, type, params)
    mesh.add_point(1, "rel", [0, 0, 1.0])
    mesh.add_point(2, "rel", [0, 180, 1.0])

    dam = mesh.add_submesh("dam")
    dam.add_curve(1, 2, Nel=50, arcLen=np.pi)

    simulation = Simulation()

    # Set some global configuration values
    simulation.config.flow.froude_num = 1.0
    simulation.config.flow.waterline_height = 1.0
    simulation.config.plotting._pressure_scale_pct = 1e-8
    simulation.config.plotting.show = True  # Need this so the plot is updated for residual
    simulation.config.solver.max_it = 200

    body = simulation.add_rigid_body()
    dam = FlexibleMembraneSubstructure(
        name="dam",
        seal_pressure_method="hydrostatic",
    )
    body.add_substructure(dam)

    simulation.load_mesh(mesh)

    simulation._figure = MyFigure(simulation=simulation, figsize=(8, 8))
    simulation.run()

    return simulation


def get_figure(simulation):
    simulation.figure.update()
    return simulation.figure.figure


if __name__ == "__main__":
    sim = run_dam_case()
    figure = get_figure(simulation=sim)
    figure.show()
