from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from matplotlib import pyplot as plt

from planingfsi import FlexibleMembraneSubstructure
from planingfsi import Mesh
from planingfsi import Simulation
from planingfsi.figure import Figure
from planingfsi.figure import GeometrySubplot
from planingfsi.figure import Subplot


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


class MyFigure(Figure):
    def create_subplots(self) -> Iterator[Subplot]:
        yield MyGeometrySubplot((0.05, 0.6, 0.9, 0.35), parent=self)


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
    simulation.config.solver.max_it = 100

    body = simulation.add_rigid_body()
    dam = FlexibleMembraneSubstructure(
        name="dam",
        seal_pressure_method="hydrostatic",
    )
    body.add_substructure(dam)

    simulation.load_mesh(mesh)
    simulation.run()

    simulation._figure = MyFigure(simulation=simulation, figsize=(8, 6))
    simulation.figure.update()
    return simulation.figure.figure


if __name__ == "__main__":
    run_dam_case()
    plt.show()
