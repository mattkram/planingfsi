import numpy as np

from planingfsi import FlexibleMembraneSubstructure
from planingfsi import Mesh
from planingfsi import Simulation


def generate_mesh() -> Mesh:
    mesh = Mesh()

    # Create points (ID, type, params)
    mesh.add_point(1, "rel", [0, 0, 1.0])
    mesh.add_point(2, "rel", [0, 180, 1.0])

    dam = mesh.add_submesh("dam")
    dam.add_curve(1, 2, Nel=50, arcLen=np.pi)

    return mesh


def main() -> None:
    mesh = generate_mesh()
    # mesh.plot(show=True)

    simulation = Simulation()

    # Set some global configuration values
    simulation.config.flow.froude_num = 1.0
    simulation.config.flow.waterline_height = 0.8
    simulation.config.plotting.show = True
    simulation.config.plotting._pressure_scale_pct = 1e-8
    simulation.config.solver.max_it = 200

    body = simulation.add_rigid_body()
    body.add_substructure(
        FlexibleMembraneSubstructure(
            name="dam",
            seal_pressure_method="hydrostatic",
        )
    )

    simulation.load_mesh(mesh)
    simulation.run()


if __name__ == "__main__":
    main()
