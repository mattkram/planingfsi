import tempfile
from pathlib import Path

from planingfsi import Mesh
from planingfsi import PlaningSurface
from planingfsi import PressureCushion
from planingfsi import RigidSubstructure
from planingfsi import Simulation


def generate_mesh() -> Mesh:
    angle_of_attack = 10.0
    immersed_length = 0.5
    cushion_length = 5.0
    cushion_height = 0.3

    mesh = Mesh()

    # Create points (ID, type, params)
    mesh.add_point(1, "rel", [0, 180, immersed_length])
    mesh.add_point(2, "rel", [0, 0, 1.0])
    mesh.add_point(3, "con", [1, "y", cushion_height], angle=angle_of_attack + 90)
    mesh.rotate_points(0, angle_of_attack, [1, 2])

    mesh.add_point(5, "rel", [0, 180, cushion_length])
    mesh.add_point(6, "rel", [5, 180, immersed_length])
    mesh.rotate_points(5, angle_of_attack, [6])

    mesh.add_point(4, "con", [6, "y", cushion_height], angle=angle_of_attack)

    mesh.rotate_all_points(0, 2)

    mesh_fwd = mesh.add_submesh("fwd_plate")
    mesh_fwd.add_curve(1, 2, Nel=10)

    cushion = mesh.add_submesh("wetdeck")
    cushion.add_curve(4, 3, Nel=10)
    cushion.add_curve(3, 1, Nel=10)

    mesh_aft = mesh.add_submesh("aft_plate")
    mesh_aft.add_curve(6, 4, Nel=10)
    # mesh.plot(show=True)

    return mesh


def main() -> None:
    froude_num = 0.75

    with tempfile.TemporaryDirectory() as tmpdir:
        simulation = Simulation()
        simulation.case_dir = Path(tmpdir)

        # Set some global configuration values
        simulation.config.solver.wetted_length_relax = 0.7
        simulation.config.flow.froude_num = froude_num
        simulation.config.plotting.show = True
        simulation.config.plotting._pressure_scale_pct = 2.5e-8
        simulation.config.plotting.xmin = -10
        simulation.config.plotting.ymin = -1
        simulation.config.plotting.ymax = 1
        simulation.config.plotting.growth_rate = 1.05

        body = simulation.add_rigid_body()
        fwd_substructure = body.add_substructure(RigidSubstructure(name="fwd_plate"))
        fwd_planing_surface = fwd_substructure.add_planing_surface(
            PlaningSurface(
                name="fwd_plate",
                initial_length=0.73,
                minimum_length=0.1,
                num_fluid_elements=40,
                point_spacing="cosine",
            )
        )

        aft_substructure = body.add_substructure(RigidSubstructure(name="aft_plate"))
        aft_planing_surface = aft_substructure.add_planing_surface(
            PlaningSurface(
                name="aft_plate",
                initial_length=1.0,
                minimum_length=0.1,
                num_fluid_elements=40,
                point_spacing="cosine",
            )
        )

        wetdeck = body.add_substructure(RigidSubstructure(name="wetdeck"))
        wetdeck.add_pressure_cushion(
            PressureCushion(
                name="cushion",
                cushion_pressure=1000.0,
                upstream_planing_surface=fwd_planing_surface,
                downstream_planing_surface=aft_planing_surface,
            )
        )

        mesh = generate_mesh()
        simulation.load_mesh(mesh)
        simulation.run()


if __name__ == "__main__":
    main()
