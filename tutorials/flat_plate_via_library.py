from planingfsi import Mesh
from planingfsi import PlaningSurface
from planingfsi import RigidSubstructure
from planingfsi import Simulation


def generate_mesh(angle_of_attack: float) -> Mesh:
    mesh = Mesh()

    # Create points (ID, type, params)
    mesh.add_point(1, "rel", [0, 180, 0.5])
    mesh.add_point(2, "rel", [0, 0, 1.0])

    mesh.rotate_points(0, angle_of_attack, [1, 2])

    mesh_fwd = mesh.add_submesh("plate")
    mesh_fwd.add_curve(1, 2, Nel=10)

    return mesh


def main() -> None:
    froude_num = 1.0
    mesh = generate_mesh(angle_of_attack=10.0)
    # mesh.plot(show=True)

    simulation = Simulation()

    # Set some global configuration values
    simulation.config.flow.froude_num = froude_num
    simulation.config.plotting.show = True
    simulation.config.plotting._pressure_scale_pct = 1e-8

    body = simulation.add_rigid_body()
    substructure = body.add_substructure(RigidSubstructure(name="plate"))
    substructure.add_planing_surface(
        PlaningSurface(
            name="plate",
            initial_length=0.48,
            minimum_length=0.01,
            num_fluid_elements=40,
            point_spacing="cosine",
        )
    )

    simulation.load_mesh(mesh)
    simulation.run()


if __name__ == "__main__":
    main()
