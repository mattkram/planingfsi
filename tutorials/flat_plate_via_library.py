from planingfsi import Mesh
from planingfsi.fe.substructure import RigidSubstructure
from planingfsi.fsi.simulation import Simulation
from planingfsi.potentialflow.pressurepatch import PlaningSurface


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

    # TODO: We currently require writing to mesh files, because the structure is loaded from files
    mesh.write("mesh")

    simulation = Simulation()

    # Set some global configuration values
    simulation.config.flow.froude_num = froude_num
    simulation.config.plotting.show = True
    simulation.config.plotting._pressure_scale_pct = 1e-8

    # TODO: This should happen implicitly, not during load
    # Add the default rigid body
    body = simulation.add_rigid_body()

    substructure = body.add_substructure(RigidSubstructure(name="plate"))
    planing_surface = simulation.fluid_solver.add_planing_surface(
        PlaningSurface(
            name="plate",
            initial_length=0.48,
            minimum_length=0.01,
            num_fluid_elements=40,
            point_spacing="cosine",
        )
    )
    substructure.add_planing_surface(planing_surface)

    simulation.run()


if __name__ == "__main__":
    main()
