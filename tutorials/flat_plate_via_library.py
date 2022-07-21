from planingfsi import Mesh
from planingfsi.fsi.interpolator import Interpolator
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
    mesh = generate_mesh(10.0)
    # mesh.plot(show=True)

    # TODO: We currently require writing to mesh files, because the structure is loaded from files
    mesh.write("mesh")

    simulation = Simulation()

    simulation.config.flow.froude_num = froude_num
    simulation.config.plotting.show = True
    simulation.config.plotting._pressure_scale_pct = 1e-8

    # TODO: This should happen implicitly, not during load
    # Add the default rigid body
    simulation.solid_solver.add_rigid_body()

    dict_ = {
        "substructureName": "plate",
        "substructureType": "rigid",
        "hasPlaningSurface": True,
        "Nfl": 40,
        "pointSpacing": "cosine",
        "kuttaPressure": 0.0,
        "minimumLength": 0.01,
        "structInterpType": "cubic",
        "structExtrap": True,
        "sSepPctStart": 0.5,
        "waterline_height": 0.0,
    }

    substructure = simulation.solid_solver.add_substructure(dict_)
    planing_surface = simulation.fluid_solver.add_planing_surface(
        PlaningSurface(dict_, name="plate", initial_length=0.48)
    )
    Interpolator(substructure, planing_surface, dict_)

    simulation.run()


if __name__ == "__main__":
    main()
