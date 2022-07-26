from planingfsi import Mesh
from planingfsi.fe.substructure import RigidSubstructure
from planingfsi.fsi.simulation import Simulation
from planingfsi.potentialflow.pressurepatch import PlaningSurface


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

    mesh = generate_mesh()

    # TODO: We currently require writing to mesh files, because the structure is loaded from files
    mesh.write("mesh")

    simulation = Simulation()

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
    fwd_substructure.add_planing_surface(
        PlaningSurface(
            name="fwd_plate",
            initial_length=0.73,
            minimum_length=0.1,
            num_fluid_elements=40,
            point_spacing="cosine",
        )
    )

    aft_substructure = body.add_substructure(RigidSubstructure(name="aft_plate"))
    aft_substructure.add_planing_surface(
        PlaningSurface(
            name="aft_plate",
            initial_length=1.0,
            minimum_length=0.1,
            num_fluid_elements=40,
            point_spacing="cosine",
        )
    )

    ss = body.add_substructure(RigidSubstructure(name="wetdeck"))
    simulation.config.body._cushion_pressure = 1000.0
    ss.cushion_pressure_type = "Total"

    # TODO: Not sure how to apply cushion pressure to rigid elements without PlaningSurface
    simulation.fluid_solver.add_pressure_cushion(
        {
            "pressureCushionName": "cushion",
            "cushionPressure": 1000.0,
            "upstreamPlaningSurface": "fwd_plate",
            "downstreamPlaningSurface": "aft_plate",
            "upstreamLoc": 0.0,
            "downstreamLoc": -5.0,
            "numElements": 30,
        }
    )

    simulation.run()


if __name__ == "__main__":
    main()
