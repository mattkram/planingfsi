from planingfsi import Mesh


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
    mesh = generate_mesh(10.0)
    mesh.plot(show=True)


if __name__ == "__main__":
    main()
