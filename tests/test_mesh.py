from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy
import pytest
from _pytest.monkeypatch import MonkeyPatch

from planingfsi.fe.femesh import Mesh
from planingfsi.fe.femesh import Point
from planingfsi.fe.femesh import Subcomponent


@pytest.fixture()
def mesh() -> Mesh:
    """Create a new mesh with a point added. By default, the origin is point 0."""
    mesh = Mesh()
    mesh.add_point(10, "dir", [0, 10])
    mesh.add_point(20, "dir", [0, 20])
    for pt_id in [0, 10, 20]:
        point = mesh.get_point(pt_id)
        point.set_free_dof("x", "y")
    return mesh


@pytest.mark.parametrize(
    "method, position, expected_coords",
    [
        ("dir", [0, 0], [0, 0]),
        ("rel", [0, 90, 10], [0, 10]),
        ("rel", [0, 180, 10], [-10, 0]),
        ("rel", [0, 270, 10], [0, -10]),
        ("con", [0, "x", 10], [10, 0]),
        ("con", [0, "y", -10], [0, -10]),
        ("pct", [0, 10, 0.00], [0, 0]),
        ("pct", [0, 10, 0.25], [0, 2.5]),
        ("pct", [0, 10, 0.50], [0, 5]),
        ("pct", [0, 10, 0.75], [0, 7.5]),
        ("pct", [0, 10, 1.0], [0, 10.0]),
    ],
)
def test_add_point(
    mesh: Mesh, method: str, position: list[float | str], expected_coords: list[float]
) -> None:
    """We can add points via different methods, and the coordinates should be as expected.

    Point should also be added to the `mesh.points` list.

    """
    point = mesh.add_point(1, method, position)
    assert point in mesh.points
    assert numpy.allclose(point.position, numpy.array(expected_coords))


@pytest.mark.parametrize(
    "method, position, expected_error",
    [
        pytest.param("unknown", [0, 90, 10], NameError, id="Bad method"),
        pytest.param("con", [0, "z", 10], ValueError, id="Bad direction"),
        pytest.param("con", [100, "y", -10], ValueError, id="Missing base point"),
    ],
)
def test_add_point_error(
    mesh: Mesh, method: str, position: list[float | str], expected_error: type[Exception]
) -> None:
    with pytest.raises(expected_error):
        mesh.add_point(1, method, position)


def test_add_load(mesh: Mesh) -> None:
    """Add a fixed load several times to the origin."""
    point = mesh.get_point(0)
    mesh.add_load(0, numpy.array([1.0, 1.0]))
    fixed_load_1 = point.fixed_load
    assert numpy.allclose(fixed_load_1, numpy.array([1.0, 1.0]))

    mesh.add_load(0, numpy.array([1.0, -1.0]))
    fixed_load_2 = point.fixed_load
    assert numpy.allclose(fixed_load_2, numpy.array([2.0, 0.0]))

    # The array never changes ID
    assert fixed_load_1 is fixed_load_2


def test_fix_points(mesh: Mesh) -> None:
    """Fix a few points and ensure they are fixed in both x & y."""
    mesh.fix_points([0, 10])

    for point_id in [0, 10]:
        point = mesh.get_point(point_id)
        assert numpy.all(point.is_dof_fixed == numpy.array([True, True]))

    point = mesh.get_point(20)
    assert numpy.all(point.is_dof_fixed == numpy.array([False, False]))


def test_fix_all_points(mesh: Mesh) -> None:
    """Fix all points and ensure they are fixed in both x & y."""
    mesh.fix_all_points()

    for point_id in [0, 10, 20]:
        point = mesh.get_point(point_id)
        assert numpy.all(point.is_dof_fixed == numpy.array([True, True]))


def test_rotate_points(mesh: Mesh) -> None:
    mesh.rotate_points(0, 90, [0, 10])

    # Point 10 has been rotated 90 degrees
    assert numpy.allclose(mesh.get_point(10).position, numpy.array([-10, 0]))

    # Points 0 & 20 remains in the same position
    assert numpy.allclose(mesh.get_point(0).position, numpy.array([0, 0]))
    assert numpy.allclose(mesh.get_point(20).position, numpy.array([0, 20]))


def test_rotate_all_points(mesh: Mesh) -> None:
    mesh.rotate_all_points(0, 90)

    assert numpy.allclose(mesh.get_point(0).position, numpy.array([0, 0]))
    assert numpy.allclose(mesh.get_point(10).position, numpy.array([-10, 0]))
    assert numpy.allclose(mesh.get_point(20).position, numpy.array([-20, 0]))


def test_move_points(mesh: Mesh) -> None:
    mesh.move_points(10, -10, [10])

    # Point 10 has been rotated 90 degrees
    assert numpy.allclose(mesh.get_point(10).position, numpy.array([10, 0]))

    # Points 0 & 20 remains in the same position
    assert numpy.allclose(mesh.get_point(0).position, numpy.array([0, 0]))
    assert numpy.allclose(mesh.get_point(20).position, numpy.array([0, 20]))


def test_move_all_points(mesh: Mesh) -> None:
    mesh.move_all_points(10, -10)

    assert numpy.allclose(mesh.get_point(0).position, numpy.array([10, -10]))
    assert numpy.allclose(mesh.get_point(10).position, numpy.array([10, 0]))
    assert numpy.allclose(mesh.get_point(20).position, numpy.array([10, 10]))


def test_scale_all_points(mesh: Mesh) -> None:
    mesh.scale_all_points(2.0, 0)

    assert numpy.allclose(mesh.get_point(0).position, numpy.array([0, 0]))
    assert numpy.allclose(mesh.get_point(10).position, numpy.array([0, 20]))
    assert numpy.allclose(mesh.get_point(20).position, numpy.array([0, 40]))


def test_get_diff(mesh: Mesh) -> None:
    assert numpy.allclose(mesh.get_diff(10, 20), numpy.array([0, 10]))


def test_get_length(mesh: Mesh) -> None:
    assert mesh.get_length(0, 10) == pytest.approx(10.0)
    assert mesh.get_length(10, 20) == pytest.approx(10.0)
    assert mesh.get_length(0, 20) == pytest.approx(20.0)


@pytest.fixture()
def submesh(mesh: Mesh) -> Subcomponent:
    return mesh.add_submesh("submesh_name")


def test_add_submesh(mesh: Mesh, submesh: Subcomponent) -> None:
    assert submesh.name == "submesh_name"
    assert submesh.mesh == mesh


@pytest.mark.parametrize(
    "kwargs, attr_name, expected_value",
    [
        ({}, "radius", numpy.inf),
        ({}, "curvature", pytest.approx(0.0, abs=3e-6)),
        ({}, "chord", pytest.approx(10.0)),
        ({}, "arc_length", pytest.approx(10.0)),
        (dict(radius=5), "radius", pytest.approx(5.0)),
        (dict(radius=5), "curvature", pytest.approx(0.2)),
        (dict(radius=5), "chord", pytest.approx(10.0)),
        (dict(radius=5), "arc_length", pytest.approx(5.0 * numpy.pi)),
        (dict(arcLen=5 * numpy.pi), "radius", pytest.approx(5.0)),
        (dict(arcLen=5 * numpy.pi), "curvature", pytest.approx(0.2)),
        (dict(arcLen=5 * numpy.pi), "chord", pytest.approx(10.0)),
        (dict(arcLen=5 * numpy.pi), "arc_length", pytest.approx(5.0 * numpy.pi, rel=1e-5)),
        (dict(arcLen=1), "radius", numpy.inf),
        (dict(arcLen=1), "curvature", pytest.approx(0.0, abs=3e-6)),
        (dict(arcLen=1), "chord", pytest.approx(10.0)),
        (dict(arcLen=1), "arc_length", pytest.approx(10.0)),
    ],
)
def test_add_curve(
    submesh: Subcomponent, kwargs: dict[str, Any], attr_name: str, expected_value: float
) -> None:
    curve = submesh.add_curve(0, 10, **kwargs)
    value = getattr(curve, attr_name)
    assert value == expected_value


@pytest.fixture()
def written_mesh_dir(tmp_path: Path, mesh: Mesh) -> Path:
    mesh_dir = tmp_path / "mesh"
    mesh.get_point(0).set_fixed_dof("x", "y")
    mesh.get_point(10).set_fixed_dof("x")
    mesh.get_point(20).set_fixed_dof("y")
    mesh.get_point(20).add_fixed_load(numpy.array([1.0, 2.0]))
    mesh.write(mesh_dir)
    return mesh_dir


@pytest.mark.parametrize(
    "filename, expected_data",
    [
        ("nodes.txt", numpy.array([[0.0, 0.0], [0.0, 10.0], [0.0, 20.0]])),
        ("fixedDOF.txt", numpy.array([[1, 1], [1, 0], [0, 1]])),
        ("fixedLoad.txt", numpy.array([[0.0, 0.0], [0.0, 0.0], [1.0, 2.0]])),
    ],
)
def test_mesh_write(written_mesh_dir: Path, filename: str, expected_data: numpy.ndarray) -> None:
    file_path = written_mesh_dir / filename
    assert file_path.exists()

    data = numpy.loadtxt(file_path)
    assert numpy.allclose(data, expected_data)


@pytest.mark.parametrize("disp", [True, False])
def test_display(caplog: Any, mesh: Mesh, disp: bool) -> None:
    mesh.display(disp=disp)
    assert ("Curve count:" in caplog.text) is disp
    assert ("Point count:" in caplog.text) is disp


@pytest.mark.parametrize("kwargs, expected_id", [({}, None), ({"id": 1}, 1)])
def test_point_init(kwargs: dict[str, Any], expected_id: int | None) -> None:
    point = Point(**kwargs)
    assert point.id == expected_id


@pytest.mark.parametrize(
    "is_used, expected_is_dof_fixed", [(True, [False, False]), (False, [True, True])]
)
def test_point_is_used(is_used: bool, expected_is_dof_fixed: list[bool]) -> None:
    point = Point()
    point.is_used = is_used
    assert point.is_used == is_used
    assert point.is_dof_fixed == expected_is_dof_fixed


def test_add_fixed_load() -> None:
    point = Point()
    point.add_fixed_load(numpy.array([1.0, 2.0]))
    point.add_fixed_load(numpy.array([1.0, 2.0]))
    assert numpy.allclose(point.fixed_load, numpy.array([2.0, 4.0]))


@pytest.mark.parametrize(
    "dofs, expected_is_dof_fixed",
    [
        (("x", "y"), [False, False]),
        (("x",), [False, True]),
        (("y",), [True, False]),
    ],
)
def test_set_free_dof(dofs: tuple[str, ...], expected_is_dof_fixed: list[bool]) -> None:
    point = Point()
    point.is_dof_fixed = [True, True]
    point.set_free_dof(*dofs)
    assert point.is_dof_fixed == expected_is_dof_fixed


@pytest.mark.parametrize(
    "dofs, expected_is_dof_fixed",
    [
        (("x", "y"), [True, True]),
        (("x",), [True, False]),
        (("y",), [False, True]),
    ],
)
def test_set_fixed_dof(dofs: tuple[str, ...], expected_is_dof_fixed: list[bool]) -> None:
    point = Point()
    point.is_dof_fixed = [False, False]
    point.set_fixed_dof(*dofs)
    assert point.is_dof_fixed == expected_is_dof_fixed


def test_get_position() -> None:
    point = Point()
    point.position = numpy.array([10.0, 20.0])
    assert point.get_x_pos() == pytest.approx(10.0)
    assert point.get_y_pos() == pytest.approx(20.0)

    assert point.x_pos == pytest.approx(10.0)
    assert point.y_pos == pytest.approx(20.0)


def test_plot_mesh(
    monkeypatch: MonkeyPatch, tmp_path: Path, mesh: Mesh, submesh: Subcomponent
) -> None:
    """Smoke test for plotting the mesh."""
    monkeypatch.chdir(tmp_path)
    submesh.add_curve(0, 10, Nel=3)
    mesh.plot(save=True)
    assert (tmp_path / "meshLayout.eps").exists()


def test_rotate_point(mesh: Mesh) -> None:
    point = mesh.get_point(20)
    point.rotate(10, 90)
    assert numpy.allclose(point.position, numpy.array([-10, 10]))

    point.rotate(mesh.get_point(10), 90)
    assert numpy.allclose(point.position, numpy.array([0, 0]))

    another_point = Point()
    with pytest.raises(LookupError):
        another_point.rotate(10, 90)


def test_point_index(mesh: Mesh) -> None:
    """The point index is only available if the point is added via mesh.add_point."""
    point_in_mesh = mesh.add_point(200, "dir", [0, 0])
    assert point_in_mesh.index is not None

    point_not_in_mesh = Point(id=200)
    with pytest.raises(ValueError):
        _ = point_not_in_mesh.index
