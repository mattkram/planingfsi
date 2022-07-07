from __future__ import annotations

from typing import Any
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import numpy
import pytest

from planingfsi.fe.femesh import Mesh
from planingfsi.fe.femesh import Point


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
    mesh: Mesh, method: str, position: List[Union[float, str]], expected_coords: List[float]
) -> None:
    """We can add points via different methods, and the coordinates should be as expected.

    Point should also be added to the `mesh.points` list.

    """
    point = mesh.add_point(1, method, position)
    assert point in mesh.points
    assert numpy.allclose(point.pos, numpy.array(expected_coords))


@pytest.mark.parametrize(
    "method, position, expected_error",
    [
        pytest.param("unknown", [0, 90, 10], NameError, id="Bad method"),
        pytest.param("con", [0, "z", 10], ValueError, id="Bad direction"),
        pytest.param("con", [100, "y", -10], ValueError, id="Missing base point"),
    ],
)
def test_add_point_error(
    mesh: Mesh, method: str, position: List[Union[float, str]], expected_error: Type[Exception]
) -> None:
    with pytest.raises(expected_error):
        mesh.add_point(1, method, position)


def test_add_load(mesh: Mesh) -> None:
    """Add a fixed load several times to the origin."""
    point = mesh.get_point(0)
    mesh.add_load(0, numpy.array([1.0, 1.0]))
    fixed_load_1 = point.get_fixed_load()
    assert numpy.allclose(fixed_load_1, numpy.array([1.0, 1.0]))

    mesh.add_load(0, numpy.array([1.0, -1.0]))
    fixed_load_2 = point.get_fixed_load()
    assert numpy.allclose(fixed_load_2, numpy.array([2.0, 0.0]))

    # The array never changes ID
    assert fixed_load_1 is fixed_load_2


def test_fix_points(mesh: Mesh) -> None:
    """Fix a few points and ensure they are fixed in both x & y."""
    mesh.fix_points([0, 10])

    for point_id in [0, 10]:
        point = mesh.get_point(point_id)
        assert numpy.all(point.get_fixed_dof() == numpy.array([True, True]))

    point = mesh.get_point(20)
    assert numpy.all(point.get_fixed_dof() == numpy.array([False, False]))


def test_fix_all_points(mesh: Mesh) -> None:
    """Fix all points and ensure they are fixed in both x & y."""
    mesh.fix_all_points()

    for point_id in [0, 10, 20]:
        point = mesh.get_point(point_id)
        assert numpy.all(point.get_fixed_dof() == numpy.array([True, True]))


def test_rotate_points(mesh: Mesh) -> None:
    mesh.rotate_points(0, 90, [0, 10])

    # Point 10 has been rotated 90 degrees
    assert numpy.allclose(mesh.get_point(10).get_position(), numpy.array([-10, 0]))

    # Points 0 & 20 remains in the same position
    assert numpy.allclose(mesh.get_point(0).get_position(), numpy.array([0, 0]))
    assert numpy.allclose(mesh.get_point(20).get_position(), numpy.array([0, 20]))


def test_rotate_all_points(mesh: Mesh) -> None:
    mesh.rotate_all_points(0, 90)

    assert numpy.allclose(mesh.get_point(0).get_position(), numpy.array([0, 0]))
    assert numpy.allclose(mesh.get_point(10).get_position(), numpy.array([-10, 0]))
    assert numpy.allclose(mesh.get_point(20).get_position(), numpy.array([-20, 0]))


def test_move_points(mesh: Mesh) -> None:
    mesh.move_points(10, -10, [10])

    # Point 10 has been rotated 90 degrees
    assert numpy.allclose(mesh.get_point(10).get_position(), numpy.array([10, 0]))

    # Points 0 & 20 remains in the same position
    assert numpy.allclose(mesh.get_point(0).get_position(), numpy.array([0, 0]))
    assert numpy.allclose(mesh.get_point(20).get_position(), numpy.array([0, 20]))


def test_move_all_points(mesh: Mesh) -> None:
    mesh.move_all_points(10, -10)

    assert numpy.allclose(mesh.get_point(0).get_position(), numpy.array([10, -10]))
    assert numpy.allclose(mesh.get_point(10).get_position(), numpy.array([10, 0]))
    assert numpy.allclose(mesh.get_point(20).get_position(), numpy.array([10, 10]))


def test_scale_all_points(mesh: Mesh) -> None:
    mesh.scale_all_points(2.0, 0)

    assert numpy.allclose(mesh.get_point(0).get_position(), numpy.array([0, 0]))
    assert numpy.allclose(mesh.get_point(10).get_position(), numpy.array([0, 20]))
    assert numpy.allclose(mesh.get_point(20).get_position(), numpy.array([0, 40]))


def test_get_diff(mesh: Mesh) -> None:
    assert numpy.allclose(mesh.get_diff(10, 20), numpy.array([0, 10]))


def test_get_length(mesh: Mesh) -> None:
    assert mesh.get_length(0, 10) == pytest.approx(10.0)
    assert mesh.get_length(10, 20) == pytest.approx(10.0)
    assert mesh.get_length(0, 20) == pytest.approx(20.0)


@pytest.mark.parametrize("disp", [True, False])
def test_display(caplog: Any, mesh: Mesh, disp: bool) -> None:
    mesh.display(disp=disp)
    assert ("Line count:" in caplog.text) is disp
    assert ("Point count:" in caplog.text) is disp


@pytest.mark.parametrize("kwargs, expected_id", [({}, None), ({"id": 1}, 1)])
def test_point_init(kwargs: dict[str, int], expected_id: Optional[int]) -> None:
    point = Point(**kwargs)
    assert point.ID == expected_id
