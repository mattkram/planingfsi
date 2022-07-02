from typing import List
from typing import Union

import numpy
import pytest

from planingfsi.fe.femesh import Mesh


@pytest.fixture()
def mesh() -> Mesh:
    """Create a new mesh with a point added. By default, the origin is point 0."""
    mesh = Mesh()
    mesh.add_point(10, "dir", [0, 10])
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
    assert len(mesh.points) == 3
    assert point in mesh.points
    assert numpy.allclose(point.pos, numpy.array(expected_coords))
