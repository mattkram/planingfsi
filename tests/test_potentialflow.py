import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from planingfsi.potentialflow.solver import _grow_points


@pytest.mark.parametrize(
    "x0, x1, max_x",
    [
        (0.0, 1.0, 20.0),
        (0.0, -1.0, -20.0),
        (1.0, 2.0, 51.0),
        (-1.0, -2.0, -51.0),
        (50.0, 51.0, 100.0),
        (-50.0, -51.0, -100.0),
    ],
)
def test_grow_points(x0, x1, max_x):
    growth_rate = 1.1
    points = _grow_points(x0, x1, max_x, rate=growth_rate)
    point_ratio = (points[2:] - points[1:-1]) / (points[1:-1] - points[:-2])

    # https://math.stackexchange.com/a/1897065
    n = 0
    while (abs(x1 - x0) * (1 - growth_rate**n) / (1 - growth_rate)) < abs(max_x - x0):
        n += 1
    tmp = (x1 - x0) * growth_rate ** np.array(range(n))
    tmp = np.cumsum(tmp)
    tmp = x0 + np.hstack((np.array([0]), tmp))

    assert_array_almost_equal(points, tmp)

    # We include the first and second points
    assert x0 in points
    assert x1 in points

    # The growth rate is constant and as-specified
    assert all(p == pytest.approx(growth_rate) for p in point_ratio)

    # Ensure the final point is the only one past max_x
    next_pt = (points[-1] - points[-2]) * growth_rate + points[-1]
    if x1 > x0:
        assert max_x <= points[-1] <= next_pt
    else:
        assert max_x >= points[-1] >= next_pt
