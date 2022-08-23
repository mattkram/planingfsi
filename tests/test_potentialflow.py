import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from planingfsi.potentialflow.solver import _grow_points


def test_grow_points():
    x0, x1 = 0.0, -1.0
    growth_rate = 1.1
    max_dist = 20.0
    points = _grow_points(x0, x1, max_dist, rate=growth_rate)
    point_ratio = (points[2:] - points[1:-1]) / (points[1:-1] - points[:-2])
    assert all(p == pytest.approx(growth_rate) for p in point_ratio)

    # https://math.stackexchange.com/a/1897065
    n = 0
    while (abs(x1 - x0) * (1 - growth_rate**n) / (1 - growth_rate)) < max_dist:
        n += 1
    tmp = (x1 - x0) * growth_rate ** np.array(range(n))
    tmp = np.cumsum(tmp)
    tmp = x0 + np.hstack((np.array([0]), tmp))
    tmp_ratio = (tmp[2:] - tmp[1:-1]) / (tmp[1:-1] - tmp[:-2])
    assert all(p == pytest.approx(growth_rate) for p in tmp_ratio)
    assert_array_almost_equal(points, tmp)
