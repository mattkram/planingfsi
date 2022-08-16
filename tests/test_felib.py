from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from planingfsi.fe.felib import Node


class TestNode:
    """Tests for the `felib.Node` class."""

    @pytest.mark.parametrize(
        "coords",
        [
            np.array([1.0, 2.0]),
            [1.0, 2.0],
            (1.0, 2.0),
        ],
    )
    def test_set_coordinates_from_iterable(self, coords: Iterable[float]):
        """We can set coordinates from any length-two iterable."""
        node = Node(coordinates=coords, node_num=0)
        assert_array_equal(node.coordinates, np.array(coords))

    def test_get_xy_coordinates(self):
        node = Node(np.array([1.0, 2.0]), node_num=0)
        assert node.x == 1.0
        assert node.y == 2.0
