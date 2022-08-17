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
        node = Node(coordinates=coords)
        assert_array_equal(node.coordinates, np.array(coords))

    def test_get_xy_coordinates(self):
        node = Node(np.array([1.0, 2.0]))
        assert node.x == 1.0
        assert node.y == 2.0

    def test_free_by_default(self):
        node = Node(np.array([0, 0]))
        assert node.is_dof_fixed == (False, False)

    def test_fixed_load_is_float_array(self):
        node = Node(np.array([0, 0]), fixed_load=[1, 2])
        assert_array_equal(node.fixed_load, np.array([1.0, 2.0]))

    def test_move_node(self):
        node = Node(np.array([0, 0]))
        node.move(4.0, 5.0)
        assert_array_equal(node.coordinates, np.array([4.0, 5.0]))
