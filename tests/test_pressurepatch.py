from __future__ import annotations

import pytest

from planingfsi.potentialflow.pressurepatch import PlaningSurface


def test_must_set_parent_to_access_config() -> None:
    surface = PlaningSurface()
    with pytest.raises(ValueError):
        surface.config
