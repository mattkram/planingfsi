from __future__ import annotations

import pytest

from planingfsi.dictionary import _apply_key_map


def test_apply_key_map() -> None:
    dict_ = {"wettedLength": 1.0, "minLength": 0.0, "min_length": 0.0}
    target_dict = {"wetted_length": 1.0, "minLength": 0.0, "min_length": 0.0}

    key_map = {"wettedLength": "wetted_length", "anotherKey": "another_key"}

    dict_ = _apply_key_map(dict_.copy(), key_map)
    assert dict_ == target_dict

    with pytest.raises(KeyError):
        _apply_key_map(dict_.copy(), {"minLength": "min_length"})
