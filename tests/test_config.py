import math

import pytest

from planingfsi import config


def test_flow_defaults():
    flow = config.flow
    assert flow.density == 998.2
    assert flow.gravity == 9.81
    assert flow.froude_num == 1.0
    assert flow.kinematic_viscosity == 1e-6
    assert flow.waterline_height == 0.0


def test_flow_speed_calculations():
    flow = config.flow

    assert flow._flow_speed is None
    assert flow.flow_speed == pytest.approx(flow.froude_num * math.sqrt(flow.gravity))

    flow._froude_num = None
    with pytest.raises(ValueError):
        _ = flow.flow_speed
    with pytest.raises(ValueError):
        _ = flow.froude_num


def test_body_defaults():
    body = config.body

    assert body.xCofG == 0.0
    assert body.yCofG == 0.0
    assert body.xCofR == 0.0
    assert body.yCofR == 0.0
    assert body.reference_length == 1.0
    assert body.mass == 1.0
    assert body.weight == config.flow.gravity


def test_body_pressure_calculations():
    body = config.body

    assert body.Pc == 0.0
    assert body.PcBar == 0.0

    assert body.Ps == 0.0
    assert body.PsBar == 0.0

    body.weight = 5.0
    body.PcBar = 10.0
    assert body.Pc == 50.0
