import math

import pytest

from planingfsi import config
from planingfsi.config import SubConfig, ConfigItem


@pytest.fixture()
def config_class():
    class TestClass(SubConfig):
        float_attr = ConfigItem(default=0.0)
        int_attr = ConfigItem(type_=int)

    return TestClass()


def test_config_init(config_class):
    assert config_class is not None
    assert config_class.float_attr == 0.0
    assert config_class.int_attr is None
    assert isinstance(config_class.float_attr, float)


def test_config_type_conversion(config_class):
    config_class.int_attr = 55.0
    assert config_class.int_attr == 55
    assert isinstance(config_class.int_attr, int)


def test_flow_defaults():
    """Test the raw default values in the FlowConfig class are set correctly."""
    flow = config.flow
    assert flow.density == 998.2
    assert flow.gravity == 9.81
    assert flow.kinematic_viscosity == 1e-6
    assert flow.waterline_height == 0.0
    assert flow.num_dim == 2
    assert not flow.include_friction
    assert flow._froude_num is None
    assert flow._flow_speed is None


def test_flow_speed_requires_value():
    """If Froude number and flow speed are both unset, access should raise ValueError."""
    flow = config.flow
    flow._froude_num = None
    flow._flow_speed = None
    with pytest.raises(ValueError):
        _ = flow.flow_speed
    with pytest.raises(ValueError):
        _ = flow.froude_num


def test_set_flow_speed():
    """Set the flow speed directly. Froude number will be calculated."""
    flow = config.flow
    flow._froude_num = None
    flow._flow_speed = 1.0
    assert flow.flow_speed == 1.0
    assert flow.froude_num == pytest.approx(
        flow.flow_speed / math.sqrt(flow.gravity * flow.reference_length)
    )


def test_set_froude_number():
    """Set the Froude number. Flow speed will be calculated."""
    flow = config.flow
    flow._froude_num = 1.0
    flow._flow_speed = None
    assert flow.flow_speed == pytest.approx(
        flow.froude_num * math.sqrt(flow.gravity * flow.reference_length)
    )
    assert flow.froude_num == 1.0


def test_flow_derived_quantities():
    """Derived quantities should return a value once flow speed is set."""
    flow = config.flow
    flow.froude_num = 1.0
    assert flow.stagnation_pressure is not None
    assert flow.k0 is not None
    assert flow.lam is not None


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
