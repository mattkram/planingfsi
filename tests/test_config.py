import math
from typing import Union

import pytest

from planingfsi import config
from planingfsi.config import SubConfig, ConfigItem


class TestClass(SubConfig):
    """A simple configuration class to test behavior of attribute descriptors."""
    float_attr = ConfigItem(default=0.0)
    int_attr = ConfigItem(type=int)
    bool_attr = ConfigItem(default=True)


@pytest.fixture()
def config_instance() -> TestClass:
    """An instance of a configuration class containing some attributes."""
    return TestClass()


def test_config_init(config_instance):
    """Given an instance of the TestClass, attributes are None unless a default is provided."""
    assert config_instance is not None
    assert config_instance.float_attr == 0.0
    assert isinstance(config_instance.float_attr, float)


def test_config_attribute_without_default_raises_exception(config_instance):
    """An AttributeError is raised if there is no default."""
    with pytest.raises(AttributeError):
        _ = config_instance.int_attr


def test_config_type_conversion(config_instance):
    """Conversion is performed based on the specified type."""
    config_instance.int_attr = 55.0
    assert config_instance.int_attr == 55
    assert isinstance(config_instance.int_attr, int)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("False", False),
        ("false", False),
        ("True", True),
        ("true", True),
        (False, False),
        (True, True),
    ],
)
def test_config_bool_type_converion(
    config_instance: TestClass, value: Union[str, bool], expected: bool
) -> None:
    """When setting a Boolean value, True and False can be passed in as strings."""
    config_instance.bool_attr = value
    assert config_instance.bool_attr == expected


def test_flow_defaults():
    """Test the raw default values in the FlowConfig class are set correctly."""
    flow = config.flow
    assert flow.density == 998.2
    assert flow.gravity == 9.81
    assert flow.kinematic_viscosity == 1e-6
    assert flow.waterline_height == 0.0
    assert flow.num_dim == 2
    assert not flow.include_friction


def test_flow_speed_requires_value():
    """If Froude number and flow speed are both unset, access should raise ValueError."""
    flow = config.flow
    flow._froude_num = None
    flow._flow_speed = None
    with pytest.raises(ValueError):
        _ = flow.flow_speed
    with pytest.raises(ValueError):
        _ = flow.froude_num


def test_set_flow_speed_only_once():
    """The Froude number and flow speed can't both be set, otherwise a ValueError is raised."""
    flow = config.flow
    flow._froude_num = 1.0
    flow._flow_speed = 1.0
    with pytest.raises(ValueError):
        _ = flow.flow_speed


def test_set_flow_speed():
    """Setting the flow speed directly, Froude number will be calculated."""
    flow = config.flow
    flow._froude_num = None
    flow._flow_speed = 1.0
    assert flow.flow_speed == 1.0
    assert flow.froude_num == pytest.approx(
        flow.flow_speed / math.sqrt(flow.gravity * flow.reference_length)
    )


def test_set_froude_number():
    """Setting the Froude number, flow speed will be calculated."""
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
    assert body.relax_draft == 1.0
    assert body.relax_trim == 1.0


def test_body_pressure_calculations():
    body = config.body

    assert body.Pc == 0.0
    assert body.PcBar == 0.0

    assert body.Ps == 0.0
    assert body.PsBar == 0.0

    body.weight = 5.0
    body.PcBar = 10.0
    assert body.Pc == 50.0


@pytest.fixture()
def config_from_file(test_dir):
    config.load_from_file(test_dir / "input_files" / "configDict")


@pytest.mark.usefixtures("config_from_file")
def test_load_config_from_file():
    """Configuration loaded from file overrides defaults."""
    assert config.flow.density == 998.2
    assert config.flow.kinematic_viscosity == 1.0048e-6
