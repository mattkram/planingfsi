from __future__ import annotations

import math
from pathlib import Path

import pytest
from _pytest.fixtures import SubRequest

from planingfsi.config import Config
from planingfsi.config import ConfigItem
from planingfsi.config import PlotConfig
from planingfsi.config import SubConfig


class TestClass(SubConfig):
    """A simple configuration class to test behavior of attribute descriptors."""

    float_attr = ConfigItem(default=0.0)
    int_attr = ConfigItem(type=int)
    bool_attr = ConfigItem(default=True)


@pytest.fixture()
def config_instance() -> TestClass:
    """An instance of a configuration class containing some attributes."""
    return TestClass()


def test_config_init(config_instance: TestClass) -> None:
    """Given an instance of the TestClass, attributes are None unless a default is provided."""
    assert config_instance is not None
    assert config_instance.float_attr == 0.0
    assert isinstance(config_instance.float_attr, float)


def test_config_attribute_without_default_raises_exception(config_instance: TestClass) -> None:
    """An AttributeError is raised if there is no default."""
    with pytest.raises(AttributeError):
        _ = config_instance.int_attr


def test_config_type_conversion(config_instance: TestClass) -> None:
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
def test_config_bool_setter(config_instance: TestClass, value: str | bool, expected: bool) -> None:
    """When setting a Boolean value, True and False can be passed in as strings."""
    config_instance.bool_attr = value
    assert config_instance.bool_attr == expected


@pytest.fixture()
def config() -> Config:
    config = Config()
    # config.load_from_file("configDict")
    return config


def test_flow_defaults(config: Config) -> None:
    """Test the raw default values in the FlowConfig class are set correctly."""
    flow = config.flow
    assert flow.density == 998.2
    assert flow.gravity == 9.81
    assert flow.kinematic_viscosity == 1e-6
    assert flow.waterline_height == 0.0
    assert not flow.include_friction


def test_flow_speed_requires_value(config: Config) -> None:
    """If Froude number and flow speed are both unset, access should raise ValueError."""
    flow = config.flow
    flow._froude_num = None
    flow._flow_speed = None
    with pytest.raises(ValueError):
        _ = flow.flow_speed
    with pytest.raises(ValueError):
        _ = flow.froude_num


def test_set_flow_speed_only_once(config: Config) -> None:
    """The Froude number and flow speed can't both be set, otherwise a ValueError is raised."""
    flow = config.flow
    flow._froude_num = 1.0
    flow._flow_speed = 1.0
    with pytest.raises(ValueError):
        _ = flow.flow_speed


def test_set_flow_speed(config: Config) -> None:
    """Setting the flow speed directly, Froude number will be calculated."""
    flow = config.flow
    flow._froude_num = None
    flow._flow_speed = 1.0
    assert flow.flow_speed == 1.0
    assert flow.froude_num == pytest.approx(
        flow.flow_speed / math.sqrt(flow.gravity * flow.reference_length)
    )


def test_set_froude_number(config: Config) -> None:
    """Setting the Froude number, flow speed will be calculated."""
    flow = config.flow
    flow._froude_num = 1.0
    flow._flow_speed = None
    assert flow.flow_speed == pytest.approx(
        flow.froude_num * math.sqrt(flow.gravity * flow.reference_length)
    )
    assert flow.froude_num == 1.0


def test_flow_derived_quantities(config: Config) -> None:
    """Derived quantities should return a value once flow speed is set."""
    flow = config.flow
    flow.froude_num = 1.0
    assert flow.stagnation_pressure is not None
    assert flow.k0 is not None
    assert flow.lam is not None


def test_body_defaults(config: Config) -> None:
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


def test_body_pressure_calculations(config: Config) -> None:
    body = config.body

    assert body.Pc == 0.0
    assert body.PcBar == 0.0

    assert body.Ps == 0.0
    assert body.PsBar == 0.0

    body.weight = 5.0
    body.PcBar = 10.0
    assert body.Pc == 50.0

    config.body._seal_pressure = 0.0
    assert config.body.PsBar == 0.0

    config.body._cushion_pressure = 10.0
    config.body._seal_pressure = 100.0
    assert config.body.PsBar == 10.0

    config.body.PsBar = 20.0
    assert config.body._seal_pressure == 200.0


@pytest.fixture()
def config_from_file(test_dir: Path, config: Config) -> Config:
    config.load_from_file(test_dir / "input_files" / "configDict")
    return config


def test_load_config_from_file(config_from_file: Config) -> None:
    """Configuration loaded from file overrides defaults."""
    assert config_from_file.flow.density == 998.2
    assert config_from_file.flow.kinematic_viscosity == 1.0048e-6


@pytest.mark.parametrize(
    "attr_to_set_true, expected", [(None, False), ("_watch", True), ("show", True)]
)
def test_plot_config_watch(config: Config, attr_to_set_true: str | None, expected: bool) -> None:
    """By default, we don't watch the plot unless certain attributes are set to True."""
    if attr_to_set_true is not None:
        setattr(config.plotting, attr_to_set_true, True)
    assert config.plotting.watch is expected


@pytest.mark.parametrize(
    "attr_to_set_true, expected",
    [(None, False), ("show", True), ("show_pressure", True), ("watch", True)],
)
def test_plot_config_plot_any_getter(
    config: Config, attr_to_set_true: str | None, expected: bool
) -> None:
    if attr_to_set_true is not None:
        setattr(config.plotting, attr_to_set_true, True)
    assert config.plotting.plot_any is expected


@pytest.fixture()
def config_with_all_plot_true(config: Config) -> Config:
    config.plotting.save = True
    config.plotting.show = True
    config.plotting.show_pressure = True
    config.plotting.watch = True
    return config


@pytest.mark.parametrize("attr_to_set_false", ["save", "show", "show_pressure", "_watch"])
def test_plot_config_plot_any_setter(
    config_with_all_plot_true: Config, attr_to_set_false: str
) -> None:
    config_with_all_plot_true.plotting.plot_any = False
    value = getattr(config_with_all_plot_true.plotting, attr_to_set_false)
    assert value is False


def test_plot_config_plot_any_setter_true_raises_exception(config: Config) -> None:
    with pytest.raises(ValueError):
        config.plotting.plot_any = True


@pytest.fixture(params=["_x_fs_min_max", "xmin_xmax", "lambda_min_max"])
def expected_x_min_max(config: Config, request: SubRequest) -> float:
    """Each of the provided configs should have an x_fs_min and x_fs_max of 10.0.

    These are tested together since the logic should be identical.

    """
    value = 10.0
    if request.param == "_x_fs_min_max":
        config.plotting._x_fs_min = value
        config.plotting._x_fs_max = value
    elif request.param == "xmin_xmax":
        config.plotting.xmin = value
        config.plotting.xmax = value
    elif request.param == "lambda_min_max":
        # Need to set flow speed, can be anything
        config.flow.flow_speed = 1.0
        config.plotting.lambda_min = value / config.flow.lam
        config.plotting.lambda_max = value / config.flow.lam
    else:
        raise ValueError("Invalid request.param")  # pragma: no cover

    return value


@pytest.mark.parametrize("attr_name", ["x_fs_min", "x_fs_max"])
def test_plot_config_x_fs_min_max(
    config: Config, expected_x_min_max: float, attr_name: str
) -> None:
    assert getattr(config.plotting, attr_name) == expected_x_min_max


@pytest.mark.parametrize("attr_name", ["x_fs_min", "x_fs_max"])
def test_plot_config_x_fs_min_max_requires_parent(attr_name: str) -> None:
    flow_config = PlotConfig()
    with pytest.raises(ValueError):
        getattr(flow_config, attr_name)


@pytest.fixture(params=["stagnation", "cushion-0", "cushion-10", "hydrostatic", "other"])
def expected_pressure_scale(config: Config, request: SubRequest) -> float:
    """Apply various configuration settings, returning the expected pressure scale for plotting."""
    pressure_scale_pct = 0.1
    config.plotting.pressure_scale_method = request.param
    config.plotting._pressure_scale_pct = pressure_scale_pct

    if request.param == "stagnation":
        config.flow.flow_speed = 1.0
        config.flow.density = 2.0
        return 1.0 * pressure_scale_pct
    elif request.param == "cushion-0":
        config.plotting.pressure_scale_method = "cushion"
        config.body._cushion_pressure = 0.0
        return 1.0 * pressure_scale_pct
    elif request.param == "cushion-10":
        config.plotting.pressure_scale_method = "cushion"
        config.body._cushion_pressure = 10.0
        return 10.0 * pressure_scale_pct
    elif request.param == "hydrostatic":
        config.flow.density = 1.0
        config.flow.gravity = 10.0
        config.plotting._pressure_scale_head = 10.0
        return 100.0 * pressure_scale_pct
    else:
        config.plotting._pressure_scale = 10.0
        return 10.0 * pressure_scale_pct


def test_plot_config_pressure_scale(config: Config, expected_pressure_scale: float) -> None:
    assert config.plotting.pressure_scale == expected_pressure_scale
