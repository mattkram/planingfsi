"""This module is used to store the global configuration. Values are stored
after reading the configDict file, and values can be accessed by other
packages and modules by importing the config module.

Usage: from planingfsi import config

The global attributes can then be simply accessed via config.attribute_name

"""
import math
import os
from pathlib import Path
from typing import Any, List, Type

import matplotlib
from krampy.iotools import load_dict_from_file

from . import logger

DICT_NAME = "configDict"


class ConfigItem:
    """A descriptor to represent a configuration item.

    Attributes are loaded from a dictionary with fancy default handling.

    """

    name: str

    def __init__(
        self, *alt_keys: str, default: Any = None, type_: Type = None,
    ):
        self.alt_keys: List[str] = list(alt_keys)
        self.default = default
        self.type_ = type_

    def __get__(self, instance, owner):
        """Retrieve the value from the instance dictionary or return the default.

        Raises:
            AttributeError: If the attribute hasn't been set and the  default value isn't specified.

        """
        if self.name in instance.__dict__:
            return instance.__dict__[self.name]
        elif self.default is not None:
            return self.default
        raise AttributeError(
            f"Attribute {self.name} has not been set and no default specified"
        )

    def __set__(self, instance: "SubConfig", value: Any):
        """When the value is set, try to convert it and then store it in the instance dictionary."""
        if value is not None and self.type_ is not None:
            value = self.type_(value)
        instance.__dict__[self.name] = value

    def __set_name__(self, _, name: str):
        """Set the name when the ConfigItem is defined."""
        self.name = name

    @property
    def keys(self) -> List[str]:
        """A list of keys to look for when reading in the value."""
        return [self.name] + self.alt_keys

    def get_from_dict(self, dict_):
        """Try to read all keys from the dictionary until a non-None value is found.

        Returns the default value if no appropriate value is found in the dictionary.

        """
        for key in self.keys:
            value = dict_.get(key)
            if value is not None:
                return value
        raise KeyError(
            'None of the following keys "{}" found in dictionary.'.format(self.keys)
        )


class SubConfig:
    """An empty class used simply for dividing the configuration into
    different sections. Also useful in helping define the namespace scopes.
    """

    def load_from_dict(self, dict_name: str):
        """Load the configuration from a dictionary file.

        Parameters
        ----------
        dict_name
            The path to the dictionary file.

        """
        dict_ = load_dict_from_file(dict_name)
        for key, config_item in self.__class__.__dict__.items():
            if isinstance(config_item, ConfigItem):
                try:
                    value = config_item.get_from_dict(dict_)
                except KeyError:
                    pass
                else:
                    setattr(self, config_item.name, value)


class FlowConfig(SubConfig):
    density = ConfigItem("rho", default=998.2)
    gravity = ConfigItem("g", default=9.81)
    kinematic_viscosity = ConfigItem("nu", default=1e-6)
    waterline_height = ConfigItem("hWL", default=0.0)
    num_dim = ConfigItem("dim", default=2)
    include_friction = ConfigItem("shearCalc", default=False)

    _froude_num = ConfigItem("Fr", default=None, type_=float)
    _flow_speed = ConfigItem("U", default=None, type_=float)

    @property
    def reference_length(self):
        return body.reference_length

    @property
    def flow_speed(self) -> float:
        """The flow speed is the native variable to store free-stream velocity. However, if Froude
        number is set from input file, that should override the flow speed input.

        """
        if self._froude_num is not None:
            if self._flow_speed is not None:
                raise ValueError(
                    "Only one flow speed variable (either Froude number or flow "
                    "speed) must be set in {0}".format(DICT_NAME)
                )
            self.froude_num = self._froude_num
        elif self._flow_speed is None:
            raise ValueError("Must specify either U or Fr in {0}".format(DICT_NAME))
        return getattr(self, "_flow_speed")

    @flow_speed.setter
    def flow_speed(self, value):
        """Set the raw flow speed variable and ensure raw Froude number is not also set."""
        self._flow_speed = value
        self._froude_num = None

    @property
    def froude_num(self):
        return self.flow_speed / math.sqrt(self.gravity * self.reference_length)

    @froude_num.setter
    def froude_num(self, value):
        self.flow_speed = value * math.sqrt(self.gravity * self.reference_length)

    @property
    def stagnation_pressure(self):
        return 0.5 * self.density * self.flow_speed ** 2

    @property
    def k0(self):
        return self.gravity / self.flow_speed ** 2

    @property
    def lam(self):
        return 2 * math.pi / self.k0


class BodyConfig(SubConfig):
    xCofG = ConfigItem(default=0.0)
    yCofG = ConfigItem(default=0.0)
    _xCofR = ConfigItem(type_=float)
    _yCofR = ConfigItem(type_=float)

    mass = ConfigItem("m", default=1.0)
    _weight = ConfigItem("W")

    reference_length = ConfigItem("Lref", "Lc", default=1.0)

    _cushion_pressure = ConfigItem("Pc", default=0.0)
    _seal_pressure = ConfigItem("Ps", default=0.0)

    # Rigid body motion parameters
    time_step = ConfigItem("timeStep", default=1e-3)
    relax_rigid_body = ConfigItem("rigidBodyRelax", default=1.0)
    motion_method = ConfigItem("motionMethod", default="Physical")
    motion_jacobian_first_step = ConfigItem("motionJacobianFirstStep", default=1e-6)

    bow_seal_tip_load = ConfigItem("bowSealTipLoad", default=0.0)
    tip_constraint_ht = ConfigItem("tipConstraintHt", type_=float)

    seal_load_pct = ConfigItem("sealLoadPct", default=1.0)
    cushion_force_method = ConfigItem("cushionForceMethod", default="Fixed")

    initial_draft = ConfigItem("initialDraft", default=0.0)
    initial_trim = ConfigItem("initialTrim", default=0.0)

    max_draft_step = ConfigItem("maxDraftStep", default=1e-3)
    max_trim_step = ConfigItem("maxTrimStep", default=1e-3)

    max_draft_acc = ConfigItem("maxDraftAcc", default=1000.0)
    max_trim_acc = ConfigItem("maxTrimAcc", default=1000.0)

    free_in_draft = ConfigItem("freeInDraft", default=False)
    free_in_trim = ConfigItem("freeInTrim", default=False)

    draft_damping = ConfigItem("draftDamping", default=1000.0)
    trim_damping = ConfigItem("trimDamping", default=500.0)

    _relax_draft = ConfigItem("draftRelax", type_=float)
    _relax_trim = ConfigItem("trimRelax", type_=float)

    @property
    def relax_draft(self):
        if self._relax_draft is not None:
            return self._relax_draft
        return body.relax_rigid_body

    @property
    def relax_trim(self):
        if self._relax_trim is not None:
            return self._relax_trim
        return body.relax_rigid_body

    @property
    def Pc(self):
        return self._cushion_pressure

    @property
    def PcBar(self):
        return self._cushion_pressure * self.reference_length / self.weight

    @PcBar.setter
    def PcBar(self, value):
        self._cushion_pressure = value * self.weight / self.reference_length

    @property
    def Ps(self):
        return self._seal_pressure

    @property
    def PsBar(self):
        if self._cushion_pressure == 0.0:
            return 0.0
        return self._seal_pressure / self._cushion_pressure

    @PsBar.setter
    def PsBar(self, value):
        self._seal_pressure = value * self._cushion_pressure

    @property
    def xCofR(self):
        try:
            return self._xCofR
        except AttributeError:
            return self.xCofG

    @property
    def yCofR(self):
        try:
            return self._yCofR
        except AttributeError:
            return self.yCofG

    @property
    def weight(self):
        try:
            return self._weight
        except AttributeError:
            return self.mass * flow.gravity

    @weight.setter
    def weight(self, value):
        self.mass = value / flow.gravity


class PlotConfig(SubConfig):
    pType = ConfigItem("pScaleType", default="stagnation")
    _pScale = ConfigItem("pScale", default=1.0)
    _pScalePct = ConfigItem("pScalePct", default=1.0)
    _pScaleHead = ConfigItem("pScaleHead", default=1.0)
    growth_rate = ConfigItem("growthRate", default=1.1)
    CofR_grid_len = ConfigItem("CofRGridLen", default=0.5)
    fig_format = ConfigItem("figFormat", default="eps")

    pressure_limiter = ConfigItem("pressureLimiter", default=False)

    # Load plot extents
    ext_e = ConfigItem("extE", default=0.1)
    ext_w = ConfigItem("extW", default=0.1)
    ext_n = ConfigItem("extN", default=0.1)
    ext_s = ConfigItem("extS", default=0.1)

    xmin = ConfigItem("plotXMin", type_=float)
    xmax = ConfigItem("plotXMax", type_=float)
    ymin = ConfigItem("plotYMin", type_=float)
    ymax = ConfigItem("plotYMax", type_=float)

    lambda_min = ConfigItem("lamMin", default=-1.0)
    lambda_max = ConfigItem("lamMax", default=1.0)

    _x_fs_min = ConfigItem("xFSMin", type_=float)
    _x_fs_max = ConfigItem("xFSMax", type_=float)

    # Whether to save, show, or watch plots
    save = ConfigItem("plotSave", default=False)
    show_pressure = ConfigItem("plotPressure", default=False)
    show = ConfigItem("plotShow", default=False)
    _watch = ConfigItem("plotWatch", default=False)

    @property
    def watch(self):
        return self._watch or self.show

    @property
    def plot_any(self):
        return self.show or self.save or self.watch or self.show_pressure

    @plot_any.setter
    def plot_any(self, value: bool) -> None:
        if not value:
            self.save = False
            self.show = False
            self.show_pressure = False
            self._watch = False
        else:
            raise ValueError("config.plotting.plot_any cannot be set to True")

    @property
    def x_fs_min(self):
        if self._x_fs_min is not None:
            return self._x_fs_min
        if self.xmin is not None:
            return self.xmin
        return self.lambda_min * flow.lam

    @property
    def x_fs_max(self):
        if self._x_fs_max is not None:
            return self._x_fs_max
        if self.xmax is not None:
            return self.xmax
        return self.lambda_max * flow.lam

    @property
    def pScale(self):
        if plotting.pType == "stagnation":
            return flow.stagnation_pressure
        elif plotting.pType == "cushion":
            return body.Pc if body.Pc > 0.0 else 1.0
        elif plotting.pType == "hydrostatic":
            return flow.density * flow.gravity * self._pScaleHead
        return self._pScale * self._pScalePct


class PathConfig(SubConfig):
    # Directories and file formats
    case_dir = ConfigItem("caseDir", default=".")
    fig_dir_name = ConfigItem("figDirName", default="figures")
    body_dict_dir = ConfigItem("bodyDictDir", default="bodyDict")
    input_dict_dir = ConfigItem("inputDictDir", default="inputDict")
    cushion_dict_dir = ConfigItem(
        "pressureCushionDictDir", "cushionDictDir", default="cushionDict"
    )
    mesh_dir = ConfigItem("meshDir", default="mesh")
    mesh_dict_dir = ConfigItem("meshDictDir", default="meshDict")


class IOConfig(SubConfig):
    data_format = ConfigItem("dataFormat", default="txt")
    write_interval = ConfigItem("writeInterval", default=1)
    write_time_histories = ConfigItem("writeTimeHistories", default=False)
    results_from_file = ConfigItem("resultsFromFile", default=False)


class SolverConfig(SubConfig):
    # Parameters for wetted length solver
    wetted_length_solver = ConfigItem("wettedLengthSolver", default="Secant")
    wetted_length_tol = ConfigItem("wettedLengthTol", default=1e-6)
    wetted_length_relax = ConfigItem("wettedLengthRelax", default=1.0)
    wetted_length_max_it = ConfigItem("wettedLengthMaxIt", default=20)
    wetted_length_max_it_0 = ConfigItem("wettedLengthMaxIt0", default=100)
    wetted_length_max_step_pct = ConfigItem("wettedLengthMaxStepPct", default=0.2)
    _wetted_length_max_step_pct_inc = ConfigItem(
        "wettedLengthMaxStepPctInc", type_=float
    )
    _wetted_length_max_step_pct_dec = ConfigItem(
        "wettedLengthMaxStepPctDec", type_=float
    )
    wetted_length_max_jacobian_reset_step = ConfigItem(
        "wettedLengthMaxJacobianResetStep", default=100
    )

    max_it = ConfigItem("maxIt", default=1)
    num_ramp_it = ConfigItem("rampIt", default=0)
    relax_initial = ConfigItem("relaxI", default=0.01)
    relax_final = ConfigItem("relaxF", default=0.5)
    max_residual = ConfigItem("tolerance", default=1e-6)
    pretension = ConfigItem("pretension", default=0.1)
    relax_FEM = ConfigItem("FEMRelax", "relaxFEM", default=1.0)
    max_FEM_disp = ConfigItem("maxFEMDisp", default=1.0)
    num_damp = ConfigItem("numDamp", default=0.0)

    @property
    def wetted_length_max_step_pct_inc(self):
        if self._wetted_length_max_step_pct_inc is not None:
            return self._wetted_length_max_step_pct_inc
        return self.wetted_length_max_step_pct

    @property
    def wetted_length_max_step_pct_dec(self):
        if self._wetted_length_max_step_pct_dec is not None:
            return self._wetted_length_max_step_pct_dec
        return self.wetted_length_max_step_pct


# Flow-related variables
flow = FlowConfig()
body = BodyConfig()
plotting = PlotConfig()
path = PathConfig()
io = IOConfig()
solver = SolverConfig()


def load_from_file(filename: str) -> None:
    """Load the configuration from a file.

    Args:
        filename: The name of the file.

    """
    logger.info(f"Loading values from {filename}")
    for c in [flow, body, plotting, path, io, solver]:
        c.load_from_dict(filename)


# Load the default config dict file
if Path(DICT_NAME).exists():
    load_from_file(DICT_NAME)

# Initialized constants
ramp = 1.0
has_free_structure = False
it_dir = ""
it = -1

# Use tk by default. Otherwise try Agg. Otherwise, disable plotting.
_fallback_engine = "Agg"
if os.environ.get("DISPLAY") is None:
    matplotlib.use(_fallback_engine)
else:
    try:
        from matplotlib import pyplot
    except ImportError:
        try:
            matplotlib.use(_fallback_engine)
        except ImportError:
            plotting.plot_any = False
