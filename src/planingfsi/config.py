"""This module is used to store the global configuration. Values are stored
after reading the configDict file, and values can be accessed by other
packages and modules by importing the config module.

Usage: from planingfsi import config

The global attributes can then be simply accessed via config.attribute_name

"""
import math
from pathlib import Path
from typing import Any, List, Type, Dict, Optional, Union

from . import logger
from .dictionary import load_dict_from_file

DICT_NAME = "configDict"


class ConfigItem:
    """A descriptor to represent a configuration item.

    Attributes are loaded from a dictionary with fancy default handling.

    """

    name: str
    default: Any

    def __init__(self, *alt_keys: str, **kwargs: Any):
        self.alt_keys: List[str] = list(alt_keys)
        self.type_: Optional[Type] = kwargs.get("type")
        try:
            self.default = kwargs["default"]
            if self.default is not None:
                self.type_ = type(self.default)
        except KeyError:
            # If default not specified, leave undefined. Handled in __get__.
            # Allows None to be the default value.
            pass

    def __get__(self, instance: "SubConfig", _: Any) -> Any:
        """Retrieve the value from the instance dictionary or return the default.

        Raises:
            AttributeError: If the attribute hasn't been set and the  default value isn't specified.

        """
        if self.name in instance.__dict__:
            return instance.__dict__[self.name]
        try:
            return self.default
        except AttributeError:
            raise AttributeError(f"Attribute {self.name} has not been set and no default specified")

    def __set__(self, instance: "SubConfig", value: Any) -> None:
        """When the value is set, try to convert it and then store it in the instance dictionary."""
        if value is not None and self.type_ is not None:
            if self.type_ == bool and isinstance(value, str):
                value = value.lower() == "true"
            value = self.type_(value)
        instance.__dict__[self.name] = value

    def __set_name__(self, _: Any, name: str) -> None:
        """Set the name when the ConfigItem is defined."""
        self.name = name

    @property
    def keys(self) -> List[str]:
        """A list of keys to look for when reading in the value."""
        return [self.name] + self.alt_keys

    def get_from_dict(self, dict_: Dict[str, Any]) -> Any:
        """Try to read all keys from the dictionary until a non-None value is found.

        Returns the default value if no appropriate value is found in the dictionary.

        """
        for key in self.keys:
            value = dict_.get(key)
            if value is not None:
                return value
        raise KeyError('None of the following keys "{}" found in dictionary.'.format(self.keys))


class SubConfig:
    """An empty class used simply for dividing the configuration into
    different sections. Also useful in helping define the namespace scopes.
    """

    def load_from_file(self, filename: Union[Path, str]) -> None:
        """Load the configuration from a dictionary file.

        Args:
            filename: The path to the dictionary file.

        """
        # Clear the attributes before loading the new ones
        self.__dict__ = {}
        dict_ = load_dict_from_file(filename)
        for key, config_item in self.__class__.__dict__.items():
            if isinstance(config_item, ConfigItem):
                try:
                    value = config_item.get_from_dict(dict_)
                except KeyError:
                    pass
                else:
                    setattr(self, config_item.name, value)


class FlowConfig(SubConfig):
    """Configuration related to the fluid dynamics problem.

    Attributes:
        density (float): Mass density of the fluid.
        gravity (float): Acceleration due to gravity.
        kinematic_viscosity (float): Kinematic viscosity of the fluid.
        waterline_height (float): Height of the waterline above the reference.
        num_dim (int): Number of dimensions.
        include_friction (bool): If True, include a flat-plate estimation for the frictional drag
            component.

    """

    density = ConfigItem("rho", default=998.2)
    gravity = ConfigItem("g", default=9.81)
    kinematic_viscosity = ConfigItem("nu", default=1e-6)
    waterline_height = ConfigItem("hWL", default=0.0)
    num_dim = ConfigItem("dim", default=2)
    include_friction = ConfigItem("shearCalc", default=False)

    _froude_num = ConfigItem("Fr", default=None, type=float)
    _flow_speed = ConfigItem("U", default=None, type=float)

    @property
    def reference_length(self) -> float:
        """float: Reference length used for potential-flow solver.

        Defaults to reference length of the rigid body.

        """
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
    def flow_speed(self, value: float) -> None:
        """Set the raw flow speed variable and ensure raw Froude number is not also set."""
        self._flow_speed = value
        self._froude_num = None

    @property
    def froude_num(self) -> float:
        """float: The Froude number is the non-dimensional speed."""
        return self.flow_speed / math.sqrt(self.gravity * self.reference_length)

    @froude_num.setter
    def froude_num(self, value: float) -> None:
        self.flow_speed = value * math.sqrt(self.gravity * self.reference_length)

    @property
    def stagnation_pressure(self) -> float:
        """float: The pressure at the stagnation point."""
        return 0.5 * self.density * self.flow_speed ** 2

    @property
    def k0(self) -> float:
        """float: A wave number used internally in the potential-flow solver."""
        return self.gravity / self.flow_speed ** 2

    @property
    def lam(self) -> float:
        """float: A wavelength used internally in the potential-flow solver."""
        return 2 * math.pi / self.k0


class BodyConfig(SubConfig):
    """Configuration for the rigid body.

    Attributes:
        xCofG (float): x-coordinate of the center of gravity.
        yCofG (float): y-coordinate of the center of gravity.
        mass (float): Mass of the rigid body.
        reference_length (float): Reference length of the body.
        time_step (float): Time step to use when solving time-domain rigid body motion.
        relax_rigid_body (float): Under-relaxation factor for static rigid body motion solver.
        motion_method (str): Motion method to use for rigid body solver.
        motion_jacobian_first_step (float): Step length for first step in Jacobian calculation.
        bow_seal_tip_load (float): Fixed load to apply to bow seal tip.
        tip_constraint_ht (float): Height of constraint for seal tip.
        seal_load_pct (float): Contribution of seals as percentage of total weight.
        cushion_force_method (str): Method to use for cushion force calculation.
        initial_draft (float): Initial draft to use in motion solver.
        initial_trim (float): Initial trim to use in motion solver.
        max_draft_step (float): Maximum step to use when solving for draft.
        max_trim_step (float): Maximum step to use when solving for trim.
        max_draft_acc (float): Maximum step in acceleration to use when solving for draft.
        max_trim_acc (float): Maximum step in acceleration to use when solving for trim.
        free_in_draft (bool): If True, body will be free in draft.
        free_in_trim (bool): If True, body will be free in trim.
        draft_damping (float): Damping value to use when solving for draft.
        trim_damping (float): Damping value to use when solving for trim.

    """

    xCofG = ConfigItem(default=0.0)
    yCofG = ConfigItem(default=0.0)
    _xCofR = ConfigItem(type=float)
    _yCofR = ConfigItem(type=float)

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
    tip_constraint_ht = ConfigItem("tipConstraintHt", type=float)

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

    _relax_draft = ConfigItem("draftRelax", type=float)
    _relax_trim = ConfigItem("trimRelax", type=float)

    @property
    def relax_draft(self) -> float:
        """float: Under-relaxation factor to apply to the draft solver.

        Defaults to rigid body under-relaxation factor.

        """
        try:
            return self._relax_draft
        except AttributeError:
            return body.relax_rigid_body

    @property
    def relax_trim(self) -> float:
        """float: Under-relaxation factor to apply to the trim solver.

        Defaults to rigid body under-relaxation factor.

        """
        try:
            return self._relax_trim
        except AttributeError:
            return body.relax_rigid_body

    @property
    def Pc(self) -> float:
        """float: Alias for cushion pressure."""
        return self._cushion_pressure

    @property
    def PcBar(self) -> float:
        """float: Non-dimensional cushion pressure."""
        return self._cushion_pressure * self.reference_length / self.weight

    @PcBar.setter
    def PcBar(self, value: float) -> None:
        self._cushion_pressure = value * self.weight / self.reference_length

    @property
    def Ps(self) -> float:
        """float: Pressure inside the seal."""
        return self._seal_pressure

    @property
    def PsBar(self) -> float:
        """float: Non-dimensional seal pressure as ratio of cushion pressure."""
        if self._cushion_pressure == 0.0:
            return 0.0
        return self._seal_pressure / self._cushion_pressure

    @PsBar.setter
    def PsBar(self, value: float) -> None:
        self._seal_pressure = value * self._cushion_pressure

    @property
    def xCofR(self) -> float:
        """float: x-coordinate of the center of rotation. Defaults to center of gravity."""
        try:
            return self._xCofR
        except AttributeError:
            return self.xCofG

    @property
    def yCofR(self) -> float:
        """float: y-coordinate of the center of rotation. Defaults to center of gravity."""
        try:
            return self._yCofR
        except AttributeError:
            return self.yCofG

    @property
    def weight(self) -> float:
        """float: Weight of the body. Defaults to mass times gravity."""
        try:
            return self._weight
        except AttributeError:
            return self.mass * flow.gravity

    @weight.setter
    def weight(self, value: float) -> None:
        self.mass = value / flow.gravity


class PlotConfig(SubConfig):
    """Configuration for plotting of the response.

    Attributes:
        pType (str): Selected method by which to scale pressure lines.
        growth_rate (float): Rate at which to grow points when plotting free surface.
        CofR_grid_len (float): Length of grid for plotting center-of-rotation.
        fig_format (str): Format to save figures in.
        pressure_limiter (bool): If True, limit the pressure bar length.
        save (bool): If True, figures will be saved.
        show (bool): If True, show the plot window while running the solver.
        show_pressure (bool): If True, show the pressure profile lines.

    """

    pType = ConfigItem("pScaleType", default="stagnation")
    _pScale = ConfigItem("pScale", default=1.0)
    _pScalePct = ConfigItem("pScalePct", default=1.0)
    _pScaleHead = ConfigItem("pScaleHead", default=1.0)
    growth_rate = ConfigItem("growthRate", default=1.1)
    CofR_grid_len = ConfigItem("CofRGridLen", default=0.5)
    fig_format = ConfigItem("figFormat", default="png")

    pressure_limiter = ConfigItem("pressureLimiter", default=False)

    # Load plot extents
    ext_e = ConfigItem("extE", default=0.1)
    ext_w = ConfigItem("extW", default=0.1)
    ext_n = ConfigItem("extN", default=0.1)
    ext_s = ConfigItem("extS", default=0.1)

    xmin = ConfigItem("plotXMin", type=float)
    xmax = ConfigItem("plotXMax", type=float)
    ymin = ConfigItem("plotYMin", type=float)
    ymax = ConfigItem("plotYMax", type=float)

    lambda_min = ConfigItem("lamMin", default=-1.0)
    lambda_max = ConfigItem("lamMax", default=1.0)

    _x_fs_min = ConfigItem("xFSMin", type=float)
    _x_fs_max = ConfigItem("xFSMax", type=float)

    # Whether to save, show, or watch plots
    save = ConfigItem("plotSave", default=False)
    show_pressure = ConfigItem("plotPressure", default=False)
    show = ConfigItem("plotShow", default=False)
    _watch = ConfigItem("plotWatch", default=False)

    @property
    def watch(self) -> bool:
        """bool: If True, watch the plot figure."""
        return self._watch or self.show

    @property
    def plot_any(self) -> bool:
        """bool: If True, plot will be generated, otherwise skip to save compute time."""
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
    def x_fs_min(self) -> float:
        """float: Minimum x-location to use for plotting free surface."""
        if self._x_fs_min is not None:
            return self._x_fs_min
        if self.xmin is not None:
            return self.xmin
        return self.lambda_min * flow.lam

    @property
    def x_fs_max(self) -> float:
        """float: Maximum x-location to use for plotting free surface."""
        if self._x_fs_max is not None:
            return self._x_fs_max
        if self.xmax is not None:
            return self.xmax
        return self.lambda_max * flow.lam

    @property
    def pScale(self) -> float:
        """float: Pressure value to use to scale the pressure profile."""
        if plotting.pType == "stagnation":
            pScale = flow.stagnation_pressure
        elif plotting.pType == "cushion":
            pScale = body.Pc if body.Pc > 0.0 else 1.0
        elif plotting.pType == "hydrostatic":
            pScale = flow.density * flow.gravity * self._pScaleHead
        else:
            pScale = self._pScale
        return pScale * self._pScalePct


class PathConfig(SubConfig):
    """Directories and file formats.

    Attributes:
        case_dir (str): Path to the case directory.
        fig_dir_name (str): Path to the figures directory, relative to the case directory.
        body_dict_dir (str): Path to the body dict directory, relative to the case directory.
        input_dict_dir (str): Path to the input dict directory, relative to the case directory.
        cushion_dict_dir (str): Path to the cushion dict directory, relative to the case directory.
        mesh_dir (str): Path to the mesh directory, relative to the case directory.
        mesh_dict_dir (str): Path to the mesh dictionary file, relative to the case directory.

    Todo:
        * Verify these attribute descriptions.

    """

    case_dir = ConfigItem("caseDir", default=".")
    fig_dir_name = ConfigItem("figDirName", default="figures")
    body_dict_dir = ConfigItem("bodyDictDir", default="bodyDict")
    input_dict_dir = ConfigItem("inputDictDir", default="inputDict")
    cushion_dict_dir = ConfigItem("pressureCushionDictDir", "cushionDictDir", default="cushionDict")
    mesh_dir = ConfigItem("meshDir", default="mesh")
    mesh_dict_dir = ConfigItem("meshDictDir", default="meshDict")


class IOConfig(SubConfig):
    """Configuration for file I/O.

    Attributes:
        data_format (str): Format of text files to save data in.
        write_interval (int): Interval in iterations for which to write result files.
        write_time_histories (bool): If True, time histories of motion will be written to files.
        results_from_file (bool): If True, load the results from previously-saved files.

    """

    data_format = ConfigItem("dataFormat", default="txt")
    write_interval = ConfigItem("writeInterval", default=1)
    write_time_histories = ConfigItem("writeTimeHistories", default=False)
    results_from_file = ConfigItem("resultsFromFile", default=False)


class SolverConfig(SubConfig):
    """Parameters for solvers.

    Attributes:
        wetted_length_solver (str): Chosen solver to use for wetted length.
        wetted_length_tol (float): Tolerance to use for wetted-length solver.
        wetted_length_relax (float): Under-relaxation factor to use for wetted-length solver.
        wetted_length_max_it (int): Maximum number of iterations to use for wetted-length solver.
        wetted_length_max_it_0 (int): Maximum number of iterations for wetted-length solver in first
            rigid body iteration.
        wetted_length_max_step_pct (float): Maximum allowable change in wetted length as fraction of
            wetted length.


    """

    wetted_length_solver = ConfigItem("wettedLengthSolver", default="Secant")
    wetted_length_tol = ConfigItem("wettedLengthTol", default=1e-6)
    wetted_length_relax = ConfigItem("wettedLengthRelax", default=1.0)
    wetted_length_max_it = ConfigItem("wettedLengthMaxIt", default=20)
    wetted_length_max_it_0 = ConfigItem("wettedLengthMaxIt0", default=100)
    wetted_length_max_step_pct = ConfigItem("wettedLengthMaxStepPct", default=0.2)
    _wetted_length_max_step_pct_inc = ConfigItem("wettedLengthMaxStepPctInc", type=float)
    _wetted_length_max_step_pct_dec = ConfigItem("wettedLengthMaxStepPctDec", type=float)
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
    def wetted_length_max_step_pct_inc(self) -> float:
        """float: Maximum allowable increase in wetted length as fraction of wetted length."""
        try:
            return self._wetted_length_max_step_pct_inc
        except AttributeError:
            return self.wetted_length_max_step_pct

    @property
    def wetted_length_max_step_pct_dec(self) -> float:
        """float: Maximum allowable decrease in wetted length as fraction of wetted length."""
        try:
            return self._wetted_length_max_step_pct_dec
        except AttributeError:
            return self.wetted_length_max_step_pct


# Create instances of each class and store on module
flow = FlowConfig()
body = BodyConfig()
plotting = PlotConfig()
path = PathConfig()
io = IOConfig()
solver = SolverConfig()


def load_from_file(filename: Union[Path, str]) -> None:
    """Load the configuration from a file.

    Args:
        filename: The name of the file.

    """
    logger.info(f"Loading values from {filename}")
    for c in [flow, body, plotting, path, io, solver]:
        c.load_from_file(filename)


# Load the default config dict file
if Path(DICT_NAME).exists():
    load_from_file(DICT_NAME)

# Initialized constants
# TODO: These should be moved to the Simulation class
ramp = 1.0
has_free_structure = False

# TODO: This need to be factored out eventually
res_l = 1.0
res_m = 1.0
