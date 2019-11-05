"""This module is used to store the global configuration. Values are stored
after reading the configDict file, and values can be accessed by other
packages and modules by importing the config module.

Usage: from planingfsi import config

The global attributes can then be simply accessed via ``config.attribute_name``.

"""
import math

from . import logger
from .dictionary import load_dict_from_file


class ConfigItem(object):
    """A configuration item with type conversion, multiple key support, and default value."""

    def __init__(self, alt_key=None, alt_keys=None, default=None, type_=None):
        self.name = None
        self.alt_keys = []
        if alt_key:
            self.alt_keys.append(alt_key)
        if alt_keys:
            self.alt_keys.extend(alt_keys)
        self.default = default
        self.type_ = type_ if type_ else type(default)

    def __get__(self, instance, owner):
        """Retrieve the value from the instance dictionary. If it doesn't exist, return the default."""
        try:
            return instance.__dict__[self.name]
        except KeyError:
            return self.default

    def __set__(self, instance, value):
        """When the value is set, try to convert it and then store it in the instance dictionary."""
        if value is not None and self.type_ is not None:
            value = self.type_(value)
        instance.__dict__[self.name] = value

    def __set_name__(self, _, name):
        """Store the name when instantiating a new ConfigItem."""
        self.name = name

    @property
    def keys(self):
        """A list of keys to look for when reading in the value from a dictionary."""
        return [self.name] + self.alt_keys

    def load_value_from_dict(self, dict_):
        """Try to read all keys from the dictionary until a non-None value is found.

        Returns the default value if no appropriate value is found in the dictionary.

        """
        for key in self.keys:
            value = dict_.get(key)
            if value is not None:
                return value
        raise KeyError(f'None of the following keys "{self.keys}" found in dictionary.')


class SubConfig(object):
    """An empty class used simply for dividing the configuration into
    different sections. Also useful in helping define the namespace scopes.
    """

    def load_from_dict_file(self, dict_name: str):
        """Load the configuration from a dictionary file.

        Args:
            dict_name: The path to the dictionary file.

        """
        dict_ = load_dict_from_file(dict_name)
        for key, item in self.__class__.__dict__.items():
            if isinstance(item, ConfigItem):
                try:
                    value = item.load_value_from_dict(dict_)
                except KeyError:
                    pass
                else:
                    setattr(self, item.name, value)


class FlowConfig(SubConfig):
    """Configuration related to the fluid dynamics problem.

    Attributes:
        density (float): Mass density of the fluid.
        gravity (float): Acceleration due to gravity.
        kinematic_viscosity (float): Kinematic viscosity of the fluid.
        waterline_height (float): Height of the waterline above the reference.
        num_dim (int): Number of dimensions.
        include_friction (bool): If True, include a flat-plate estimation for the frictional drag component.

    """

    density = ConfigItem(alt_key="rho", default=998.2)
    gravity = ConfigItem(alt_key="g", default=9.81)
    kinematic_viscosity = ConfigItem(alt_key="nu", default=1e-6)
    waterline_height = ConfigItem(alt_key="hWL", default=0.0)
    num_dim = ConfigItem(alt_key="dim", default=2)
    include_friction = ConfigItem(alt_key="shearCalc", default=False)

    _froude_num = ConfigItem(alt_key="Fr", default=None, type_=float)
    _flow_speed = ConfigItem(alt_key="U", default=None, type_=float)

    @property
    def reference_length(self):
        """float: Reference length used for potential-flow solver.

        Defaults to reference length of the rigid body.

        """
        return body.reference_length

    @property
    def froude_num(self):
        """float: The Froude number is the non-dimensional speed."""
        return self.flow_speed / math.sqrt(self.gravity * self.reference_length)

    @froude_num.setter
    def froude_num(self, value):
        # noinspection PyAttributeOutsideInit
        self.flow_speed = value * math.sqrt(self.gravity * self.reference_length)
        self._froude_num = (
            None
        )  # Only store _froude_num on reading, then use _flow_speed

    @property
    def flow_speed(self):
        """float: The flow speed is the native variable to store free-stream velocity. However, if Froude
        number is set from input file, that should override the flow speed input.

        """
        if self._froude_num is not None and self._flow_speed is not None:
            raise ValueError(
                "Only one flow speed variable (either Froude number or flow speed) can be set."
            )

        if self._froude_num is not None:
            # Set the self._flow_speed via self.froud_num property
            # noinspection PyAttributeOutsideInit
            self.froude_num = self._froude_num

        if self._flow_speed is None:
            raise ValueError(f"Must specify either U or Fr")
        return self._flow_speed

    @flow_speed.setter
    def flow_speed(self, value):
        """Set the raw flow speed variable and ensure raw Froude number is not also set."""
        self._flow_speed = value

    @property
    def stagnation_pressure(self):
        """float: The pressure at the stagnation point."""
        return 0.5 * self.density * self.flow_speed ** 2

    @property
    def k0(self):
        """float: A wave number used internally in the potential-flow solver."""
        return self.gravity / self.flow_speed ** 2

    @property
    def lam(self):
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
        relax_rigid_body (float): Under-relaxation factor to use for static rigid body motion solver.
        motion_method (str): Motion method to use for rigid body solver.
        motion_jacobian_first_step (float): Step length to use for first step in Jacobian calculation.
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
    _xCofR = ConfigItem(type_=float)
    _yCofR = ConfigItem(type_=float)

    mass = ConfigItem(alt_key="m", default=1.0)
    _weight = ConfigItem(alt_key="W")

    reference_length = ConfigItem(alt_keys=["Lref", "Lc"], default=1.0)

    _cushion_pressure = ConfigItem(alt_key="Pc", default=0.0)
    _seal_pressure = ConfigItem(alt_key="Ps", default=0.0)

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
        """float: Under-relaxation factor to apply to the draft solver.

        Defaults to rigid body under-relaxation factor.

        """
        if self._relax_draft is not None:
            return self._relax_draft
        return self.relax_rigid_body

    @property
    def relax_trim(self):
        """float: Under-relaxation factor to apply to the trim solver.

        Defaults to rigid body under-relaxation factor.

        """
        if self._relax_trim is not None:
            return self._relax_trim
        return self.relax_rigid_body

    @property
    def Pc(self):
        """float: Alias for cushion pressure."""
        return self._cushion_pressure

    @property
    def PcBar(self):
        """float: Non-dimensional cushion pressure."""
        return self._cushion_pressure * self.reference_length / self.weight

    @PcBar.setter
    def PcBar(self, value):
        self._cushion_pressure = value * self.weight / self.reference_length

    @property
    def Ps(self):
        """float: Pressure inside the seal."""
        return self._seal_pressure

    @property
    def PsBar(self):
        """float: Non-dimensional seal pressure as ratio of cushion pressure."""
        if self._cushion_pressure == 0.0:
            return 0.0
        return self._seal_pressure / self._cushion_pressure

    @PsBar.setter
    def PsBar(self, value):
        self._seal_pressure = value * self._cushion_pressure

    @property
    def xCofR(self):
        """float: x-coordinate of the center of rotation. Defaults to center of gravity."""
        if self._xCofR is not None:
            return self._xCofR
        return self.xCofG

    @property
    def yCofR(self):
        """float: y-coordinate of the center of rotation. Defaults to center of gravity."""
        if self._yCofR is not None:
            return self._yCofR
        return self.yCofG

    @property
    def weight(self):
        """float: Weight of the body. Defaults to mass times gravity."""
        if self._weight is not None:
            return self._weight
        return self.mass * flow.gravity

    @weight.setter
    def weight(self, value):
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

    pType = ConfigItem(alt_key="pScaleType", default="stagnation")
    _pScale = ConfigItem(alt_key="pScale", default=1.0)
    _pScalePct = ConfigItem(alt_key="pScalePct", default=1.0)
    _pScaleHead = ConfigItem(alt_key="pScaleHead", default=1.0)
    growth_rate = ConfigItem(alt_key="growthRate", default=1.1)
    CofR_grid_len = ConfigItem(alt_key="CofRGridLen", default=0.5)
    fig_format = ConfigItem(alt_key="figFormat", default="eps")

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
        """bool: If True, watch the plot figure."""
        return self._watch or self.show

    @property
    def plot_any(self):
        """bool: If True, plot will be generated, otherwise skip to save compute time."""
        return self.show or self.save or self.watch or self.show_pressure

    @property
    def x_fs_min(self):
        """float: Minimum x-location to use for plotting free surface."""
        if self._x_fs_min is not None:
            return self._x_fs_min
        if self.xmin is not None:
            return self.xmin
        return self.lambda_min * flow.lam

    @property
    def x_fs_max(self):
        """float: Maximum x-location to use for plotting free surface."""
        if self._x_fs_max is not None:
            return self._x_fs_max
        if self.xmax is not None:
            return self.xmax
        return self.lambda_max * flow.lam

    @property
    def pScale(self):
        """float: Pressure value to use to scale the pressure profile."""
        if plotting.pType == "stagnation":
            return flow.stagnation_pressure
        elif plotting.pType == "cushion":
            return body.Pc if body.Pc > 0.0 else 1.0
        elif plotting.pType == "hydrostatic":
            return flow.density * flow.gravity * self._pScaleHead
        return self._pScale * self._pScalePct


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
    cushion_dict_dir = ConfigItem(
        alt_keys=["pressureCushionDictDir", "cushionDictDir"], default="cushionDict"
    )
    mesh_dir = ConfigItem("meshDir", default="mesh")
    mesh_dict_dir = ConfigItem("meshDictDir", default="meshDict")


class IOConfig(SubConfig):
    """Configuration for file I/O.

    Attributes:
        data_format (str): Format of text files to save data in.
        write_interval (int): Interval in iterations for which to write result files.
        write_time_histories (bool): If True, time histories of rigid-body motion will be written to files.
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
        wetted_length_max_it_0 (int): Maximum number of iterations to use for wetted-length solver in first rigid body
            iteration.
        wetted_length_max_step_pct (float): Maximum allowable change in wetted length as fraction of wetted length.


    """

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
    relax_FEM = ConfigItem(alt_keys=["FEMRelax", "relaxFEM"], default=1.0)
    max_FEM_disp = ConfigItem("maxFEMDisp", default=1.0)
    num_damp = ConfigItem("numDamp", default=0.0)

    @property
    def wetted_length_max_step_pct_inc(self):
        """float: Maximum allowable increase in wetted length as fraction of wetted length."""
        if self._wetted_length_max_step_pct_inc is not None:
            return self._wetted_length_max_step_pct_inc
        return self.wetted_length_max_step_pct

    @property
    def wetted_length_max_step_pct_dec(self):
        """float: Maximum allowable decrease in wetted length as fraction of wetted length."""
        if self._wetted_length_max_step_pct_dec is not None:
            return self._wetted_length_max_step_pct_dec
        return self.wetted_length_max_step_pct


# Create instances of each class and store on module
flow = FlowConfig()
body = BodyConfig()
plotting = PlotConfig()
path = PathConfig()
io = IOConfig()
solver = SolverConfig()


def load_from_dict_file(filename: str):
    """Load all SubConfig's from a dictionary file.

    Args:
        filename: Path to the dictionary file.

    """
    logger.info(f"Loading configuration from {filename}")
    for c in [flow, body, plotting, path, io, solver]:
        c.load_from_dict_file(filename)


# Initialized constants
# TODO: These globals should refactored
ramp = 1.0
has_free_structure = False
it_dir = ""
it = -1
