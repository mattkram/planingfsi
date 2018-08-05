"""This module is used to store the global configuration. Values are stored
after reading the configDict file, and values can be accessed by other
packages and modules by importing the config module.

Usage: from planingfsi import config

The global attributes can then be simply accessed via config.attribute_name

"""
import math
from pathlib import Path

from krampy.iotools import Dictionary

from planingfsi import logger

DICT_NAME = 'configDict'

logger.info('Loading {0}'.format(DICT_NAME))

if Path(DICT_NAME).exists():
    config_dict = Dictionary(DICT_NAME)
else:
    config_dict = Dictionary()

config_module_path = Path(__file__).parent
default_dict = Dictionary(from_file=str(config_module_path / 'defaultConfigDict'))


class ConfigItem(object):
    """A descriptor to represent a configuration item.

    Attributes are loaded from a dictionary with fancy default handling.

    """

    def __init__(self, alt_key=None, alt_keys=None, default=None):
        self.name = None
        self.alt_keys = []
        if alt_key:
            self.alt_keys.append(alt_key)
        if alt_keys:
            self.alt_keys.extend(alt_keys)
        self.default = default

    def __get__(self, instance, owner):
        if self.name in instance.__dict__:
            return instance.__dict__[self.name]
        return self.default

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __set_name__(self, _, name):
        self.name = name

    @property
    def keys(self):
        return [self.name] + self.alt_keys

    def get_from_dict(self, dict_):
        """Try to read all keys from the dictionary until a non-None value is found.

        Returns the default value if no appropriate value is found in the dictionary.

        """
        for key in self.keys:
            value = dict_.get(key)
            if value is not None:
                return value
        return self.default


class SubConfig(object):
    """An empty class used simply for dividing the configuration into
    different sections. Also useful in helping define the namespace scopes.
    """

    def __init__(self):
        self.load_from_dict()

    def load_from_dict(self):
        for key, config_item in self.__class__.__dict__.items():
            if isinstance(config_item, ConfigItem):
                value = config_item.get_from_dict(config_dict)
                if value is None:
                    value = config_item.get_from_dict(default_dict)
                setattr(self, key, value)


class FlowConfig(SubConfig):
    density = ConfigItem(alt_key='rho', default=998.2)
    gravity = ConfigItem(alt_key='g', default=9.81)
    kinematic_viscosity = ConfigItem(alt_key='nu', default=1e-6)
    waterline_height = ConfigItem(alt_key='hWL', default=0.0)
    num_dim = ConfigItem(alt_key='dim', default=2)
    include_friction = ConfigItem(alt_key='shearCalc', default=False)

    _froude_num = ConfigItem(alt_key='Fr', default=None)
    _flow_speed = ConfigItem(alt_key='U', default=None)

    @property
    def reference_length(self):
        global body
        return body.reference_length

    @property
    def flow_speed(self):
        """The flow speed is the native variable to store free-stream velocity. However, if Froude
        number is set from input file, that should override the flow speed input.

        """
        if self._froude_num is not None:
            if self._flow_speed is not None:
                raise ValueError('Only one flow speed variable (either Froude number or flow '
                                 'speed) must be set in {0}'.format(DICT_NAME))
            self.froude_num = self._froude_num
        elif self._flow_speed is None:
            raise ValueError('Must specify either U or Fr in {0}'.format(DICT_NAME))
        return self._flow_speed

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
    xCofG = ConfigItem()
    yCofG = ConfigItem()
    _xCofR = ConfigItem()
    _yCofR = ConfigItem()
    mass = ConfigItem(alt_key='m')
    _weight = ConfigItem(alt_key='W')
    reference_length = ConfigItem(alt_keys=['Lref', 'Lc'])

    _cushion_pressure = ConfigItem(alt_key='Pc')
    PcBar = ConfigItem()
    _seal_pressure = ConfigItem(alt_key='Ps')
    PsBar = ConfigItem()

    # Rigid body motion parameters
    time_step = ConfigItem('timeStep')
    relax_rigid_body = ConfigItem('rigidBodyRelax')
    motion_method = ConfigItem('motionMethod')
    motion_jacobian_first_step = ConfigItem('motionJacobianFirstStep')

    bow_seal_tip_load = ConfigItem('bowSealTipLoad')
    tip_constraint_ht = ConfigItem('tipConstraintHt')

    seal_load_pct = ConfigItem('sealLoadPct')
    cushion_force_method = ConfigItem('cushionForceMethod')

    initial_draft = ConfigItem('initialDraft')
    initial_trim = ConfigItem('initialTrim')

    max_draft_step = ConfigItem('maxDraftStep')
    max_trim_step = ConfigItem('maxTrimStep')

    max_draft_acc = ConfigItem('maxDraftAcc')
    max_trim_acc = ConfigItem('maxTrimAcc')

    free_in_draft = ConfigItem('freeInDraft')
    free_in_trim = ConfigItem('freeInTrim')

    draft_damping = ConfigItem('draftDamping')
    trim_damping = ConfigItem('trimDamping')

    _relax_draft = ConfigItem('draftRelax')
    _relax_trim = ConfigItem('trimRelax')

    @property
    def relax_draft(self):
        if self._relax_draft is not None:
            return self._relax_draft
        global body
        return body.relax_rigid_body

    @property
    def relax_trim(self):
        if self._relax_trim is not None:
            return self._relax_trim
        global body
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
        if self._xCofR is not None:
            return self._xCofR
        return self.xCofG

    @property
    def yCofR(self):
        if self._yCofR is not None:
            return self._yCofR
        return self.yCofG

    @property
    def weight(self):
        if self._weight is not None:
            return self._weight
        global flow
        return self.mass * flow.gravity

    @weight.setter
    def weight(self, value):
        global flow
        self.mass = value / flow.gravity


class PlotConfig(SubConfig):
    pType = ConfigItem(alt_key='pScaleType')
    _pScale = ConfigItem(alt_key='pScale')
    _pScaleHead = ConfigItem(alt_key='pScaleHead')
    growth_rate = ConfigItem(alt_key='growthRate')
    CofR_grid_len = ConfigItem(alt_key='CofRGridLen')
    fig_format = ConfigItem(alt_key='figFormat')

    # plotting.pScale = read('pScalePct') / plotting.pScale
    pressure_limiter = ConfigItem('pressureLimiter')

    # Load plot extents
    ext_e = ConfigItem('extE')
    ext_w = ConfigItem('extW')
    ext_n = ConfigItem('extN')
    ext_s = ConfigItem('extS')

    xmin = ConfigItem('plotXMin')
    xmax = ConfigItem('plotXMax')
    ymin = ConfigItem('plotYMin')
    ymax = ConfigItem('plotYMax')

    lambda_min = ConfigItem('lamMin')
    lambda_max = ConfigItem('lamMax')

    _x_fs_min = ConfigItem('xFSMin')
    _x_fs_max = ConfigItem('xFSMax')

    # Whether to save, show, or watch plots
    save = ConfigItem('plotSave')
    show_pressure = ConfigItem('plotPressure')
    show = ConfigItem('plotShow')
    _watch = ConfigItem('plotWatch')

    @property
    def watch(self):
        return self._watch or self.show

    @property
    def plot_any(self):
        return self.show or self.save or self.watch or self.show_pressure

    @property
    def x_fs_min(self):
        if self._x_fs_min is not None:
            return self._x_fs_min
        if self.xmin is not None:
            return self.xmin
        global flow
        return self.lambda_min * flow.lam

    @property
    def x_fs_max(self):
        if self._x_fs_max is not None:
            return self._x_fs_max
        if self.xmax is not None:
            return self.xmax
        global flow
        return self.lambda_max * flow.lam

    @property
    def pScale(self):
        global flow
        global body
        if plotting.pType == 'stagnation':
            return flow.stagnation_pressure
        elif plotting.pType == 'cushion':
            return body.Pc if body.Pc > 0.0 else 1.0
        elif plotting.pType == 'hydrostatic':
            return flow.density * flow.gravity * self._pScaleHead
        return self._pScale


class PathConfig(SubConfig):
    # Directories and file formats
    case_dir = ConfigItem('caseDir')
    fig_dir_name = ConfigItem('figDirName')
    body_dict_dir = ConfigItem('bodyDictDir')
    input_dict_dir = ConfigItem('inputDictDir')
    cushion_dict_dir = ConfigItem(alt_keys=['pressureCushionDictDir', 'cushionDictDir'])
    mesh_dir = ConfigItem('meshDir')
    mesh_dict_dir = ConfigItem('meshDictDir')


class IOConfig(SubConfig):
    data_format = ConfigItem('dataFormat')
    write_interval = ConfigItem('writeInterval')
    write_time_histories = ConfigItem('writeTimeHistories')
    results_from_file = ConfigItem('resultsFromFile')


class SolverConfig(SubConfig):
    # Parameters for wetted length solver
    wetted_length_solver = ConfigItem('wettedLengthSolver')
    wetted_length_tol = ConfigItem('wettedLengthTol')
    wetted_length_relax = ConfigItem('wettedLengthRelax')
    wetted_length_max_it = ConfigItem('wettedLengthMaxIt')
    wetted_length_max_it_0 = ConfigItem('wettedLengthMaxIt0')
    wetted_length_max_step_pct = ConfigItem('wettedLengthMaxStepPct')
    _wetted_length_max_step_pct_inc = ConfigItem('wettedLengthMaxStepPctInc')
    _wetted_length_max_step_pct_dec = ConfigItem('wettedLengthMaxStepPctDec')
    wetted_length_max_jacobian_reset_step = ConfigItem('wettedLengthMaxJacobianResetStep')

    max_it = ConfigItem('maxIt')
    num_ramp_it = ConfigItem('rampIt')
    relax_initial = ConfigItem('relaxI')
    relax_final = ConfigItem('relaxF')
    max_residual = ConfigItem('tolerance')
    pretension = ConfigItem('pretension')
    relax_FEM = ConfigItem('FEMRelax')
    max_FEM_disp = ConfigItem('maxFEMDisp')
    num_damp = ConfigItem('numDamp')

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

# Initialized constants
ramp = 1.0
has_free_structure = False
it_dir = ''
it = -1

del config_dict
del default_dict
