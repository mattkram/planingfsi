"""This module is used to store the global configuration. Values are stored
after reading the configDict file, and values can be accessed by other
packages and modules by importing the config module.

Usage: from planingfsi import config

The global attributes can then be simply accessed via config.attribute_name

"""
import os
import math

import planingfsi.io

dict_name = 'configDict'

print('Loading {0}'.format(dict_name))

if os.path.exists(dict_name):
    config_dict = planingfsi.io.Dictionary(dict_name)
else:
    config_dict = planingfsi.io.Dictionary()

config_module_path = os.path.abspath(os.path.dirname(__file__))
default_dict = planingfsi.io.Dictionary(os.path.join(config_module_path, 'defaultConfigDict'))


# Function to read value from dictionary or default dictionary
def read(key, default=None):
    return config_dict.read(key, default_dict.read(key, default))


class Subconfig(object):
    """An empty class used simply for dividing the configuration into
    different sections. Also useful in helping define the namespace scopes.
    """
    pass


# Create subconfigs, used for sorting
flow = Subconfig() # Flow-related variables
body = Subconfig() # Related to rigid body
path = Subconfig() # File paths, extensions, etc.
plotting = Subconfig() # Related to plotting
io = Subconfig()
solver = Subconfig()

# Load run properties from dictionary
flow.density = read('rho')
flow.gravity = read('g')
flow.kinematic_viscosity = read('nu')
flow.waterline_height = read('hWL')

flow.flow_speed = read('U')
flow.froude_num = read('Fr')

body.xCofG = read('xCofG')
body.yCofG = read('yCofG')

body.xCofR = read('xCofR', default=body.xCofG)
body.yCofR = read('yCofR', default=body.yCofG)

body.mass = read('m')
body.weight = read('W', default=body.mass * flow.gravity)

body.reference_length = read('Lref', default=read('Lc', default=1.0))  # read Lc for backwards-compatibility

flow.num_dim = read('dim')
flow.include_friction = read('shearCalc')

# Calculate U or Fr depending on which was specified in the file
if flow.flow_speed is not None:
    flow.froude_num = flow.flow_speed / math.sqrt(flow.gravity * body.reference_length)
elif flow.froude_num is not None:
    flow.flow_speed = flow.froude_num * math.sqrt(flow.gravity * body.reference_length)
else:
    raise NameError(
        'Must specify either U or Fr in {0}'.format(dict_name))

flow.stagnation_pressure = 0.5 * flow.density * flow.flow_speed**2
flow.k0 = flow.gravity / flow.flow_speed**2
flow.lam = 2 * math.pi / flow.k0

# Calculate Pc or Pcbar depending on which was specified in the file
body.Pc = read('Pc')
body.PcBar = read('PcBar')
if body.PcBar is not None:
    body.Pc = body.PcBar * body.weight / body.reference_length
else:
    body.PcBar = body.Pc * body.reference_length / body.weight

body.Ps = read('Ps')
body.PsBar = read('PsBar')
if body.PsBar is not None:
    body.Ps = body.PsBar * body.Pc
elif body.Pc == 0.0:
    body.PsBar = 0.0
else:
    body.PsBar = body.Ps / body.Pc

# Set pressure scale for plotting purposes
plotting.pType = read('pScaleType')
if plotting.pType == 'stagnation':
    plotting.pScale = flow.stagnation_pressure
elif plotting.pType == 'cushion':
    plotting.pScale = body.Pc if body.Pc > 0.0 else 1.0
elif plotting.pType == 'hydrostatic':
    plotting.pScale = flow.density * flow.gravity * read('pScaleHead')
else:
    plotting.pScale = read('pScale', default=1.0)

plotting.growth_rate = read('growthRate')
plotting.CofR_grid_len = read('CofRGridLen')

# Directories and file formats
path.case_dir = read('caseDir')
io.data_format = read('dataFormat')
plotting.fig_format = read('figFormat')
path.fig_dir_name = read('figDirName')
path.body_dict_dir = read('bodyDictDir')
path.input_dict_dir = read('inputDictDir')
path.cushion_dict_dir = read(
    'pressureCushionDictDir', default=read('cushionDictDir'))
path.mesh_dir = read('meshDir')
path.mesh_dict_dir = read('meshDictDir')

plotting.pScale = read('pScalePct') / plotting.pScale
plotting.pressure_limiter = read('pressureLimiter')

# Load plot extents
plotting.ext_e = read('extE')
plotting.ext_w = read('extW')
plotting.ext_n = read('extN')
plotting.ext_s = read('extS')

plotting.xmin = read('plotXMin')
plotting.xmax = read('plotXMax')
plotting.ymin = read('plotYMin')
plotting.ymax = read('plotYMax')

plotting.lambda_min = read('lamMin')
plotting.lambda_max = read('lamMax')

plotting.x_fs_min = read(
    'xFSMin', default=plotting.xmin if plotting.xmin is not None else plotting.lambda_min * flow.lam)
plotting.x_fs_max = read(
    'xFSMax', default=plotting.xmax if plotting.xmax is not None else plotting.lambda_max * flow.lam)

# Whether to save, show, or watch plots
plotting.save = read('plotSave')
plotting.show_pressure = read('plotPressure')
plotting.show = read('plotShow')
plotting.watch = read('plotWatch') or plotting.show
plotting.plot_any = plotting.show or plotting.save or plotting.watch or plotting.show_pressure

# File IO settings
io.write_interval = read('writeInterval')
io.write_time_histories = read('writeTimeHistories')
io.results_from_file = read('resultsFromFile')

# Rigid body motion parameters
body.time_step = read('timeStep')
body.relax_rigid_body = read('rigidBodyRelax')
body.motion_method = read('motionMethod')
body.motion_jacobian_first_step = read('motionJacobianFirstStep')

body.bow_seal_tip_load = read('bowSealTipLoad')
body.tip_constraint_ht = read('tipConstraintHt')

body.seal_load_pct = read('sealLoadPct')
body.cushion_force_method = read('cushionForceMethod')

body.initial_draft = read('initialDraft')
body.initial_trim = read('initialTrim')

body.max_draft_step = read('maxDraftStep')
body.max_trim_step = read('maxTrimStep')

body.max_draft_acc = read('maxDraftAcc')
body.max_trim_acc = read('maxTrimAcc')

body.free_in_draft = read('freeInDraft')
body.free_in_trim = read('freeInTrim')

body.draft_damping = read('draftDamping')
body.trim_damping = read('trimDamping')

body.relax_draft = read('draftRelax', default=body.relax_rigid_body)
body.relax_trim = read('trimRelax', default=body.relax_rigid_body)

# Parameters for wetted length solver
solver.wetted_length_solver = read('wettedLengthSolver')
solver.wetted_length_tol = read('wettedLengthTol')
solver.wetted_length_relax = read('wettedLengthRelax')
solver.wetted_length_max_it = read('wettedLengthMaxIt')
solver.wetted_length_max_it_0 = read('wettedLengthMaxIt0')
solver.wetted_length_max_step_pct = read('wettedLengthMaxStepPct')
solver.wetted_length_max_step_pct_inc = read(
    'wettedLengthMaxStepPctInc', default=solver.wetted_length_max_step_pct)
solver.wetted_length_max_step_pct_dec = read(
    'wettedLengthMaxStepPctDec', default=solver.wetted_length_max_step_pct)
solver.wetted_length_max_jacobian_reset_step = read(
    'wettedLengthMaxJacobianResetStep')

solver.max_it = read('maxIt')
solver.num_ramp_it = read('rampIt')
solver.relax_initial = read('relaxI')
solver.relax_final = read('relaxF')
solver.max_residual = read('tolerance')
solver.pretension = read('pretension')
solver.relax_FEM = read('FEMRelax')
solver.max_FEM_disp = read('maxFEMDisp')
solver.num_damp = read('numDamp')

# Initialized constants
ramp = 1.0

has_free_structure = False
it_dir = ''
it = -1

del os
del math
del config_dict
del default_dict
del read
