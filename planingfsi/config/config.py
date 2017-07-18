"""This module is used to store the global configuration. Values are stored
after reading the configDit file, and values can be accessed by other
packages and modules by importing the config module.

Usage: import planingfsi.config as config
"""
import os

import numpy as np
from planingfsi import io

dict_name = 'configDict'

print('Loading {0}'.format(dict_name))

config_dict = io.Dictionary(dict_name)
default_dict = io.Dictionary(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'defaultDict'))

# Function to read value from dictionary or default dictionary
def read(key, **kwargs):
    if 'default' not in kwargs:
        return config_dict.read(key, default_dict.read(key, None))
    else:
        return config_dict.read(key, kwargs.get('default'))

class Subconfig(object):
    pass

# Load run properties from dictionary
rho = read('rho')
g = read('g')
nu = read('nu')
hWL = read('hWL')

U = read('U')
Fr = read('Fr')

xCofG = read('xCofG')
yCofG = read('yCofG')

xCofR = read('xCofR', default=xCofG)
yCofR = read('yCofR', default=yCofG)

m = read('m')
W = read('W', default=m * g)

Lref = read('Lref', default=read('Lc', default=1.0))  # read Lc for backwards-compatibility

dim = read('dim')
shear_calc = read('shearCalc')

# Calculate U or Fr depending on which was specified in the file
if U is not None:
    Fr = U * (g * Lref) ** -0.5
elif Fr is not None:
    U = Fr * (g * Lref) ** 0.5
else:
    raise NameError(
        'Must specify either U or Fr in {0}'.format(dict_name))

pStag = 0.5 * rho * U**2
k0 = g / U**2
lam = 2 * np.pi / k0

# Calculate Pc or Pcbar depending on which was specified in the file
Pc = read('Pc')
PcBar = read('PcBar')
if PcBar is not None:
    Pc = PcBar * W / Lref
else:
    PcBar = Pc * Lref / W

Ps = read('Ps')
PsBar = read('PsBar')
if PsBar is not None:
    Ps = PsBar * Pc
elif Pc == 0.0:
    PsBar = 0.0
else:
    PsBar = Ps / Pc

# Set pressure scale for plotting purposes
pType = read('pScaleType')
if pType == 'stagnation':
    pScale = pStag
elif pType == 'cushion':
    pScale = Pc if Pc > 0.0 else 1.0
elif pType == 'hydrostatic':
    pScale = rho * g * read('pScaleHead')
else:
    pScale = read('pScale', default=1.0)
    
growth_rate = read('growthRate')
CofR_grid_len = read('CofRGridLen')

# Directories and file formats
path = Subconfig()
path.case_dir = read('caseDir')
data_format = read('dataFormat')
fig_format = read('figFormat')
path.fig_dir_name = read('figDirName')
path.body_dict_dir = read('bodyDictDir')
path.input_dict_dir = read('inputDictDir')
path.cushion_dict_dir = read(
    'pressureCushionDictDir', default=read('cushionDictDir'))
path.mesh_dir = read('meshDir')
path.mesh_dict_dir = read('meshDictDir')

pScale = read('pScalePct') / pScale
pressure_limiter = read('pressureLimiter')

plotting = Subconfig()

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
    'xFSMin', default=plotting.xmin if plotting.xmin is not None else plotting.lambda_min * lam)
plotting.x_fs_max = read(
    'xFSMax', default=plotting.xmax if plotting.xmax is not None else plotting.lambda_max * lam)

# Whether to save, show, or watch plots
plotting.save = read('plotSave')
plotting.show_pressure = read('plotPressure')
plotting.show = read('plotShow')
plotting.watch = read('plotWatch') or plotting.show
plotting.plot_any = plotting.show or plotting.save or plotting.watch or plotting.show_pressure

# File IO settings
write_interval = read('writeInterval')
write_time_histories = read('writeTimeHistories')
results_from_file = read('resultsFromFile')

# Rigid body motion parameters
time_step = read('timeStep')
relax_rigid_body = read('rigidBodyRelax')
motion_method = read('motionMethod')
motion_jacobian_first_step = read('motionJacobianFirstStep')

bow_seal_tip_load = read('bowSealTipLoad')
tip_constraint_ht = read('tipConstraintHt')

seal_load_pct = read('sealLoadPct')
cushion_force_method = read('cushionForceMethod')

initial_draft = read('initialDraft')
initial_trim = read('initialTrim')

max_draft_step = read('maxDraftStep')
max_trim_step = read('maxTrimStep')

max_draft_acc = read('maxDraftAcc')
max_trim_acc = read('maxTrimAcc')

free_in_draft = read('freeInDraft')
free_in_trim = read('freeInTrim')

draft_damping = read('draftDamping')
trim_damping = read('trimDamping')

relax_draft = read('draftRelax', default=relax_rigid_body)
relax_trim = read('trimRelax', default=relax_rigid_body)

# Parameters for wetted length solver
wetted_length_solver = read('wettedLengthSolver')
wetted_length_tol = read('wettedLengthTol')
wetted_length_relax = read('wettedLengthRelax')
wetted_length_max_it = read('wettedLengthMaxIt')
wetted_length_max_it_0 = read('wettedLengthMaxIt0')
wetted_length_max_step_pct = read('wettedLengthMaxStepPct')
wetted_length_max_step_pct_inc = read(
    'wettedLengthMaxStepPctInc', default=wetted_length_max_step_pct)
wetted_length_max_step_pct_dec = read(
    'wettedLengthMaxStepPctDec', default=wetted_length_max_step_pct)
wetted_length_max_jacobian_reset_step = read(
    'wettedLengthMaxJacobianResetStep')

max_it = read('maxIt')
num_ramp_it = read('rampIt')
relax_initial = read('relaxI')
relax_final = read('relaxF')
max_residual = read('tolerance')
pretension = read('pretension')
relax_FEM = read('FEMRelax')
max_FEM_disp = read('maxFEMDisp')
num_damp = read('numDamp')

# Initialized constants
ramp = 1.0

has_free_structure = False
it_dir = ''
it = -1

del os
del np
del io
del config_dict
del default_dict
del read
