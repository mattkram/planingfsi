"""This module is used to store the global configuration. Values are stored
after reading the configDit file, and values can be accessed by other
packages and modules by importing the config module.

Usage: import planingfsi.config as config
"""
import os

import numpy as np
import planingfsi.krampy as kp

dict_name = 'configDict'

print 'Loading {0}'.format(dict_name)

config_dict = kp.Dictionary('configDict')
default_dict = kp.Dictionary(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'defaultDict'))

# Function to read value from dictionary or default dictionary
def read(key, **kwargs):
    if 'default' not in kwargs:
        return config_dict.read(key, default_dict.read(key, None))
    else:
        return config_dict.read(key, kwargs.get('default'))

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
shearCalc = read('shearCalc')

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

pScale = read('pScalePct') / pScale
pressureLimiter = read('pressureLimiter')

# Load plot extents
extE = read('extE')
extW = read('extW')
extN = read('extN')
extS = read('extS')

plot_xmin = read('plotXMin')
plot_xmax = read('plotXMax')
plot_ymin = read('plotYMin')
plot_ymax = read('plotYMax')

lambda_min = read('lamMin')
lambda_max = read('lamMax')

xFSMin = read(
    'xFSMin', default=plot_xmin if plot_xmin is not None else lambda_min * lam)
xFSMax = read(
    'xFSMax', default=plot_xmax if plot_xmax is not None else lambda_max * lam)

growthRate = read('growthRate')
CofRGridLen = read('CofRGridLen')

# Directories and file formats
case_dir = read('caseDir')
data_format = read('dataFormat')
fig_format = read('figFormat')
fig_dir_name = read('figDirName')
body_dict_dir = read('bodyDictDir')
input_dict_dir = read('inputDictDir')
cushion_dict_dir = read(
    'pressureCushionDictDir', default=read('cushionDictDir'))
mesh_dir = read('meshDir')
mesh_dict_dir = read('meshDictDir')

# Whether to save, show, or watch plots
plotSave = read('plotSave')
plot_pressure = read('plot_pressure')
plotShow = read('plotShow')
plotWatch = read('plotWatch') or plotShow
plot = plotShow or plotSave or plotWatch or plot_pressure

# File IO settings
writeInterval = read('writeInterval')
writeTimeHistories = read('writeTimeHistories')
resultsFromFile = read('resultsFromFile')

# Rigid body motion parameters
timeStep = read('timeStep')
relaxRB = read('rigidBodyRelax')
motionMethod = read('motionMethod')
motionJacobianFirstStep = read('motionJacobianFirstStep')

bowSealTipLoad = read('bowSealTipLoad')
tipConstraintHt = read('tipConstraintHt')

sealLoadPct = read('sealLoadPct')
cushionForceMethod = read('cushionForceMethod')

initialDraft = read('initialDraft')
initialTrim = read('initialTrim')

maxDraftStep = read('maxDraftStep')
maxTrimStep = read('maxTrimStep')

maxDraftAcc = read('maxDraftAcc')
maxTrimAcc = read('maxTrimAcc')

freeInDraft = read('freeInDraft')
freeInTrim = read('freeInTrim')

draftDamping = read('draftDamping')
trimDamping = read('trimDamping')

relaxDraft = read('draftRelax', default=relaxRB)
relaxTrim = read('trimRelax', default=relaxRB)

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

maxIt = read('maxIt')
rampIt = read('rampIt')
relaxI = read('relaxI')
relaxF = read('relaxF')
maxRes = read('tolerance')
pretension = read('pretension')
relaxFEM = read('FEMRelax')
maxFEMDisp = read('maxFEMDisp')
numDamp = read('numDamp')

# Initialized constants
ramp = 1.0

has_free_structure = False
it_dir = ''
it = -1

del os
del kp
del np
del config_dict
del default_dict
del read
