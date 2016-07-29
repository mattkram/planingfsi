"""This module is used to store the global configuration. Values are stored
after reading the configDit file, and values can be accessed by other
packages and modules by importing the config module.

Usage: import planingfsi.config as config
"""

import numpy as np
import planingfsi.krampy as kp

config_dict_name = 'configDict'

print 'Loading {0}'.format(config_dict_name)
config_dict = kp.Dictionary(config_dict_name)

# TODO: Read default values from a defaults file. Specialized logic and import
# of values from configDict should be handled in this file

# TODO: readOrDefault is not necessary, read method should handle default value
# on its own.

# Shortcut since called a lot
read = config_dict.readOrDefault

# Load run properties from dictionary
rho = read('rho', 1.0)
g = read('g', 1.0)
U = read('U', None)
Fr = read('Fr', 1.0)
nu = read('nu', 1e-6)
hWL = read('hWL', 0.0)
xCofG = read('xCofG', 0.0)
yCofG = read('yCofG', 0.0)
xCofR = read('xCofR', xCofG)
yCofR = read('yCofR', yCofG)
m = read('m', 1 / g)
W = read('W', m * g)
Lref = read('Lref', read('Lc', 1.0)) # read Lc for backwards-compatibility
dim = read('dim', 2)
shearCalc = read('shearCalc', False)
Ps = read('Ps', None)
PsBar = read('PsBar', None)

# Calculate U or Fr depending on which was specified in the file
if U is not None:
    Fr = U * (g * Lref) ** -0.5
elif Fr is not None:
    U = Fr * (g * Lref) ** 0.5
else:
    raise NameError('Must specify either U or Fr in {0}'.format(config_dict_name))

pStag = 0.5 * rho * U**2
k0 = g / U**2
lam = 2 * np.pi / k0

# Calculate Pc or Pcbar depending on which was specified in the file
Pc = read('Pc', None)
PcBar = read('PcBar', None)
if Pc is not None:
    PcBar = Pc * Lref / W
elif PcBar is not None:
    Pc = PcBar * W / Lref
else:
    Pc = 0.0
    PcBar = 0.0

if Ps is not None:
    PsBar = Ps / Pc
elif PsBar is not None:
    Ps = PsBar * Pc
else:
    Ps = 0.0
    PsBar = 0.0

# Set pressure scale for plotting purposes
pType = read('pScaleType', 'stagnation')
if pType == 'stagnation':
    pScale = pStag
elif pType == 'cushion':
    if Pc == 0.0:
        pScale = 1.0
    else:
        pScale = Pc
elif pType == 'hydrostatic':
    pScale = rho * g * read('pScaleHead', 1.0)
else:
    pScale = read('pScale', 1.0)
pScale = read('pScalePct', 0.1) / pScale
pressureLimiter = read('pressureLimiter', False)

# Load plot extents
extE = read('extE', 0.1)
extW = read('extW', 0.1)
extN = read('extN', 0.1)
extS = read('extS', 0.1)

plotXMin = read('plotXMin', None)
plotXMax = read('plotXMax', None)
plotYMin = read('plotYMin', None)
plotYMax = read('plotYMax', None)

lamMin = read('lamMin', -1.0)
lamMax = read('lamMax', 1.0)

xFSMin = read('xFSMin', plotXMin if plotXMin is not None else lamMin * lam)
xFSMax = read('xFSMax', plotXMax if plotXMax is not None else lamMax * lam)

growthRate = read('growthRate', 1.1)
CofRGridLen = read('CofRGridLen', 0.5)

# Directories and file formats
caseDir = read('caseDir', '.')
dataFormat = read('dataFormat', 'txt')
figFormat = read('figFormat', 'eps')
figDirName = read('figDirName', 'figures')
bodyDictDir = read('bodyDictDir', 'bodyDict')
inputDictDir = read('inputDictDir', 'inputDict')
cushionDictDir = read('cushionDictDir', 'cushionDict')
cushionDictDir = read('pressureCushionDictDir', cushionDictDir)
meshDir = read('meshDir', 'mesh')
meshDictDir = read('meshDictDir', 'meshDict')

# Whether to save, show, or watch plots
plotSave = read('plotSave', False)
plotPressure = read('plotPressure', False)
plotShow = read('plotShow', False)
plotWatch = read('plotWatch', False) or plotShow
plot = plotShow or plotSave or plotWatch or plotPressure

# File IO settings
writeInterval = read('writeInterval', 1)
writeTimeHistories = read('writeTimeHistories', False)
resultsFromFile = read('resultsFromFile', False)

# Rigid body motion parameters
timeStep = read('timeStep', 1e-3)
relaxRB = read('rigidBodyRelax', 1.0)
motionMethod = read('motionMethod', 'Physical')
motionJacobianFirstStep = read('motionJacobianFirstStep', 1e-6)

bowSealTipLoad = read('bowSealTipLoad', 0.0)
tipConstraintHt = read('tipConstraintHt', None)

sealLoadPct = read('sealLoadPct', 1.0)
cushionForceMethod = read('cushionForceMethod', 'Fixed')

initialDraft = read('initialDraft', 0.0)
initialTrim = read('initialTrim', 0.0)

maxDraftStep = read('maxDraftStep', 1e-3)
maxTrimStep = read('maxTrimStep', 1e-3)

maxDraftAcc = read('maxDraftAcc', 1000.0)
maxTrimAcc = read('maxTrimAcc', 1000.0)

freeInDraft = read('freeInDraft', False)
freeInTrim = read('freeInTrim', False)

draftDamping = read('draftDamping', 1000.0)
trimDamping = read('trimDamping', 500.0)

relaxDraft = read('draftRelax', relaxRB)
relaxTrim = read('trimRelax', relaxRB)

# Parameters for wetted length solver
wettedLengthSolver = read('wettedLengthSolver', 'Secant')
wettedLengthTol = read('wettedLengthTol', 1e-6)
wettedLengthRelax = read('wettedLengthRelax', 1.0)
wettedLengthMaxIt = read('wettedLengthMaxIt', 20)
wettedLengthMaxIt0 = read('wettedLengthMaxIt0', 100)
wettedLengthMaxStepPct = read('wettedLengthMaxStepPct', 0.2)
wettedLengthMaxStepPctInc = read('wettedLengthMaxStepPctDec', wettedLengthMaxStepPct)
wettedLengthMaxStepPctDec = read('wettedLengthMaxStepPctInc', wettedLengthMaxStepPct)
wettedLengthMaxJacobianResetStep = read('wettedLengthMaxJacobianResetStep', 100)

ramp = 1.0

maxIt = read('maxIt', 1)
rampIt = read('rampIt', 0)
relaxI = read('relaxI', 0.01)
relaxF = read('relaxF', 0.5)
maxRes = read('tolerance', 1e-6)
pretension = read('pretension', 0.1)
relaxFEM = read('FEMRelax', 1.0)
maxFEMDisp = read('maxFEMDisp', 1.0)
numDamp = read('numDamp', 0.0)

hasFreeStructure = False
itDir = ''
it = -1
