"""This module is used to store the global configuration. Values are stored
after reading the configuration dictionary, and values can be accessed by other
packages and modules by importing the config module.

Usage: import planingfsi.config as config
"""

import numpy as np
import planingfsi.krampy as kp

config_dict_name = 'configDict'

print 'Loading {0}'.format(config_dict_name)
config_dict = kp.Dictionary(config_dict_name)
#defaultDict = kp.Dictionary('defaultDict')

ROD = config_dict.readOrDefault

# Load run properties from dictionary
rho = ROD('rho', 1.0)
g = ROD('g', 1.0)
U = ROD('U', None)
Fr = ROD('Fr', 1.0)
nu = ROD('nu', 1e-6)
hWL = ROD('hWL', 0.0)
xCofG = ROD('xCofG', 0.0)
yCofG = ROD('yCofG', 0.0)
xCofR = ROD('xCofR', xCofG)
yCofR = ROD('yCofR', yCofG)
m = ROD('m', 1 / g)
W = ROD('W', m * g)
Lref = ROD('Lref', ROD('Lc', 1.0)) # read Lc for backwards-compatibility
dim = ROD('dim', 2)
shearCalc = ROD('shearCalc', False)
Ps = ROD('Ps', None)
PsBar = ROD('PsBar', None)

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
Pc = ROD('Pc', None)
PcBar = ROD('PcBar', None)
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
pType = ROD('pScaleType', 'stagnation')
if pType == 'stagnation':
    pScale = pStag
elif pType == 'cushion':
    if Pc == 0.0:
        pScale = 1.0
    else:
        pScale = Pc
elif pType == 'hydrostatic':
    pScale = rho * g * ROD('pScaleHead', 1.0)
else:
    pScale = ROD('pScale', 1.0)
pScale = ROD('pScalePct', 0.1) / pScale
pressureLimiter = ROD('pressureLimiter', False)

# Load plot extents
extE = ROD('extE', 0.1)
extW = ROD('extW', 0.1)
extN = ROD('extN', 0.1)
extS = ROD('extS', 0.1)

plotXMin = ROD('plotXMin', None)
plotXMax = ROD('plotXMax', None)
plotYMin = ROD('plotYMin', None)
plotYMax = ROD('plotYMax', None)

lamMin = ROD('lamMin', -1.0)
lamMax = ROD('lamMax', 1.0)

xFSMin = ROD('xFSMin', plotXMin if plotXMin is not None else lamMin * lam)
xFSMax = ROD('xFSMax', plotXMax if plotXMax is not None else lamMax * lam)

growthRate = ROD('growthRate', 1.1)
CofRGridLen = ROD('CofRGridLen', 0.5)

# Directories and file formats
caseDir = ROD('caseDir', '.')
dataFormat = ROD('dataFormat', 'txt')
figFormat = ROD('figFormat', 'eps')
figDirName = ROD('figDirName', 'figures')
bodyDictDir = ROD('bodyDictDir', 'bodyDict')
inputDictDir = ROD('inputDictDir', 'inputDict')
cushionDictDir = ROD('cushionDictDir', 'cushionDict')
cushionDictDir = ROD('pressureCushionDictDir', cushionDictDir)
meshDir = ROD('meshDir', 'mesh')
meshDictDir = ROD('meshDictDir', 'meshDict')

# Whether to save, show, or watch plots
plotSave = ROD('plotSave', False)
plotPressure = ROD('plotPressure', False)
plotShow = ROD('plotShow', False)
plotWatch = ROD('plotWatch', False) or plotShow
plot = plotShow or plotSave or plotWatch or plotPressure

# File IO settings
writeInterval = ROD('writeInterval', 1)
writeTimeHistories = ROD('writeTimeHistories', False)
resultsFromFile = ROD('resultsFromFile', False)

# Rigid body motion parameters
timeStep = ROD('timeStep', 1e-3)
relaxRB = ROD('rigidBodyRelax', 1.0)
motionMethod = ROD('motionMethod', 'Physical')
motionJacobianFirstStep = ROD('motionJacobianFirstStep', 1e-6)

bowSealTipLoad = ROD('bowSealTipLoad', 0.0)
tipConstraintHt = ROD('tipConstraintHt', None)

sealLoadPct = ROD('sealLoadPct', 1.0)
cushionForceMethod = ROD('cushionForceMethod', 'Fixed')

initialDraft = ROD('initialDraft', 0.0)
initialTrim = ROD('initialTrim', 0.0)

maxDraftStep = ROD('maxDraftStep', 1e-3)
maxTrimStep = ROD('maxTrimStep', 1e-3)

maxDraftAcc = ROD('maxDraftAcc', 1000.0)
maxTrimAcc = ROD('maxTrimAcc', 1000.0)

freeInDraft = ROD('freeInDraft', False)
freeInTrim = ROD('freeInTrim', False)

draftDamping = ROD('draftDamping', 1000.0)
trimDamping = ROD('trimDamping', 500.0)

relaxDraft = ROD('draftRelax', relaxRB)
relaxTrim = ROD('trimRelax', relaxRB)

# Parameters for wetted length solver
wettedLengthSolver = ROD('wettedLengthSolver', 'Secant')
wettedLengthTol = ROD('wettedLengthTol', 1e-6)
wettedLengthRelax = ROD('wettedLengthRelax', 1.0)
wettedLengthMaxIt = ROD('wettedLengthMaxIt', 20)
wettedLengthMaxIt0 = ROD('wettedLengthMaxIt0', 100)
wettedLengthMaxStepPct = ROD('wettedLengthMaxStepPct', 0.2)
wettedLengthMaxStepPctInc = ROD('wettedLengthMaxStepPctDec', wettedLengthMaxStepPct)
wettedLengthMaxStepPctDec = ROD('wettedLengthMaxStepPctInc', wettedLengthMaxStepPct)
wettedLengthMaxJacobianResetStep = ROD('wettedLengthMaxJacobianResetStep', 100)

ramp = 1.0

maxIt = ROD('maxIt', 1)
rampIt = ROD('rampIt', 0)
relaxI = ROD('relaxI', 0.01)
relaxF = ROD('relaxF', 0.5)
maxRes = ROD('tolerance', 1e-6)
pretension = ROD('pretension', 0.1)
relaxFEM = ROD('FEMRelax', 1.0)
maxFEMDisp = ROD('maxFEMDisp', 1.0)
numDamp = ROD('numDamp', 0.0)

hasFreeStructure = False
