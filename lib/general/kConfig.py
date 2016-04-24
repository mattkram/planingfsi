# Module used to store global run configuration. Values are stored after reading
# the dictionary files, and values can be accessed by importing the config module
import numpy as np
import krampy as kp
import config

# Store config and readOrDefault to minimize verbosity
C = config
K = kp.Dictionary('configDict')
ROD = K.readOrDefault

# Load run properties from C.dictionary
C.rho       = ROD('rho',  1.0)
C.g         = ROD('g',    1.0)
C.U         = ROD('U',    None)
C.Fr        = ROD('Fr',   1.0)
C.nu        = ROD('nu',   1e-6)
C.hWL       = ROD('hWL',  0.0)
C.xCofG     = ROD('xCofG', 0.0)
C.yCofG     = ROD('yCofG', 0.0)
C.xCofR     = ROD('xCofR', C.xCofG)
C.yCofR     = ROD('yCofR', C.yCofG)
C.m         = ROD('m', 1 / C.g)
C.W         = ROD('W', C.m * C.g)
C.Lref      = ROD('Lref',   ROD('Lc', 1.0)) # read Lc for backwards-compatibility
C.dim       = ROD('dim',  2)
C.shearCalc = ROD('shearCalc', False)
C.Ps        = ROD('Ps', None)
C.PsBar     = ROD('PsBar', None)
 
# Calculate U or Fr depending on which was specified in the file
if C.U is not None:
    C.Fr = C.U  * (C.g * C.Lref) ** -0.5
elif C.Fr is not None:
    C.U  = C.Fr * (C.g * C.Lref) **  0.5
else:
    raise NameError('Must specify either U or Fr in configDict')

#print 'Flow Speed: {0} m/s'.format(C.U)

C.pStag = 0.5 * C.rho * C.U**2
C.k0    = C.g / C.U**2
C.lam   = 2 * np.pi / C.k0

# Calculate Pc or Pcbar depending on which was specified in the file
C.Pc    = ROD('Pc',    None)
C.PcBar = ROD('PcBar', None)
if C.Pc is not None:
    C.PcBar = C.Pc * C.Lref / C.W
elif C.PcBar is not None:
    C.Pc = C.PcBar * C.W / C.Lref
else:
    C.Pc = 0.0
    C.PcBar = 0.0

if C.Ps is not None:
    C.PsBar = C.Ps / C.Pc
elif C.PsBar is not None:
    C.Ps = C.PsBar * C.Pc
else:
    C.Ps = 0.0
    C.PsBar = 0.0

# Set pressure scale for plotting purposes 
pType     = ROD('pScaleType', 'stagnation')
if pType == 'stagnation':
    pScale  = C.pStag
elif pType == 'cushion':
    if C.Pc == 0.0:
        pScale = 1.0
    else:
        pScale  = C.Pc
elif pType == 'hydrostatic':
    h       = ROD('pScaleHead', 1.0)
    pScale  = C.rho * C.g * h
else:
    pScale  = ROD('pScale', 1.0)
C.pScale          = ROD('pScalePct', 0.1) / pScale
C.pressureLimiter = ROD('pressureLimiter', False)

# Load plot extents
C.extE = ROD('extE', 0.1)
C.extW = ROD('extW', 0.1)
C.extN = ROD('extN', 0.1)
C.extS = ROD('extS', 0.1)

C.plotXMin = ROD('plotXMin', None)
C.plotXMax = ROD('plotXMax', None)
C.plotYMin = ROD('plotYMin', None)
C.plotYMax = ROD('plotYMax', None)

C.lamMin = ROD('lamMin', -1.0)
C.lamMax = ROD('lamMax',  1.0)

C.xFSMin = ROD('xFSMin', C.plotXMin if C.plotXMin is not None else C.lamMin * C.lam)
C.xFSMax = ROD('xFSMax', C.plotXMax if C.plotXMax is not None else C.lamMax * C.lam)

C.growthRate  = ROD('growthRate',  1.1)
C.CofRGridLen = ROD('CofRGridLen', 0.5)

# Directories and file formats
C.caseDir        = ROD('caseDir'       , '.')
C.dataFormat     = ROD('dataFormat'    , 'txt')
C.figFormat      = ROD('figFormat'     , 'eps')
C.figDirName     = ROD('figDirName'    , 'figures')
C.bodyDictDir    = ROD('bodyDictDir'   , 'bodyDict')
C.inputDictDir   = ROD('inputDictDir'  , 'inputDict')
C.cushionDictDir = ROD('cushionDictDir', 'cushionDict')
C.cushionDictDir = ROD('pressureCushionDictDir', C.cushionDictDir)
C.meshDir        = ROD('meshDir'       , 'mesh')
C.meshDictDir    = ROD('meshDictDir'   , 'meshDict')
#C.geomDictPath   = ROD('geomDictPath', '.')
#C.resultsDir     = ROD('resultsDir', C.caseDir)

# Whether to save, show, or watch plots
C.plotSave     = ROD('plotSave',      False)
C.plotPressure = ROD('plotPressure',  False)
C.plotShow     = ROD('plotShow',      False)
C.plotWatch    = ROD('plotWatch',     False) or C.plotShow
C.plot         = C.plotShow or C.plotSave or C.plotWatch or C.plotPressure
 
# File IO settings
C.writeInterval   = ROD('writeInterval', 1)
C.writeTimeHistories = ROD('writeTimeHistories', False)
C.resultsFromFile = ROD('resultsFromFile', False)

# Rigid body motion parameters
C.timeStep     = ROD('timeStep',       1e-3)
C.relaxRB      = ROD('rigidBodyRelax', 1.0)
C.motionMethod = ROD('motionMethod', 'Physical')
C.motionJacobianFirstStep = ROD('motionJacobianFirstStep' , 1e-6)
 
C.bowSealTipLoad  = ROD('bowSealTipLoad', 0.0)
C.tipConstraintHt = ROD('tipConstraintHt', None)

C.sealLoadPct        = ROD('sealLoadPct', 1.0)
C.cushionForceMethod = ROD('cushionForceMethod', 'Fixed')

C.initialDraft = ROD('initialDraft', 0.0)
C.initialTrim  = ROD('initialTrim',  0.0)

C.maxDraftStep = ROD('maxDraftStep', 1e-3)
C.maxTrimStep  = ROD('maxTrimStep',  1e-3)

C.maxDraftAcc  = ROD('maxDraftAcc', 1000.0)
C.maxTrimAcc   = ROD('maxTrimAcc',  1000.0)
 
C.freeInDraft  = ROD('freeInDraft', False)
C.freeInTrim   = ROD('freeInTrim',  False)

C.draftDamping = ROD('draftDamping', 1000.0)
C.trimDamping  = ROD('trimDamping',   500.0)

C.relaxDraft   = ROD('draftRelax', C.relaxRB)
C.relaxTrim    = ROD('trimRelax',  C.relaxRB)
 
# Parameters for wetted length solver 
C.wettedLengthSolver               = ROD('wettedLengthSolver', 'Secant')
C.wettedLengthTol                  = ROD('wettedLengthTol', 1e-6)
C.wettedLengthRelax                = ROD('wettedLengthRelax', 1.0)
C.wettedLengthMaxIt                = ROD('wettedLengthMaxIt', 20)
C.wettedLengthMaxIt0               = ROD('wettedLengthMaxIt0', 100)
C.wettedLengthMaxStepPct           = ROD('wettedLengthMaxStepPct', 0.2)
C.wettedLengthMaxStepPctInc        = ROD('wettedLengthMaxStepPctDec', C.wettedLengthMaxStepPct)
C.wettedLengthMaxStepPctDec        = ROD('wettedLengthMaxStepPctInc', C.wettedLengthMaxStepPct)
C.wettedLengthMaxJacobianResetStep = ROD('wettedLengthMaxJacobianResetStep', 100)

C.maxIt      = ROD('maxIt', 1)
C.rampIt     = ROD('rampIt', 0)
C.relaxI     = ROD('relaxI', 0.01)
C.relaxF     = ROD('relaxF', 0.5)
C.maxRes     = ROD('tolerance', 1e-6)
C.pretension = ROD('pretension', 0.1)
C.relaxFEM   = ROD('FEMRelax', 1.0)
C.maxFEMDisp = ROD('maxFEMDisp', 1.0)
C.numDamp    = ROD('numDamp', 0.0)

C.hasFreeStructure = False
