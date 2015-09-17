import numpy as np
import krampy as kp
from scipy.interpolate import interp1d
import FEMLib as fe
import config, os
if config.plot:
  import matplotlib.pyplot as plt


class FEStructure:    

  def __init__(self):
    self.rigidBody    = []
    self.substructure = []
    self.node         = []
    self.res          = 1.0

  def addRigidBody(self, Dict=''):
    bd = RigidBody(kp.ensureDict(Dict))
    self.rigidBody.append(bd)
    return bd

  def addSubstructure(self, Dict=''):
    Dict = kp.ensureDict(Dict)

    ssType = Dict.readOrDefault('substructureType', 'rigid')
    if ssType.lower() == 'flexible' or ssType.lower() == 'truss':
      ss = FlexibleSubstructure(Dict)
      FlexibleSubstructure.obj.append(ss)
    elif ssType.lower() == 'rotating':
      ss = RotatingSubstructure(Dict)
    elif ssType.lower() == 'torsionalspring':
      ss = TorsionalSpringSubstructure(Dict)
    else:
      ss = RigidSubstructure(Dict)
    self.substructure.append(ss)

    # Find parent body and add substructure to it
    body = [b for b in self.rigidBody if b.name == Dict.readOrDefault('bodyName', 'default')]
    if len(body) > 0:
      body = body[0]
    else:
      body = self.rigidBody[0]
    body.addSubstructure(ss)
    ss.addParent(body)
    print "Adding Substructure {0} of type {1} to rigid body {2}".format(ss.name, ss.type, body.name)

    return ss

  def initializeRigidBodies(self):
    for bd in self.rigidBody:
      bd.initializePosition()

  def updateFluidForces(self):
    for bd in self.rigidBody:
      bd.updateFluidForces()
          
  def calculateResponse(self):
    if config.resultsFromFile:
      self.loadResponse()
    else:
#      self.getPtDispTotal()
#      self.updateNodalPositions()
      for bd in self.rigidBody:
        bd.updatePosition()
        bd.updateSubstructurePositions()
#      if all([ss.getResidual() < 1e-4 for ss in bd.substructure]):
#        bd.updatePosition()

  def getResidual(self):
    self.res = 0.0
    for bd in self.rigidBody:
      if bd.freeInDraft or bd.freeInTrim:
        self.res = np.max([np.abs(bd.resL), self.res])
        self.res = np.max([np.abs(bd.resM), self.res])
#      for ss in bd.substructure:
#        self.res = np.max([np.abs(ss.getResidual()), self.res])
      self.res = np.max([FlexibleSubstructure.res, self.res])

  def loadResponse(self):
    self.updateFluidForces()
    
    for bd in self.rigidBody:
      bd.loadMotion()
      for ss in bd.substructure:
        ss.loadCoordinates()
        ss.updateGeometry()

#    for struct in self.substructure:
#      struct.loadCoordinates()
#
#    for struct in self.substructure:
#      struct.updateGeometry()

  def writeResults(self):
    for bd in self.rigidBody:
      bd.writeMotion()
      for ss in bd.substructure:
        ss.writeCoordinates()

  def plot(self):
    for body in self.rigidBody:
      for struct in body.substructure:
        struct.plot()

  def generateMesh(self):
    # Create all nodes
    x,  y  = np.loadtxt(os.path.join(config.meshDir, 'nodes.txt'),    unpack=True)
    xf, yf = np.loadtxt(os.path.join(config.meshDir, 'fixedDOF.txt'), unpack=True)
    fx, fy = np.loadtxt(os.path.join(config.meshDir, 'fixedLoad.txt'), unpack=True)
    for xx, yy, xxf, yyf, ffx, ffy in zip(x, y, xf, yf, fx, fy):
      nd = fe.Node()
      nd.setCoordinates(xx, yy)
      nd.fixedDOF = [bool(xxf), bool(yyf)]
      nd.fixedLoad = np.array([ffx, ffy])
      self.node.append(nd)

    for struct in self.substructure:
      struct.generateMesh()
      if struct.type == 'rigid' or struct.type == 'rotating' or struct.type == 'torsionalSpring':
        struct.setFixedDOF()
    for ss in self.substructure:
      ss.setAttachments()

     
class RigidBody:
  
  def __init__(self, Dict):
    self.Dict = Dict
    self.dim = 2
    self.draft = 0.0
    self.trim  = 0.0 

    self.name = self.Dict.readOrDefault('bodyName', 'default')
    self.W  = self.Dict.readOrDefault('W', self.Dict.readOrDefault('loadPct', 1.0) * config.W)
    self.W *= config.sealLoadPct
    self.m  = self.Dict.readOrDefault('m', self.W / config.g)
    self.Iz = self.Dict.readOrDefault('Iz', self.m * config.Lref**2 / 12)
    self.hasPlaningSurface = self.Dict.readOrDefault('hasPlaningSurface', False) 
  
    var = ['maxDraftStep', 'maxTrimStep', 'freeInDraft', 'freeInTrim', 'draftDamping', 'trimDamping', 'maxDraftAcc', 'maxTrimAcc', 'xCofG', 'yCofG', 'xCofR', 'yCofR', 'initialDraft', 'initialTrim', 'relaxDraft', 'relaxTrim', 'timeStep', 'numDamp']
    for v in var:
      exec('self.{0} = self.Dict.readOrDefault(\'{0}\', config.{0})'.format(v))
    
#    self.xCofR = self.Dict.readOrDefault('xCofR', self.xCofG)
#    self.yCofR = self.Dict.readOrDefault('yCofR', self.yCofG)

    self.xCofR0 = self.xCofR
    self.yCofR0 = self.yCofR

    self.maxDisp = np.array([self.maxDraftStep, self.maxTrimStep])
    self.freeDoF = np.array([self.freeInDraft,  self.freeInTrim])
    self.Cdamp   = np.array([self.draftDamping, self.trimDamping])
    self.maxAcc  = np.array([self.maxDraftAcc,  self.maxTrimAcc])
    self.relax   = np.array([self.relaxDraft, self.relaxTrim])
    if self.freeInDraft or self.freeInTrim:
      config.hasFreeStructure = True

    self.v = np.zeros((self.dim))
    self.a = np.zeros((self.dim))
    self.vOld = np.zeros((self.dim))
    self.aOld = np.zeros((self.dim))
    
    self.beta  = self.Dict.readOrDefault('beta',  0.25)
    self.gamma = self.Dict.readOrDefault('gamma', 0.5)
     
    self.D = 0.0
    self.L = 0.0
    self.M = 0.0
    self.Da = 0.0
    self.La = 0.0
    self.Ma = 0.0

    self.solver     = None
    self.dispOld    = 0.0
    self.resOld     = None
    self.twoAgoDisp = 0.0
    self.predictor  = True
    self.fOld = 0.0
    self.twoAgoF = 0.0 
    self.resL = 1.0
    self.resM = 1.0
     
    self.J = None   
    self.Jtmp = None   

    # Assign displacement function depending on specified method
    self.getDisp = lambda: (0.0, 0.0)
    if any(self.freeDoF):
      if config.motionMethod == 'Secant':
        self.getDisp = self.getDispSecant
      elif config.motionMethod == 'Broyden':
        self.getDisp = self.getDispBroyden
      elif config.motionMethod == 'BroydenNew':
        self.getDisp = self.getDispBroydenNew
      elif config.motionMethod == 'Physical':
        self.getDisp = self.getDispPhysical
      elif config.motionMethod == 'Newmark-Beta':
        self.getDisp = self.getDispNewmarkBeta
      elif config.motionMethod == 'PhysicalNoMass':
        self.getDisp = self.getDispPhysicalNoMass
      elif config.motionMethod == 'Sep':
        self.getDisp = self.getDispSep
        self.trimSolver = None
        self.draftSolver = None

    self.substructure = []
    self.node = None

    print "Adding Rigid Body: {0}".format(self.name)

  def addSubstructure(self, ss):
    self.substructure.append(ss)

  def storeNodes(self):
    self.node = []
    for ss in self.substructure:
      for nd in ss.node:
        if not any([n.nodeNum == nd.nodeNum for n in self.node]):
          self.node.append(nd)
    
  def initializePosition(self):
    self.setPosition(self.initialDraft, self.initialTrim) 

  def setPosition(self, draft, trim):
    self.updatePosition(draft - self.draft, trim - self.trim)

  def updatePosition(self, dDraft=None, dTrim=None):
    if dDraft is None:
      dDraft, dTrim = self.getDisp()
      if np.isnan(dDraft):
        dDraft = 0.0
      if np.isnan(dTrim):
        dTrim = 0.0
     
    if self.node is None:
      self.storeNodes()

    for nd in self.node:
      xo, yo = nd.getCoordinates()
      newPos = kp.rotatePt([xo, yo], [self.xCofR, self.yCofR], dTrim)
      nd.moveCoordinates(newPos[0] - xo, newPos[1] - yo - dDraft)

    for s in self.substructure:
      s.updateGeometry()
       
    self.xCofG, self.yCofG = kp.rotatePt([self.xCofG, self.yCofG], [self.xCofR, self.yCofR], dTrim)
    self.yCofG -= dDraft
    self.yCofR -= dDraft

    self.draft += dDraft
    self.trim  += dTrim

    self.printMotion()

  def updateSubstructurePositions(self):
    FlexibleSubstructure.updateAll()
    for ss in self.substructure:
      print "Updating position for substructure: {0}".format(ss.name)
#      if ss.type.lower() == 'flexible':
#        ss.getPtDispFEM()

      if ss.type.lower() == 'torsionalspring' or \
         ss.type.lower() == 'rigid':
        ss.updateAngle()

  def updateFluidForces(self):
    self.resetLoads()
    for ss in self.substructure:
      ss.updateFluidForces()
      self.D += ss.D
      self.L += ss.L
      self.M += ss.M
      self.Da += ss.Da
      self.La += ss.La
      self.Ma += ss.Ma

    self.resL = self.getResL()
    self.resM = self.getResM()

  def resetLoads(self):
    self.D *= 0.0
    self.L *= 0.0
    self.M *= 0.0
    if config.cushionForceMethod.lower() == 'assumed':
      self.L += config.Pc * config.Lref * kp.cosd(config.trim)
    
#  def getDispPhysical(self):
#    a = np.array([self.W - self.L, self.M - self.W * (self.xCofG - self.xCofR)])
#    a -= self.Cdamp * self.v**2
#    a /= np.array([self.m, self.Iz])
##    a = np.min(np.vstack((np.abs(a), np.array([self.maxDraftAcc, self.maxTrimAcc]))), axis=0) * np.sign(a)
#    
#    print a
#     
#    self.v += a * self.timeStep
#    
#    print self.v
#
#    disp = self.v * self.timeStep * self.freeDoF
##    disp *= config.ramp
#    
#    print disp
##    raw_input()
#
#    return disp

  def getDispPhysical(self):
    disp = self.limitDisp(self.timeStep * self.v) 
    
    for i in range(self.dim):
      if np.abs(disp[i]) == np.abs(self.maxDisp[i]):
        self.v[i] = disp[i] / self.timeStep

    self.v += self.timeStep * self.a

    self.a = np.array([self.W - self.L, self.M - self.W * (self.xCofG - self.xCofR)])
#    self.a -= self.Cdamp * (self.v + self.v**3) * config.ramp
    self.a -= self.Cdamp * self.v * config.ramp
    self.a /= np.array([self.m, self.Iz])
    self.a = np.min(np.vstack((np.abs(self.a), np.array([self.maxDraftAcc, self.maxTrimAcc]))), axis=0) * np.sign(self.a)
    
#    accLimPct = np.min(np.vstack((np.abs(self.a), self.maxAcc)), axis=0) * np.sign(self.a)
#    for i in range(len(self.a)):
#      if self.a[i] == 0.0 or not self.freeDoF[i]:
#        accLimPct[i] = 1.0
#      else:
#        accLimPct[i] /= self.a[i]
#
#    self.a *= np.min(accLimPct)
    disp *= config.ramp
    return disp

  def getDispNewmarkBeta(self):
    self.a = np.array([self.W - self.L, self.M - self.W * (self.xCofG - self.xCofR)])
#    self.a -= self.Cdamp * self.v * config.ramp
    self.a /= np.array([self.m, self.Iz])
    self.a = np.min(np.vstack((np.abs(self.a), np.array([self.maxDraftAcc, self.maxTrimAcc]))), axis=0) * np.sign(self.a)

    dv = (1 - self.gamma) * self.aOld + self.gamma * self.a
    dv *= self.timeStep
    dv *= (1 - self.numDamp)
    self.v += dv

    disp = 0.5 * (1 - 2*self.beta) * self.aOld + self.beta * self.a
    disp *= self.timeStep
    disp += self.vOld
    disp *= self.timeStep

    self.aOld = self.a
    self.vOld = self.v
    
    disp *= self.relax
    disp *= config.ramp
    disp = self.limitDisp(disp) 
#    disp *= config.ramp

    return disp

  def getDispPhysicalNoMass(self):
    F = np.array([self.W - self.L, self.M - self.W * (config.xCofG - config.xCofR)])
#    F -= self.Cdamp * self.v
  
    if self.predictor:
      disp =  F / self.Cdamp * self.timeStep
      self.predictor = False
    else:
      disp = 0.5 * self.timeStep / self.Cdamp * (F - self.twoAgoF)
      self.predictor = True
        
    disp *= self.relax * config.ramp
    disp = self.limitDisp(disp)

#    self.v = disp / self.timeStep

    self.twoAgoF = self.fOld
    self.fOld = disp * self.Cdamp / self.timeStep 
     
    return disp

  def getDispSecant(self):
    if self.solver is None:
      self.resFun = lambda x: np.array([self.L - self.W, self.M - self.W * (config.xCofG - config.xCofR)])
      self.solver = kp.RootFinder(self.resFun, np.array([config.initialDraft, config.initialTrim]), 'secant', dxMax=self.maxDisp * self.freeDoF)

    if not self.dispOld is None:
      self.solver.takeStep(self.dispOld)
    
    # Solve for limited displacement
    disp = self.solver.limitStep(self.solver.getStep())
   
    self.twoAgoDisp = self.dispOld
    self.dispOld    = disp

    return disp

  def resetJacobian(self):
    if self.Jtmp is None:
      self.Jit = 0
      self.Jtmp = np.zeros((self.dim, self.dim))
      self.step = 0
      self.Jfo = self.resFun(self.x)
      self.resOld = self.Jfo * 1.0
    else:
      f = self.resFun(self.x)
      
      self.Jtmp[:,self.Jit] = (f - self.Jfo) / self.dispOld[self.Jit]
      self.Jit += 1
    
    disp = np.zeros((self.dim))
    if self.Jit < self.dim:
      disp[self.Jit] = config.motionJacobianFirstStep

    if self.Jit > 0:
      disp[self.Jit-1] = -config.motionJacobianFirstStep
    self.dispOld = disp
    if self.Jit >= self.dim:
      self.J = self.Jtmp * 1.0
      self.Jtmp = None
      self.dispOld = None
 
    return disp
   
  def getDispBroydenNew(self):
    if self.solver is None:
      self.resFun = lambda x: np.array([f for f, freeDoF in zip([self.getResL(), self.getResM()], self.freeDoF) if freeDoF])
      maxDisp = [m for m, freeDoF in zip(self.maxDisp, self.freeDoF) if freeDoF]

      self.x = np.array([f for f, freeDoF in zip([self.draft, self.trim], self.freeDoF) if freeDoF])
      self.solver = kp.RootFinderNew(self.resFun, \
                                     self.x, \
                                     'broyden', \
                                     dxMax=maxDisp)

    if not self.dispOld is None:
      self.solver.takeStep(self.dispOld)
    
    # Solve for limited displacement
    disp = self.solver.limitStep(self.solver.getStep())
   
    self.twoAgoDisp = self.dispOld
    self.dispOld    = disp
    return disp

  def getDispBroyden(self):   
    if self.solver is None:
      self.resFun = lambda x: np.array([self.L - self.W, self.M - self.W * (self.xCofG - self.xCofR)])
#      self.resFun = lambda x: np.array([self.getResM(), self.getResL()])
      self.solver = 1.0
      self.x = np.array([self.draft, self.trim])

    if self.J is None:
      disp = self.resetJacobian()
    else:
      self.f = self.resFun(self.x)
      if not self.dispOld is None:
        self.x += self.dispOld
        
        dx = np.reshape(self.dispOld, (self.dim, 1))
        df = np.reshape(self.f - self.resOld, (self.dim, 1))
        
        self.J += np.dot(df - np.dot(self.J, dx), dx.T) / np.linalg.norm(dx)**2
        
      dof = self.freeDoF
      dx = np.zeros_like(self.x)
      A = -self.J[np.ix_(dof,dof)]
      b =  self.f.reshape(self.dim, 1)[np.ix_(dof)]

      dx[np.ix_(dof)] = np.linalg.solve(A, b)
      
      if self.resOld is not None: 
        if any(np.abs(self.f) - np.abs(self.resOld) > 0.0):
          self.step += 1
  
      if self.step >= 6:
        print '\nResetting Jacobian for Motion\n'
        disp = self.resetJacobian()
        
      disp = dx.reshape(self.dim)
    
#      disp = self.solver.getStep()
      
      disp *= self.relax 
      disp = self.limitDisp(disp)
      
      self.dispOld = disp
      
      self.resOld = self.f * 1.0
      self.step += 1
    return disp

  def limitDisp(self, disp):
    dispLimPct = np.min(np.vstack((np.abs(disp), self.maxDisp)), axis=0) * np.sign(disp)
    for i in range(len(disp)):
      if disp[i] == 0.0 or not self.freeDoF[i]:
        dispLimPct[i] = 1.0
      else:
        dispLimPct[i] /= disp[i]

    return disp * np.min(dispLimPct) * self.freeDoF

  def getResL(self):
    if np.isnan(self.L):
      res = 1.0
    else:
      res = (self.L - self.W) / (config.pStag * config.Lref + 1e-6)
    return np.abs(res * self.freeInDraft)

  def getResM(self):
    if np.isnan(self.M):
      res = 1.0
    else:
      if self.xCofG == self.xCofR and self.M == 0.0:
        res = 1.0
      else:
        res = (self.M - self.W * (self.xCofG - self.xCofR)) / (config.pStag * config.Lref**2 + 1e-6)
    return np.abs(res * self.freeInTrim)

  def printMotion(self):
    print 'Rigid Body Motion: {0}'.format(self.name)
    print '  CofR: ({0}, {1})'.format(self.xCofR, self.yCofR)
    print '  CofG: ({0}, {1})'.format(self.xCofG, self.yCofG)
    print '  Draft:      {0:5.4e}'.format(self.draft)
    print '  Trim Angle: {0:5.4e}'.format(self.trim)
    print '  Lift Force: {0:5.4e}'.format(self.L)
    print '  Drag Force: {0:5.4e}'.format(self.D)
    print '  Moment:     {0:5.4e}'.format(self.M)
    print '  Lift Force Air: {0:5.4e}'.format(self.La)
    print '  Drag Force Air: {0:5.4e}'.format(self.Da)
    print '  Moment Air:     {0:5.4e}'.format(self.Ma)
    print '  Lift Res:   {0:5.4e}'.format(self.resL)
    print '  Moment Res: {0:5.4e}'.format(self.resM)

  def writeMotion(self):
    kp.writeasdict(os.path.join(config.itDir, 'motion_{0}.{1}'.format(self.name, config.dataFormat)),
                   ['xCofR',     self.xCofR],
                   ['yCofR',     self.yCofR],
                   ['xCofG',     self.xCofG],
                   ['yCofG',     self.yCofG],
                   ['draft',     self.draft],
                   ['trim' ,     self.trim],
                   ['liftRes',   self.resL],
                   ['momentRes', self.resM],
                   ['Lift',      self.L],
                   ['Drag',      self.D],
                   ['Moment',    self.M],
                   ['LiftAir',      self.La],
                   ['DragAir',      self.Da],
                   ['MomentAir',    self.Ma])
    for ss in self.substructure:
      if ss.type == 'torsionalSpring':
        ss.writeDeformation()

  def loadMotion(self):
    K = kp.Dictionary(os.path.join(config.itDir, 'motion_{0}.{1}'.format(self.name, config.dataFormat)))
    self.xCofR = K.readOrDefault('xCofR',     np.nan)
    self.yCofR = K.readOrDefault('yCofR',     np.nan)
    self.xCofG = K.readOrDefault('xCofG',     np.nan)
    self.yCofG = K.readOrDefault('yCofG',     np.nan)
    self.draft = K.readOrDefault('draft',     np.nan)
    self.trim  = K.readOrDefault('trim',      np.nan)
    self.resL  = K.readOrDefault('liftRes',   np.nan)
    self.resM  = K.readOrDefault('momentRes', np.nan)
    self.L     = K.readOrDefault('Lift',      np.nan)
    self.D     = K.readOrDefault('Drag',      np.nan)
    self.M     = K.readOrDefault('Moment',    np.nan)
    self.La    = K.readOrDefault('LiftAir',      np.nan)
    self.Da    = K.readOrDefault('DragAir',      np.nan)
    self.Ma    = K.readOrDefault('MomentAir',    np.nan)
    

class Substructure:
  count = 0
  obj = []
  
  @classmethod
  def All(cls):
    return [o for o in cls.obj]
 
  @classmethod
  def findByName(cls, name):
    return [o for o in cls.obj if o.name == name][0]

  def __init__(self, Dict):
    self.index = Substructure.count
    Substructure.count += 1
    Substructure.obj.append(self)

    self.Dict = Dict
    self.name = self.Dict.readOrDefault('substructureName', '')
    self.type = self.Dict.readOrDefault('substructureType', 'rigid')
    self.interpolator = None
    
    self.Ps = self.Dict.readLoadOrDefault('Ps', 0.0)
    self.PsMethod = self.Dict.readOrDefault('PsMethod', 'constant')

    self.Psx = self.Dict.readLoadOrDefault('overPressurePct', 1.0)
    self.cushionPressureType = self.Dict.readOrDefault('cushionPressureType', None)
    self.tipLoad = self.Dict.readLoadOrDefault('tipLoad', 0.0)
    self.tipConstraintHt = self.Dict.readOrDefault('tipConstraintHt', None)
    self.structInterpType = self.Dict.readOrDefault('structInterpType', 'linear')
    self.structExtrap     = self.Dict.readOrDefault('structExtrap', True)
    self.lineFluidP = None
    self.lineAirP   = None
    self.fluidS = []
    self.fluidP = []
    self.airS   = []
    self.airP   = []
    self.U = 0.0

  def addParent(self, parent):
    self.parent = parent
 
  def getResidual(self):
    return 0.0
     
  def setInterpolator(self, interpolator):
    self.interpolator = interpolator

  def setElementProperties(self):
    for el in self.el:
      el.setProperties(length=self.getArcLength() / len(self.el))

  def generateMesh(self):
    ndSt, ndEnd = np.loadtxt(os.path.join(config.meshDir, 'elements_{0}.txt'.format(self.name)), unpack=True)
    if isinstance(ndSt, float):
      ndSt  = [int(ndSt)]
      ndEnd = [int(ndEnd)]
    else:
      ndSt  = [int(nd) for nd in ndSt]
      ndEnd = [int(nd) for nd in ndEnd]
    ndInd = ndSt + [ndEnd[-1]]
    
    # Generate Element list
    self.node = [fe.Node.getInd(i) for i in ndInd]
    
    self.setInterpFunction()
    self.el   = [self.elementType() for _ in ndSt]
    self.setElementProperties()
    for ndSti, ndEndi, el in zip(ndSt, ndEnd, self.el):
      el.setNodes([fe.Node.getInd(ndSti), fe.Node.getInd(ndEndi)])
      el.setParent(self)

  def setInterpFunction(self):
    self.nodeS = np.zeros(len(self.node))
    for i, nd0, nd1 in zip(range(len(self.node)-1), self.node[:-1], self.node[1:]):
      self.nodeS[i+1] = self.nodeS[i] + ((nd1.x - nd0.x)**2 + (nd1.y - nd0.y)**2)**0.5
    
    if len(self.nodeS) == 2:
      self.structInterpType = 'linear'
    elif len(self.nodeS) == 3 and not self.structInterpType == 'linear':
      self.structInterpType = 'quadratic'
       
    x, y = [np.array(xx) for xx in zip(*[(nd.x, nd.y) for nd in self.node])]
    self.interpFuncX, self.interpFuncY = interp1d(self.nodeS, x), interp1d(self.nodeS, y, kind=self.structInterpType)

    if self.structExtrap:
      self.interpFuncX, self.interpFuncY = self.extrapCoordinates(self.interpFuncX, self.interpFuncY)

  def extrapCoordinates(self, fxi, fyi):
    def extrap1d(interpolator):
      xs = interpolator.x
      ys = interpolator.y
      def pointwise(xi):
        if xi < xs[0]:
          return ys[0]+(xi-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif xi > xs[-1]:
          return ys[-1]+(xi-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
          return interpolator(xi)

      def ufunclike(xs):
        return np.array(map(pointwise, np.array([xs])))[0]

      return ufunclike

    return extrap1d(fxi), extrap1d(fyi)

  def getCoordinates(self, si):
    return self.interpFuncX(si), self.interpFuncY(si)

  def getXCoordinates(self, s):
    return self.getCoordinates(s)[0]

  def getYCoordinates(self, s):
    return self.getCoordinates(s)[1]
  
  def getArcLength(self):
    return max(self.nodeS)

  def writeCoordinates(self):
    kp.writeaslist(os.path.join(config.itDir, 'coords_{0}.{1}'.format(self.name, config.dataFormat)),
                   ['x [m]', [nd.x for nd in self.node]],
                   ['y [m]', [nd.y for nd in self.node]])

  def loadCoordinates(self):
    x, y = np.loadtxt(os.path.join(config.itDir, 'coords_{0}.{1}'.format(self.name, config.dataFormat)), unpack=True)
    for xx, yy, nd in zip(x, y, self.node):
      nd.setCoordinates(xx, yy)

  def updateFluidForces(self):
    self.fluidS = []
    self.fluidP = []
    self.airS   = []
    self.airP   = []
    self.D = 0.0
    self.L = 0.0
    self.M = 0.0
    self.Da = 0.0
    self.La = 0.0
    self.Ma = 0.0
    if self.interpolator is not None:
      sMin, sMax = self.interpolator.getMinMaxS()

    for i, el in enumerate(self.el):
      # Get pressure at end points and all fluid points along element
      nodeS = [self.nodeS[i], self.nodeS[i+1]]
      if self.interpolator is not None:
        s, pFl, tau = self.interpolator.getLoadsInRange(nodeS[0], nodeS[1])
        # Limit pressure to be below stagnation pressure 
        if config.pressureLimiter:
          pFl = np.min(np.hstack((pFl, np.ones_like(pFl) * config.pStag)), axis=0)

      else:
        s   = np.array(nodeS)
        pFl = np.zeros_like(s)
        tau = np.zeros_like(s)

      ss = nodeS[1]
      Pc = 0.0
      if self.interpolator is not None:
        if ss > sMax:
          Pc = self.interpolator.fluid.getUpstreamPressure()
        elif ss < sMin:
          Pc = self.interpolator.fluid.getDownstreamPressure()
      elif self.cushionPressureType == 'Total':
        Pc = config.Pc
         
      # Store fluid and air pressure components for element (for plotting)
      if i == 0:
        self.fluidS += [s[0]]
        self.fluidP += [pFl[0]]
        self.airS   += [nodeS[0]]
        self.airP   += [Pc - self.Ps]
        
      self.fluidS += [ss for ss in s[1:]]
      self.fluidP += [pp for pp in pFl[1:]]
      self.airS   += [ss for ss in nodeS[1:]]
      if self.PsMethod.lower() == 'hydrostatic':
        self.airP   += [Pc - self.Ps + config.rho * config.g * (self.interpFuncY(si) - config.hWL) for si in nodeS[1:]]
      else:
        self.airP   += [Pc - self.Ps for _ in nodeS[1:]]
   
      # Apply ramp to hydrodynamic pressure
      pFl *= config.ramp**2

      # Add external cushion pressure to external fluid pressure
      pC = np.zeros_like(s)
      Pc = 0.0
      for ii, ss in enumerate(s):
        if self.interpolator is not None:
          if ss > sMax:
            Pc = self.interpolator.fluid.getUpstreamPressure()
          elif ss < sMin:
            Pc = self.interpolator.fluid.getDownstreamPressure()
        elif self.cushionPressureType == 'Total':
          Pc = config.Pc

        pC[ii] = Pc
      
      # Calculate internal pressure
      if self.PsMethod.lower() == 'hydrostatic':
        pInt = self.Ps - config.rho * config.g * (np.array([self.interpFuncY(si) for si in s]) - config.hWL)
      else:
        pInt = self.Ps * np.ones_like(s) * self.Psx
      
      pExt = pFl + pC
      pTot = pExt - pInt

      # Integrate pressure profile, calculate center of pressure and distribute force to nodes
      Int = kp.integrate(s, pTot)
      if Int == 0.0:
        qp = np.zeros(2)
      else:
        pct = (kp.integrate(s, s * pTot) / Int - s[0]) / kp.cumdiff(s)
        qp = Int * np.array([1 - pct, pct])
      
      Int = kp.integrate(s, tau)
      if Int == 0.0:
        qs = np.zeros(2)
      else:
        pct = (kp.integrate(s, s * tau) / Int - s[0]) / kp.cumdiff(s)
        qs = -Int * np.array([1 - pct, pct])

      el.setPressureAndShear(qp, qs)
    
      # Calculate external force and moment for rigid body calculation
      if config.cushionForceMethod.lower() == 'integrated' or \
         config.cushionForceMethod.lower() == 'assumed':
        if config.cushionForceMethod.lower() == 'integrated':
          integrand = pExt
        elif config.cushionForceMethod.lower() == 'assumed':
          integrand = pFl

        n = map(self.getNormalVector, s)
        t = [kp.rotateVec(ni, -90) for ni in n]
  
        f = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(integrand, tau, n, t)]

        r = [np.array([pt[0] - self.parent.xCofR, pt[1] - self.parent.yCofR]) for pt in map(self.getCoordinates, s)]
        
        m = [kp.cross2(ri, fi) for ri, fi in zip(r, f)]
        
        self.D -= kp.integrate(s, np.array(zip(*f)[0]))
        self.L += kp.integrate(s, np.array(zip(*f)[1]))
        self.M += kp.integrate(s, np.array(m))
      else:
        if self.interpolator is not None:
          self.D = self.interpolator.fluid.D
          self.L = self.interpolator.fluid.L
          self.M = self.interpolator.fluid.M
    
      integrand = pC

      n = map(self.getNormalVector, s)
      t = [kp.rotateVec(ni, -90) for ni in n]

      f = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(integrand, tau, n, t)]

      r = [np.array([pt[0] - self.parent.xCofR, pt[1] - self.parent.yCofR]) for pt in map(self.getCoordinates, s)]
      
      m = [kp.cross2(ri, fi) for ri, fi in zip(r, f)]
      
      self.Da -= kp.integrate(s, np.array(zip(*f)[0]))
      self.La += kp.integrate(s, np.array(zip(*f)[1]))
      self.Ma += kp.integrate(s, np.array(m))

  def getNormalVector(self, s):
    dxds = kp.getDerivative(lambda si: self.getCoordinates(si)[0], s)
    dyds = kp.getDerivative(lambda si: self.getCoordinates(si)[1], s)

    return kp.rotateVec(kp.ang2vecd(kp.atand2(dyds, dxds)), -90)

  def plotPressureProfiles(self):
    if self.lineFluidP is not None:
      self.lineFluidP.set_data(self.getPressurePlotPoints(self.fluidS, self.fluidP))
    if self.lineAirP is not None:
      self.lineAirP.set_data(self.getPressurePlotPoints(self.airS, self.airP))
    
  def getPressurePlotPoints(self, s0, p0):

    sp = [(s,p) for s, p in zip(s0,p0) if not np.abs(p) < 1e-4]

    if len(sp) > 0:
      s0, p0 = zip(*sp)
      nVec = map(self.getNormalVector, s0)
      coords0 = [np.array(self.getCoordinates(s)) for s in s0]
      coords1 = [c + config.pScale * p * n for c, p, n in zip(coords0, p0, nVec)]
      
      return zip(*[xyi for c0, c1 in zip(coords0, coords1) for xyi in [c0, c1, np.ones(2)*np.nan]])
    else:
      return [], []

  def updateGeometry(self):
    self.setInterpFunction()
  
  def plot(self):
    for el in self.el:
      el.plot()
#    for nd in [self.node[0],self.node[-1]]:
#      nd.plot()
    self.plotPressureProfiles()

  def setAttachments(self):
    return None

  def setAngle(self):
    return None

class FlexibleSubstructure(Substructure):
  obj = []
  res = 0.0

  @classmethod
  def updateAll(cls):

    nDOF = fe.Node.count() * config.dim
    Kg = np.zeros((nDOF, nDOF))
    Fg = np.zeros((nDOF, 1))
    Ug = np.zeros((nDOF, 1))
    
    # Assemble global matrices for all substructures together
    for ss in cls.obj:
      ss.updateFluidForces()
      ss.assembleGlobalStiffnessAndForce()
      Kg += ss.K
      Fg += ss.F
    
    for nd in fe.Node.All():
      for i in range(2):
        Fg[nd.dof[i]] += nd.fixedLoad[i]
    
      
    # Determine fixed degrees of freedom
    dof = [False for _ in Fg]

    for nd in fe.Node.All():
      for dofi, fdofi in zip(nd.dof, nd.fixedDOF):
        dof[dofi] = not fdofi

    # Solve FEM linear matrix equation
    if any(dof):
      Ug[np.ix_(dof)] = np.linalg.solve(Kg[np.ix_(dof,dof)], Fg[np.ix_(dof)])
    
    cls.res = np.max(np.abs(Ug))
    
    Ug *= config.relaxFEM
    Ug *= np.min([config.maxFEMDisp / np.max(Ug), 1.0])
     
    for nd in fe.Node.All():
      nd.moveCoordinates(Ug[nd.dof[0],0], Ug[nd.dof[1],0])

    for ss in cls.obj:
      ss.updateGeometry()

  def __init__(self, Dict):
#    FlexibleSubstructure.obj.append(self)

    Substructure.__init__(self, Dict)
    self.elementType = fe.TrussElement
    self.pretension  = self.Dict.readOrDefault('pretension', -0.5)
    self.EA          = self.Dict.readOrDefault('EA', 5e7)
    
    self.K = None
    self.F = None
#    self.U = None
    config.hasFreeStructure = True

  def getResidual(self):
    return np.max(np.abs(self.U))

  def initializeMatrices(self):
    nDOF = fe.Node.count() * config.dim
    self.K = np.zeros((nDOF, nDOF))
    self.F = np.zeros((nDOF, 1))
    self.U = np.zeros((nDOF, 1))

  def assembleGlobalStiffnessAndForce(self):
    if self.K is None:
      self.initializeMatrices()
    else:
      self.K *= 0
      self.F *= 0
    for el in self.el:
      self.addLoadsFromEl(el)

  def addLoadsFromEl(self, el):  
    K, F = el.getStiffnessAndForce()
    self.K[np.ix_(el.dof,el.dof)] += K
    self.F[np.ix_(el.dof)]        += F

#  def getPtDispFEM(self):
##    if self.K is None:
##      self.initializeMatrices()
##    self.U *= 0.0
##    self.updateFluidForces()
##    self.assembleGlobalStiffnessAndForce()
#    
#    dof = [False for dof in self.F]
#    for nd in self.node:
#      for dofi, fdofi in zip(nd.dof, nd.fixedDOF):
#        dof[dofi] = not fdofi
##    if any(dof):
##      self.U[np.ix_(dof)] = np.linalg.solve(self.K[np.ix_(dof,dof)], self.F[np.ix_(dof)])
#
#    # Relax displacement and limit step if necessary
#    self.U *= config.relaxFEM
#    self.U *= np.min([config.maxFEMDisp / np.max(self.U), 1.0])
#
#    for nd in self.node:
#      nd.moveCoordinates(self.U[nd.dof[0],0], self.U[nd.dof[1],0])
#
#    self.updateGeometry()
     
  def setElementProperties(self):
    Substructure.setElementProperties(self)
    for el in self.el:
      el.setProperties(axialForce=-self.pretension, EA=self.EA)

  def updateGeometry(self):
    for el in self.el:
      el.updateGeometry()
    Substructure.setInterpFunction(self)

class RigidSubstructure(Substructure):

  def __init__(self, Dict):
    Substructure.__init__(self, Dict)
    self.elementType = fe.RigidElement

  def setAttachments(self):
    return None

  def updateAngle(self):
    return None

  def setFixedDOF(self):
    for nd in self.node:
      for j in range(config.dim):
        nd.fixedDOF[j] = True


class TorsionalSpringSubstructure(FlexibleSubstructure, RigidSubstructure):
  
  def __init__(self, Dict):
    FlexibleSubstructure.__init__(self, Dict)
    self.elementType = fe.RigidElement
    self.tipLoadPct = self.Dict.readOrDefault('tipLoadPct', 0.0)
    self.basePtPct  = self.Dict.readOrDefault('basePtPct', 1.0)
    self.springConstant = self.Dict.readOrDefault('springConstant', 1000.0)
    self.theta = 0.0
    self.Mt = 0.0
    self.MOld = None
    self.relax = self.Dict.readOrDefault('relaxAng', config.relaxRB)
    self.attachPct  = self.Dict.readOrDefault('attachPct', 0.0)
    self.attachedNode = None
    self.attachedEl   = None
    self.minimumAngle = self.Dict.readOrDefault('minimumAngle', -float('Inf'))
    self.maxAngleStep = self.Dict.readOrDefault('maxAngleStep',  float('Inf'))
    config.hasFreeStructure = True

  def generateMesh(self):
    Substructure.generateMesh(self)
    self.setFixedDOF()
    if self.basePtPct == 1.0:
      self.basePt = self.node[-1].getCoordinates()
    elif self.basePtPct == 0.0:
      self.basePt = self.node[0].getCoordinates()
    else:
      self.basePt = self.getCoordinates(self.basePtPct * self.getArcLength())
    
    self.setElementProperties()
     
    self.setAngle(self.Dict.readOrDefault('initialAngle', 0.0))

  def setAttachments(self):
    attachedSubstructureName = self.Dict.readOrDefault('attachedSubstructure', None)
    if attachedSubstructureName is not None:
      self.attachedSubstructure = Substructure.findByName(attachedSubstructureName)
    else:
      self.attachedSubstructure = None
    
    if self.Dict.readOrDefault('attachedSubstructureEnd', 'End').lower() == 'start':
      self.attachedInd = 0
    else:
      self.attachedInd = -1
    
    if self.attachedNode is None and self.attachedSubstructure is not None:
      self.attachedNode = self.attachedSubstructure.node[self.attachedInd]
      self.attachedEl   = self.attachedSubstructure.el[self.attachedInd]

  def updateFluidForces(self):
    self.fluidS = []
    self.fluidP = []
    self.airS   = []
    self.airP   = []
    self.D = 0.0
    self.L = 0.0
    self.M = 0.0
    self.Dt = 0.0
    self.Lt = 0.0
    self.Mt = 0.0
    self.Da = 0.0
    self.La = 0.0
    self.Ma = 0.0
    if self.interpolator is not None:
      sMin, sMax = self.interpolator.getMinMaxS()

    for i, el in enumerate(self.el):
      # Get pressure at end points and all fluid points along element
      nodeS = [self.nodeS[i], self.nodeS[i+1]]
      if self.interpolator is not None:
        s, pFl, tau = self.interpolator.getLoadsInRange(nodeS[0], nodeS[1])
  
        # Limit pressure to be below stagnation pressure 
        if config.pressureLimiter:
          pFl = np.min(np.hstack((pFl, np.ones_like(pFl) * config.pStag)), axis=0)
          
      else:
        s   = np.array(nodeS)
        pFl = np.zeros_like(s)
        tau = np.zeros_like(s)
      
      ss = nodeS[1]
      Pc = 0.0
      if self.interpolator is not None:
        if ss > sMax:
          Pc = self.interpolator.fluid.getUpstreamPressure()
        elif ss < sMin:
          Pc = self.interpolator.fluid.getDownstreamPressure()
      elif self.cushionPressureType == 'Total':
        Pc = config.Pc
         
      # Store fluid and air pressure components for element (for plotting)
      if i == 0:
        self.fluidS += [s[0]]
        self.fluidP += [pFl[0]]
        self.airS   += [nodeS[0]]
        self.airP   += [Pc - self.Ps]
        
      self.fluidS += [ss for ss in s[1:]]
      self.fluidP += [pp for pp in pFl[1:]]
      self.airS   += [ss for ss in nodeS[1:]]
      self.airP   += [Pc - self.Ps for _ in nodeS[1:]]

      # Apply ramp to hydrodynamic pressure
      pFl *= config.ramp**2
      
      # Add external cushion pressure to external fluid pressure
      pC = np.zeros_like(s)
      Pc = 0.0
      for ii, ss in enumerate(s):
        if self.interpolator is not None:
          if ss > sMax:
            Pc = self.interpolator.fluid.getUpstreamPressure()
          elif ss < sMin:
            Pc = self.interpolator.fluid.getDownstreamPressure()
        elif self.cushionPressureType == 'Total':
          Pc = config.Pc

        pC[ii] = Pc
      
      pInt = self.Ps * np.ones_like(s)
      
      pExt = pFl + pC
      pTot = pExt - pInt

      # Calculate external force and moment for rigid body calculation
      if config.cushionForceMethod.lower() == 'integrated' or \
         config.cushionForceMethod.lower() == 'assumed':
        if config.cushionForceMethod.lower() == 'integrated':
          integrand = pExt
        elif config.cushionForceMethod.lower() == 'assumed':
          integrand = pFl

        n = map(self.getNormalVector, s)
        t = [kp.rotateVec(ni, -90) for ni in n]
  
        fC  = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(pExt, tau, n, t)]
        fFl = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(pFl,  tau, n, t)]
        f   = fC + fFl
        print 'Cushion Lift-to-Weight: {0}'.format(fC[1] / config.W)

        r = [np.array([pt[0] - config.xCofR, pt[1] - config.yCofR]) for pt in map(self.getCoordinates, s)]
        
        m = [kp.cross2(ri, fi) for ri, fi in zip(r, f)]
        
        self.D -= kp.integrate(s, np.array(zip(*f)[0]))
        self.L += kp.integrate(s, np.array(zip(*f)[1]))
        self.M += kp.integrate(s, np.array(m))
      else:
        if self.interpolator is not None:
          self.D = self.interpolator.fluid.D
          self.L = self.interpolator.fluid.L
          self.M = self.interpolator.fluid.M
      
      # Apply pressure loading for moment calculation
#      integrand = pFl
      integrand = pTot
      n = map(self.getNormalVector, s)
      t = [kp.rotateVec(ni, -90) for ni in n]

      f = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(integrand, tau, n, t)]
      r = [np.array([pt[0] - self.basePt[0], pt[1] - self.basePt[1]]) for pt in map(self.getCoordinates, s)]
    
      m = [kp.cross2(ri, fi) for ri, fi in zip(r, f)]
      fx, fy = zip(*f)

      self.Dt += kp.integrate(s, np.array(fx))
      self.Lt += kp.integrate(s, np.array(fy))
      self.Mt += kp.integrate(s, np.array(m))
      
      integrand = pC

      n = map(self.getNormalVector, s)
      t = [kp.rotateVec(ni, -90) for ni in n]

      f = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(integrand, tau, n, t)]

      r = [np.array([pt[0] - self.parent.xCofR, pt[1] - self.parent.yCofR]) for pt in map(self.getCoordinates, s)]
      
      m = [kp.cross2(ri, fi) for ri, fi in zip(r, f)]
      
      self.Da -= kp.integrate(s, np.array(zip(*f)[0]))
      self.La += kp.integrate(s, np.array(zip(*f)[1]))
      self.Ma += kp.integrate(s, np.array(m))

    # Apply tip load
    tipC = self.getCoordinates(self.tipLoadPct * self.getArcLength())
    tipR = np.array([tipC[i] - self.basePt[i] for i in [0,1]])
    tipF = np.array([0.0, self.tipLoad]) * config.ramp
    tipM = kp.cross2(tipR, tipF)
    self.Lt += tipF[1]
    self.Mt += tipM
    
    # Apply moment from attached substructure
#    el = self.attachedEl
#    attC = self.attachedNode.getCoordinates()
#    attR = np.array([attC[i] - self.basePt[i] for i in [0,1]])
#    attF = el.axialForce * kp.ang2vec(el.gamma + 180)
#    attM = kp.cross2(attR, attF) * config.ramp
##    attM = np.min([np.abs(attM), np.abs(self.Mt)]) * kp.sign(attM)
##    if np.abs(attM) > 2 * np.abs(tipM):
##      attM = attM * np.abs(tipM) / np.abs(attM)
#    self.Mt += attM 

  def updateAngle(self):

    if np.isnan(self.Mt):
      theta = 0.0
    else:
      theta = -self.Mt

    if not self.springConstant == 0.0:
      theta /= self.springConstant

    dTheta = (theta - self.theta) * self.relax
    dTheta = np.min([np.abs(dTheta), self.maxAngleStep]) * np.sign(dTheta)
    
    self.setAngle(self.theta + dTheta)

  def setAngle(self, ang):
    dTheta = np.max([ang, self.minimumAngle]) - self.theta

    if self.attachedNode is not None and not any([nd == self.attachedNode for nd in self.node]):
      attNd = [self.attachedNode]
    else:
      attNd = []

#    basePt = np.array([c for c in self.basePt])
    basePt = np.array([c for c in self.node[-1].getCoordinates()])
    for nd in self.node + attNd:
      oldPt = np.array([c for c in nd.getCoordinates()])
      newPt = kp.rotatePt(oldPt, basePt, -dTheta)
      nd.setCoordinates(newPt[0], newPt[1])

    self.theta += dTheta
    self.residual = dTheta
    self.updateGeometry()
    print "  Deformation for substructure {0}: {1}".format(self.name, self.theta)
  
  def getResidual(self):
    return self.residual
#    return self.theta + self.Mt / self.springConstant
     
  def writeDeformation(self):
    kp.writeasdict(os.path.join(config.itDir, 'deformation_{0}.{1}'.format(self.name, config.dataFormat)),
        ['angle', self.theta])
