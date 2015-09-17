import numpy as np
import krampy as kp
from scipy.interpolate import interp1d
from scipy.optimize import fmin, fminbound, fmin_cg, fsolve, fmin_tnc, fmin_slsqp, fmin_l_bfgs_b, leastsq, anneal, fmin_powell, fmin_bfgs
import pressureElements as pe
import os, config
if config.plot:
  import matplotlib.pyplot as plt
import random

class PotentialPlaningCalculation:
  obj = []

  @classmethod
  def getTotalFreeSurfaceHeight(cls, x):
    return cls.obj[0].getFreeSurfaceHeight(x)
  
  def __init__(self):
    self.X   = None
    self.xFS = None
    self.D   = None

    self.planingSurface  = []
    self.pressureCushion = []
    self.pressurePatch   = []
    self.pressureElement = []
    self.lineFS  = None
    self.lineFSi = None
    PotentialPlaningCalculation.obj.append(self)
    self.solver = None

    self.minLen = None
    self.initLen = None

  def addPressurePatch(self, instance):
    self.pressurePatch.append(instance)
    self.pressureElement += [el for el in instance.pressureElement]

  def addPlaningSurface(self, Dict='', **kwargs):
    Dict = kp.ensureDict(Dict)
    instance = PlaningSurface(Dict, **kwargs)
    self.planingSurface.append(instance)
    self.addPressurePatch(instance)
    return instance

  def addPressureCushion(self, Dict='', **kwargs):
    Dict = kp.ensureDict(Dict)
    instance = PressureCushion(Dict, **kwargs)
    self.pressureCushion.append(instance)
    self.addPressurePatch(instance)
    return instance

  def getPlaningSurfaceByName(self, name):
    l = [surf for surf in self.planingSurface if surf.patchName==name]
    if len(l) > 0:
      return l[0]
    else:
      return None

  def printElementStatus(self):
    for el in self.pressureElement:
      el.printElement()
    raw_input()

  def calculatePressure(self):
    # Form lists of unknown and source elements
    unknownEl = [el for el in self.pressureElement if not el.isSource() and el.getWidth() > 0.0]
    nanEl     = [el for el in self.pressureElement if el.getWidth() <= 0.0]
    sourceEl  = [el for el in self.pressureElement if el.isSource()]

    # Form arrays to build linear system
    # x location of unknown elements
    X  = np.array([el.getXLoc() for el in unknownEl])
    
    # influence coefficient matrix (to be reshaped)
    A  = np.array([el.getInfluenceCoefficient(x) for x in X for el in unknownEl])
    
    # height of each unknown element above waterline
    Z  = np.array([el.getZLoc() for el in unknownEl]) - config.hWL
    
    # subtract contribution from source elements
    Z -= np.array([np.sum([el.getInfluence(x) for el in sourceEl]) for x in X])
    
#    self.printElementStatus()

    # Solve linear system
    dim = len(unknownEl)
    if not dim == 0:
      p = np.linalg.solve(np.reshape(A, (dim,dim)), np.reshape(Z, (dim,1)))[:,0]
    else:
      p = np.zeros_like(Z)

    # Apply calculated pressure to the elements
    for pi, el in zip(p, unknownEl):
      el.setPressure(pi) 
    
    for el in nanEl:
      el.setPressure(0.0)
    
  def getResidualUnknownSep(self, endPts):
    # Set length of each planing surface
    N = len(endPts) / 2
    for i, p in enumerate(self.planingSurface):
      p.setEndPts([endPts[i*N+j] for j in range(N)])

    # Update bounds of pressure cushions
    for p in self.pressureCushion:
      p.updateEndPts()
    
    # Solve for unknown pressures and output residual
    self.calculatePressure()
    
    resP = [p.getResidual() for p in self.planingSurface]
    resSlp = [float(p.getBodyDerivative(p.getBasePt()) - self.getFreeSurfaceDerivative(p.getBasePt(), 'l')) for p in self.planingSurface]
    res = [item for z in zip(resSlp, resP) for item in z]
    print '      endPts:    ', endPts
    print '      Func value:', res
    return res

  def getResidual(self, Lw):
    # Set length of each planing surface
    for Lwi, p in zip(Lw, self.planingSurface):
      p.setLength(np.min([Lwi, p.maximumLength]))

    # Update bounds of pressure cushions
    for p in self.pressureCushion:
      p.updateEndPts()

    # Solve for unknown pressures and output residual
    self.calculatePressure()

    res = np.array([p.getResidual() for p in self.planingSurface])

#    self.xHist.append(Lw * 1.0)

    print '      Lw:        ', Lw
    print '      Func value:', res

    return res

  def calculateResponse(self):
    if config.resultsFromFile:
      self.loadResponse()
    else:
      self.calculateResponse=self.calculateResponseUnknownWettedLength
      self.calculateResponseUnknownWettedLength()

  def calculateResponseUnknownWettedLength(self):
    # Reset results so they will be recalculated after new solution
    self.xFS = None
    self.X   = None
    self.xHist = []
     
    # Initialize planing surface lengths and then solve until residual is *zero*
    if len(self.planingSurface) > 0:
      for p in self.planingSurface:
        p.initializeEndPts()
  
      print '  Solving for wetted length:'
      
      if self.minLen is None:
        self.minLen  = np.array([p.getMinLength() for p in self.planingSurface])
        self.maxLen  = np.array([p.maximumLength for p in self.planingSurface])
        self.initLen = np.array([p.initialLength for p in self.planingSurface])
      else:
        for i, p in enumerate(self.planingSurface):
          L = p.getLength()
          self.initLen[i] = p.initialLength
          if ~np.isnan(L) and L - self.minLen[i] > 1e-6:# and self.solver.it < self.solver.maxIt:
            self.initLen[i] = L * 1.0
          else:
            self.initLen[i] = p.initialLength

      dxMaxDec = config.wettedLengthMaxStepPctDec * (self.initLen - self.minLen)
      dxMaxInc = config.wettedLengthMaxStepPctInc * (self.initLen - self.minLen)

      for i, p in enumerate(self.planingSurface):
        if p.initialLength == 0.0:
          self.initLen[i] = 0.0  
      
      if self.solver is None:
        self.solver = kp.RootFinderNew(self.getResidual, \
                                    self.initLen * 1.0, \
                                    config.wettedLengthSolver, \
                                    xMin=self.minLen, \
                                    xMax=self.maxLen, \
                                    errLim=config.wettedLengthTol, \
                                    dxMaxDec=dxMaxDec, \
                                    dxMaxInc=dxMaxInc, \
                                    firstStep=1e-6, \
                                    maxIt=config.wettedLengthMaxIt0, \
                                    maxJacobianResetStep=config.wettedLengthMaxJacobianResetStep, \
                                    relax=config.wettedLengthRelax)
      else:
        self.solver.maxIt = config.wettedLengthMaxIt
        self.solver.reinitialize(self.initLen * 1.0)
        self.solver.setMaxStep(dxMaxInc, dxMaxDec)
 
      if any(self.initLen > 0.0): 
        self.solver.solve()
      
#      if not self.solver.converged:
#        for el in self.pressureElement:
#          if not el.isSource():
#            el.setPressure(0.0)

      self.calculatePressureAndShearProfile()
#      self.plotResidualsOverRange()

  def plotResidualsOverRange(self):
    self.storeLen = np.array([p.getLength() for p in self.planingSurface])

    xx, yy = zip(*self.xHist)
    
    N = 10
    L = np.linspace(0.001, 0.25, N)
    X = np.zeros((N,N))
    Y = np.zeros((N,N))
    Z1 = np.zeros((N,N))
    Z2 = np.zeros((N,N))

    for i, Li in enumerate(L):
      for j, Lj in enumerate(L):
        x = np.array([Li, Lj])
        y = self.getResidual(x)
        
        X[i,j]  = x[0]
        Y[i,j]  = x[1]
        Z1[i,j] = y[0]
        Z2[i,j] = y[1]
         
    for i, Zi, seal in zip([2,3], [Z1,Z2], ['bow','stern']):
      plt.figure(i)
      plt.contourf(X, Y, Zi, 50)
      plt.gray()
      plt.colorbar()
      plt.contour(X, Y, Z1, np.array([0.0]), colors='b')
      plt.contour(X, Y, Z2, np.array([0.0]), colors='b')
      plt.plot(xx, yy, 'k.-')
      if self.solver.converged:
        plt.plot(xx[-1], yy[-1], 'y*', markersize=10)
      else:
        plt.plot(xx[-1], yy[-1], 'rx', markersize=8)
  
      plt.title('Residual for {0} seal trailing edge pressure'.format(seal))
      plt.xlabel('Bow seal length')
      plt.ylabel('Stern seal length')
      plt.savefig('{0}SealResidual_{1}.png'.format(seal, config.it), format='png')
      plt.clf()
     
    self.getResidual(self.storeLen)

#    plt.show()

#      def f(x):
#        R = self.getResidual(x)
#        pen = 1 / (500.0 * (x - self.minLen))
#        pen = 0.0
#        return np.sum(np.abs(R**2 + pen))
#      fmin_bfgs(f, self.initLen)

#      solve = diagbroyden
#      solve(self.getResidual, self.initLen)
#      fCon = lambda x: x - self.minLen
#      fObj = lambda x: np.sum(np.linalg.norm(self.getResidual(x)))
#      fObj = lambda x: self.getResidual(x) + np.exp(-(x - self.minLen))
#      fmin_tnc(fObj, self.initLen, bounds=[(xm, float('Inf')) for xm in self.minLen], approx_grad=True, ftol=config.wettedLengthTol)

#      fsolve(self.getResidual, self.initLen, xtol=1e-4, factor=10)
#      f = lambda x: np.sum(np.linalg.norm(self.getResidual(x)))
#      fsolve(fObj, self.initLen)

#      self.calculatePressureAndShearProfile()

  def calculatePressureAndShearProfile(self):
    if self.X is None:
      if config.shearCalc:
        for p in self.planingSurface:
          p.calculateShearStress()
  
      # Calculate forces on each patch
      for p in self.pressurePatch:
        p.calculateForces()
  
      # Calculate pressure profile
      if len(self.pressurePatch) > 0:
        self.X   = np.sort(np.unique(np.hstack([p.getPts() for p in self.pressurePatch])))
        self.p   = np.zeros_like(self.X)
        self.tau = np.zeros_like(self.X)
      else:
        self.X = np.array([-1e6,1e6])
        self.p = np.zeros_like(self.X)
        self.tau = np.zeros_like(self.X)
      for el in self.pressureElement:
        if el.isOnBody():
          self.p[el.getXLoc() == self.X]   += el.getPressure()
          self.tau[el.getXLoc() == self.X] += el.getShearStress()
  
      # Calculate total forces as sum of forces on each patch
      for var in ['D', 'Dp', 'Df', 'Dw', 'L', 'Lp', 'Lf', 'M']:
        exec('self.{0} = sum([p.{0} for p in self.pressurePatch])'.format(var)) in globals(), locals()

      f = self.getFreeSurfaceHeight
      xo = -10.1 * config.lam
      xTrough, = fmin(f, xo, disp=False)
      xCrest,  = fmin(lambda x: -f(x), xo, disp=False)
      self.Dw = 0.0625 * config.rho * config.g * (f(xCrest) - f(xTrough))**2
  
      # Calculate center of pressure
      self.xBar = kp.integrate(self.X, self.p * self.X) / self.L
      if config.plotPressure:
        self.plotPressure()

  def calculateFreeSurfaceProfile(self):
    if self.xFS is None:
      xFS = []
      # Grow points upstream and downstream from first and last plate
      for surf in self.planingSurface:  
        if surf.getLength() > 0:
          pts = surf.getPts()
          xFS.append(kp.growPoints(pts[1], pts[0], config.xFSMin, config.growthRate))
          xFS.append(kp.growPoints(pts[-2], pts[-1], config.xFSMax, config.growthRate))

      # Add points from each planing surface
      fsList = [patch.getPts() for patch in self.planingSurface if patch.getLength() > 0]
      if len(fsList) > 0:
        xFS.append(np.hstack(fsList))
  
      # Add points from each pressure cushion
      xFS.append(np.linspace(config.xFSMin, config.xFSMax, 100))
#      for patch in self.pressureCushion:
#        if patch.neighborDown is not None:
#          ptsL = patch.neighborDown.getPts()
#        else:
#          ptsL = np.array([patch.endPt[0] - 0.01, patch.endPt[0]])
#
#        if patch.neighborDown is not None:
#          ptsR = patch.neighborUp.getPts()
#        else:
#          ptsR = np.array([patch.endPt[1], patch.endPt[1] + 0.01])
# 
#        xEnd = ptsL[-1] + 0.5 * patch.getLength()
#  
#        xFS.append(kp.growPoints(ptsL[-2], ptsL[-1], xEnd, config.growthRate))
#        xFS.append(kp.growPoints(ptsR[1],  ptsR[0],  xEnd, config.growthRate))
  
      # Sort x locations and calculate surface heights
      if len(xFS) > 0:
        self.xFS = np.sort(np.unique(np.hstack(xFS)))
        self.zFS = np.array(map(self.getFreeSurfaceHeight, self.xFS))
      else:
        self.xFS = np.array([config.xFSMin, config.xFSMax])
        self.zFS = np.zeros_like(self.xFS)

  def getFreeSurfaceHeight(self, x):
    return sum([patch.getFreeSurfaceHeight(x) for patch in self.pressurePatch])

  def getFreeSurfaceDerivative(self, x, direction='c'):
    ddx = kp.getDerivative(self.getFreeSurfaceHeight, x, direction='c')
    return ddx

  def writeResults(self):
    # Post-process results from current solution
#    self.calculatePressureAndShearProfile()
    self.calculateFreeSurfaceProfile()
   
    if self.D is not None: 
      self.writeForces()
    if self.X is not None:
      self.writePressureAndShear()
    if self.xFS is not None:
      self.writeFreeSurface()

  def writePressureAndShear(self):
    if len(self.pressureElement) > 0:
      kp.writeaslist(os.path.join(config.itDir, 'pressureAndShear.{0}'.format(config.dataFormat)),
                     ['x [m]'   , self.X],
                     ['p [Pa]'  , self.p],
                     ['tau [Pa]', self.tau])

  def writeFreeSurface(self):
    if len(self.pressureElement) > 0:
      kp.writeaslist(os.path.join(config.itDir, 'freeSurface.{0}'.format(config.dataFormat)),
                     ['x [m]', self.xFS],
                     ['y [m]', self.zFS])

  def writeForces(self):
    if len(self.pressureElement) > 0:
      kp.writeasdict(os.path.join(config.itDir, 'forces_total.{0}'.format(config.dataFormat)),
                     ['Drag'     , self.D],
                     ['WaveDrag' , self.Dw],
                     ['PressDrag', self.Dp],
                     ['FricDrag' , self.Df],
                     ['Lift'     , self.L],
                     ['PressLift', self.Lp],
                     ['FricLift',  self.Lf],
                     ['Moment',   self.M])

    for patch in self.pressurePatch:
      patch.writeForces()

  def loadResponse(self):
    self.loadForces()
    self.loadPressureAndShear()
    self.loadFreeSurface()

  def loadPressureAndShear(self):
    self.x, self.p, self.tau = np.loadtxt(os.path.join(config.itDir, 'pressureAndShear.{0}'.format(config.dataFormat)), unpack=True)
    for el in [el for patch in self.planingSurface for el in patch.pressureElement]:
      compare = np.abs(self.x - el.getXLoc()) < 1e-6
      if any(compare):
        el.setPressure(self.p[compare][0])
        el.setShearStress(self.tau[compare][0])

    for p in self.planingSurface:
      p.calculateForces()

  def loadFreeSurface(self):
    try:
      Data = np.loadtxt(os.path.join(config.itDir, 'freeSurface.{0}'.format(config.dataFormat)))
      self.xFS = Data[:,0]
      self.zFS = Data[:,1]
    except:
      self.yFS = 0.0 * self.xFS

  def loadForces(self):
    K = kp.Dictionary(os.path.join(config.itDir, 'forces_total.{0}'.format(config.dataFormat)))
    self.D  = K.readOrDefault('Drag', 0.0)
    self.Dw = K.readOrDefault('WaveDrag', 0.0)
    self.Dp = K.readOrDefault('PressDrag', 0.0)
    self.Df = K.readOrDefault('FricDrag', 0.0)
    self.L  = K.readOrDefault('Lift', 0.0)
    self.Lp = K.readOrDefault('PressLift', 0.0)
    self.Lf = K.readOrDefault('FricLift', 0.0)
    self.M  = K.readOrDefault('Moment', 0.0)

    for patch in self.pressurePatch:
      patch.loadForces()

  def plotPressure(self):
    plt.figure(figsize=(5.0,5.0))
    plt.xlabel(r'$x/L_i$')
    plt.ylabel(r'$p/(1/2\rho U^2)$')

    for el in self.pressureElement:
      el.plot()

    plt.plot(self.X, self.p, 'k-')
    plt.plot(self.X, self.tau*1000, 'c--')

    # Scale y axis by stagnation pressure
    for line in plt.gca().lines:
      x, y = line.get_data()
      line.set_data(x / config.Lref * 2, y / config.pStag)

    plt.xlim([-1.0,1.0])
#    plt.xlim(kp.minMax(self.X / config.Lref * 2))
    plt.ylim([0.0, np.min([1.0,1.2*np.max(self.p / config.pStag)])])
    plt.savefig('pressureElements.{0}'.format(config.figFormat), format=config.figFormat)
#    plt.show()
    plt.figure(1)

  def plotFreeSurface(self):
    self.calculateFreeSurfaceProfile()
    if self.lineFS is not None:
      self.lineFS.set_data(self.xFS, self.zFS)
    endPts = np.array([config.xFSMin, config.xFSMax])
    if self.lineFSi is not None:
      self.lineFSi.set_data(endPts, config.hWL * np.ones_like(endPts))


class PressurePatch():

  def __init__(self, **kwargs):
    self.pressureElement = []
    self.endPt = np.zeros(2)
    self.kuttaUnknown = False

    self.D  = 0.0
    self.Dw = 0.0
    self.Dp = 0.0
    self.Df = 0.0
    self.L  = 0.0
    self.Lp = 0.0
    self.Lf = 0.0
    self.M  = 0.0
     
    self.neighborDown = kwargs.get('neighborDown', None)
    self.neighborUp   = kwargs.get('neighborUp',   None)

  def setNeighbor(self, **kwargs):
    self.neighborDown = kwargs.get('down', self.neighborDown)
    if self.neighborDown is not None:
      self.neighborDown.neighborUp = self

    self.neighborUp   = kwargs.get('up', self.neighborUp)
    if self.neighborUp is not None:
      self.neighborUp.neighborDown = self

  def setEndPts(self, endPt):
    self.endPt = endPt

  def getEndPts(self):
    return self.endPt

  def getBasePt(self):
    return self.endPt[0]

  def getLength(self):
    return self.endPt[1] - self.endPt[0]

  def setKuttaUnknown(self, unknown):
    self.kuttaUnknown = unknown

  def isKuttaUnknown(self):
    return self.kuttaUnknown

  def resetElements(self):
    x = self.getPts()
    for i, el in enumerate(self.pressureElement):
      el.setXLoc(x[i])
      if not el.isSource():
        el.setZLoc(self.interpolator.getBodyHeight(x[i]))

      if isinstance(el, pe.CompleteTriangularPressureElement):
        el.setWidth([x[i] - x[i-1], x[i+1] - x[i]])
      elif isinstance(el, pe.ForwardHalfTriangularPressureElement) \
        or isinstance(el, pe.NegativeForwardHalfTriangularPressureElement) \
        or isinstance(el, pe.TransomPressureElement):
        el.setWidth(x[i+1] - x[i])
      elif isinstance(el, pe.AftHalfTriangularPressureElement):
        el.setWidth(x[i] - x[i-1])
      elif isinstance(el, pe.AftSemiInfinitePressureBand):
        el.setWidth(np.inf)
      else:
        print 'Invalid Element Type!'
        return -1

  def getFreeSurfaceHeight(self, x):
    return sum([el.getInfluence(x) for el in self.pressureElement])

  def calculateWaveDrag(self): 
    f = self.getFreeSurfaceHeight
    xo = -10.1 * config.lam
    xTrough = fmin(f, xo, disp=False)[0]
    xCrest  = fmin(lambda x: -f(x), xo, disp=False)[0]
    self.Dw = 0.0625 * config.rho * config.g * (f(xCrest) - f(xTrough))**2

  def printForces(self):
    print 'Forces and Moment for {0}:'.format(self.patchName)
    print '    Total Drag [N]      : {0:6.4e}'.format(self.D)
    print '    Wave Drag [N]       : {0:6.4e}'.format(self.Dw)
    print '    Pressure Drag [N]   : {0:6.4e}'.format(self.Dp)
    print '    Frictional Drag [N] : {0:6.4e}'.format(self.Df)
    print '    Total Lift [N]      : {0:6.4e}'.format(self.L)
    print '    Pressure Lift [N]   : {0:6.4e}'.format(self.Lp)
    print '    Frictional Lift [N] : {0:6.4e}'.format(self.Lf)
    print '    Moment [N-m]        : {0:6.4e}'.format(self.M)

  def writeForces(self):
    kp.writeasdict(os.path.join(config.itDir, 'forces_{0}.{1}'.format(self.patchName, config.dataFormat)),
                   ['Drag'     , self.D],
                   ['WaveDrag' , self.Dw],
                   ['PressDrag', self.Dp],
                   ['FricDrag' , self.Df],
                   ['Lift'     , self.L],
                   ['PressLift', self.Lp],
                   ['FricLift' , self.Lf],
                   ['Moment'   , self.M],
                   ['BasePt'   , self.getBasePt()],
                   ['Length'   , self.getLength()])

  def loadForces(self):
    K = kp.Dictionary(os.path.join(config.itDir, 'forces_{0}.{1}'.format(self.patchName, config.dataFormat)))
    self.D  = K.readOrDefault('Drag', 0.0)
    self.Dw = K.readOrDefault('WaveDrag', 0.0)
    self.Dp = K.readOrDefault('PressDrag', 0.0)
    self.Df = K.readOrDefault('FricDrag', 0.0)
    self.L  = K.readOrDefault('Lift', 0.0)
    self.Lp = K.readOrDefault('PressLift', 0.0)
    self.Lf = K.readOrDefault('FricLift', 0.0)
    self.M  = K.readOrDefault('Moment', 0.0)
    self.setBasePt(K.readOrDefault('BasePt', 0.0))
    self.setLength(K.readOrDefault('Length', 0.0))


class PressureCushion(PressurePatch):
  count = 0

  def __init__(self, Dict, **kwargs):
    PressurePatch.__init__(self, **kwargs)
    self.index = PressureCushion.count
    PressureCushion.count += 1
    
    self.Dict = Dict

    self.patchName       = self.Dict.readOrDefault('pressureCushionName', 'pressureCushion{0}'.format(self.index))
    self.cushionType     = self.Dict.readOrDefault('cushionType', '')
    self.cushionPressure = kwargs.get('Pc', self.Dict.readLoadOrDefault('cushionPressure', 0.0))

    upstreamSurf   = PlaningSurface.findByName(self.Dict.readOrDefault('upstreamPlaningSurface', ''))
    downstreamSurf = PlaningSurface.findByName(self.Dict.readOrDefault('downstreamPlaningSurface', ''))

    self.setNeighbor(down=downstreamSurf, up=upstreamSurf)
   
    if self.neighborDown is not None: 
      self.neighborDown.setUpstreamPressure(self.cushionPressure)
    if self.neighborUp is not None:
      self.neighborUp.setKuttaPressure(self.cushionPressure)

    if self.cushionType == 'infinite':
      # Dummy element, will have 0 pressure
      self.pressureElement += [pe.AftSemiInfinitePressureBand(source=True, onBody=False)]
      self.pressureElement += [pe.AftSemiInfinitePressureBand(source=True, onBody=False)]
      self.downstreamLoc = -1000.0 # doesn't matter where
  
    else:
      self.pressureElement += [pe.ForwardHalfTriangularPressureElement(source=True, onBody=False)]
  
      Nfl = self.Dict.readOrDefault('numElements', 10)
      self.sf = self.Dict.readOrDefault('smoothingFactor', np.nan)
      for n in [self.neighborDown, self.neighborUp]:
        if n is None and ~np.isnan(self.sf):
          self.pressureElement += [pe.CompleteTriangularPressureElement(source=True) for _ in range(Nfl)]
        
      self.upstreamLoc   = self.Dict.readLoadOrDefault('upstreamLoc',   0.0)
      self.downstreamLoc = self.Dict.readLoadOrDefault('downstreamLoc', 0.0)
  
      self.pressureElement += [pe.AftHalfTriangularPressureElement(source=True, onBody=False)]

    self.updateEndPts()

  def setBasePt(self, x):
    self.endPt[1] = x

  def setPressure(self, p):
    self.cushionPressure = p

  def getPressure(self):
    return self.cushionPressure

  def getPts(self):
    if self.cushionType == 'smoothed':
      if np.isnan(self.sf):
        return np.array([self.downstreamLoc, self.upstreamLoc])
      else:
        N = len(self.pressureElement) + 2
        addWidth = np.arctanh(0.99) * self.getLength() / (2*self.sf)
        addL = np.linspace(-addWidth, addWidth, N / 2)
        x = np.hstack((self.downstreamLoc + addL, self.upstreamLoc + addL))
        return x
    else:
      return self.getEndPts()

  def getBasePt(self):
    if self.neighborUp is None:
      return PressurePatch.getBasePt(self)
    else:
      return self.neighborUp.getPts()[0]

  def updateEndPts(self):
    if self.neighborDown is not None:
      self.downstreamLoc = self.neighborDown.getPts()[-1]
    if self.neighborUp is not None:
      self.upstreamLoc = self.neighborUp.getPts()[0]

    self.setEndPts([self.downstreamLoc, self.upstreamLoc]) 

    self.resetElements()

    for elNum, el in enumerate(self.pressureElement):
      if self.cushionType == 'smoothed' and ~np.isnan(self.sf):
        alf = 2 * self.sf / self.getLength()
        el.setSourcePressure(0.5 * self.getPressure() * (np.tanh(alf * el.getXLoc()) - np.tanh(alf * (el.getXLoc() - self.getLength()))))
      # for infinite pressure cushion, first element is dummy, set to zero, second is semiInfinitePressureBand and set to cushion pressure
      elif self.cushionType == 'infinite':
        if elNum == 0:
          el.setSourcePressure(0)
        else:
          el.setSourcePressure(self.getPressure())
      else:
        el.setSourcePressure(self.getPressure())

  def setLength(self, length):
    self.endPt[0] = self.endPt[1] - length

  def calculateForces(self):
    self.calculateWaveDrag()


class PlaningSurface(PressurePatch):
  count = 0
  obj = []
   
  @classmethod
  def findByName(cls, name):
    if not name == '':
      matches = [o for o in cls.obj if o.patchName == name]
      if len(matches) > 0:
        return matches[0]
      else:
        return None
    else:
      return None

  def __init__(self, Dict, **kwargs):
    PressurePatch.__init__(self)
    self.index = PlaningSurface.count
    PlaningSurface.count += 1
    PlaningSurface.obj.append(self)
      
    self.Dict = Dict
    self.patchName        = self.Dict.readOrDefault('substructureName', '')
    Nfl                   = self.Dict.readOrDefault('Nfl', 0)
    self.pointSpacing     = self.Dict.readOrDefault('pointSpacing', 'linear')
    self.initialLength    = self.Dict.readOrDefault('initialLength', None)
    self.minimumLength    = self.Dict.readOrDefault('minimumLength', 0.0)
    self.maximumLength    = self.Dict.readOrDefault('maximumLength', float('Inf'))
    self.springConstant   = self.Dict.readOrDefault('springConstant', 1e4)
    self.kuttaPressure    = kwargs.get('kuttaPressure',    self.Dict.readLoadOrDefault('kuttaPressure', 0.0))
    self.upstreamPressure = kwargs.get('upstreamPressure', self.Dict.readLoadOrDefault('upstreamPressure', 0.0))
    self.upstreamPressureCushion = None
     
    self.isInitialized = False
    self.active        = True
    self.kuttaUnknown  = True
    self.sprung = self.Dict.readOrDefault('sprung', False)
    if self.sprung:
      self.initialLength = 0.0
      self.minimumLength = 0.0
      self.maximumLength = 0.0

#    self.getResidual = self.getTransomHeightResidual
#    self.pressureElement += [pe.TransomPressureElement(onBody=False)]
    
    self.getResidual = self.getPressureResidual
    self.pressureElement += [pe.NegativeForwardHalfTriangularPressureElement(onBody=False)]
    
    self.pressureElement += [pe.ForwardHalfTriangularPressureElement(source=True, pressure=self.kuttaPressure)]
    self.pressureElement += [pe.CompleteTriangularPressureElement() for _ in range(Nfl - 1)]
    self.pressureElement += [pe.AftHalfTriangularPressureElement(source=True, pressure=self.upstreamPressure)]
    
    for el in self.pressureElement:
      el.setParent(self)

    # Define point spacing
    if self.pointSpacing == 'cosine':
      self.pct = 0.5 * (1 - kp.cosd(np.linspace(0, 180, Nfl + 1)))
    else:
      self.pct = np.linspace(0.0, 1.0, Nfl + 1)
    self.pct /= self.pct[-2]
    self.pct = np.hstack((0.0,self.pct))

    self.xBar = 0.0

  def setInterpolator(self, interpolator):
    self.interpolator = interpolator

  def getMinLength(self):
    return self.minimumLength

  def initializeEndPts(self):
    self.setBasePt(self.interpolator.getSeparationPoint()[0])

    if not self.isInitialized:
      if self.initialLength is None:
        self.setLength(self.interpolator.getImmersedLength() - self.getBasePt())
      else:
        self.setLength(self.initialLength)
      self.isInitialized = True

  def resetElements(self):
    PressurePatch.resetElements(self)
    x = self.getPts()
    self.pressureElement[0].setWidth(x[2] - x[0])

  def setKuttaPressure(self, p):
    self.kuttaPressure = p

  def setUpstreamPressure(self, p):
    self.upstreamPressure = p
    self.pressureElement[-1].setSourcePressure(p)
  
  def getKuttaPressure(self):
    return self.kuttaPressure

  def setLength(self, Lw):
    Lw = np.max([Lw, 0.0])
   
    x0 = self.interpolator.getSeparationPoint()[0]
    self.setEndPts([x0, x0 + Lw])

  def setBasePt(self, x):
    self.endPt[0] = x

  def setEndPts(self, endPt):
    self.endPt = endPt
    self.resetElements()

  def getPts(self):
    return self.pct * self.getLength() + self.endPt[0]

  def getPressureResidual(self):
    if self.getLength() <= 0.0:
      return 0.0
    else:
      return self.pressureElement[0].getPressure() / config.pStag
  
  def getTransomHeightResidual(self):
    if self.getLength() <= 0.0:
      return 0.0
    else:
      return (self.pressureElement[0].getZLoc() - self.pressureElement[0].getPressure()) * config.rho * config.g / config.pStag
  
  def getUpstreamPressure(self):
    if self.neighborUp is not None:
      return self.neighborUp.getPressure()
    else:
      return 0.0

  def getDownstreamPressure(self):
    if self.neighborDown is not None:
      return self.neighborDown.getPressure()
    else:
      return 0.0

  def calculateForces(self):
    if self.getLength() > 0.0:
      el = [el for el in self.pressureElement if el.isOnBody()]
      self.x   = np.array([eli.getXLoc()        for eli in el])
      self.p   = np.array([eli.getPressure()    for eli in el])
      self.p  += self.pressureElement[0].getPressure()
      self.tau = np.array([eli.getShearStress() for eli in el])
      self.s   = np.array([self.interpolator.getSFixedX(xx) for xx in self.x])
          
      self.fP   = interp1d(self.x, self.p,   bounds_error=False, fill_value=0.0)
      self.fTau = interp1d(self.x, self.tau, bounds_error=False, fill_value=0.0)
      
      AOA = kp.atand(self.getBodyDerivative(self.x))

      self.Dp =  kp.integrate(self.s, self.p   * kp.sind(AOA))
      self.Df =  kp.integrate(self.s, self.tau * kp.cosd(AOA))
      self.Lp =  kp.integrate(self.s, self.p   * kp.cosd(AOA))
      self.Lf = -kp.integrate(self.s, self.tau * kp.sind(AOA))
      self.D  = self.Dp + self.Df
      self.L  = self.Lp + self.Lf
      self.M  = kp.integrate(self.x, self.p * kp.cosd(AOA) * (self.x - config.xCofR))
    else:
      self.Dp = 0.0
      self.Df = 0.0
      self.Lp = 0.0
      self.Lf = 0.0
      self.D  = 0.0
      self.L  = 0.0
      self.M  = 0.0
      self.x = []
    if self.sprung:
      self.applySpring()

    self.calculateWaveDrag()

  def getLoadsInRange(self, x0, x1):
    # Get indices within range unless length is zero
    if self.getLength() > 0.0:
      ind = np.nonzero((self.x > x0) * (self.x < x1))[0]
    else:
      ind = []
    
    # Output all values within range
    if not ind == []:
      x   = np.hstack((        x0,    self.x[ind],           x1))
      p   = np.hstack((self.fP(x0),   self.p[ind],   self.fP(x1)))
      tau = np.hstack((self.fTau(x0), self.tau[ind], self.fTau(x1)))
    else:
      x   = np.array([x0, x1])
      p   = np.zeros_like(x)
      tau = np.zeros_like(x)
    return x, p, tau
    

  def applySpring(self):
    xs = self.pressureElement[0].getXLoc()
    zs = self.pressureElement[0].getZLoc()
    disp = zs - PotentialPlaningCalculation.getTotalFreeSurfaceHeight(xs)
    Fs = -self.springConstant * disp
    self.L += Fs
    self.M += Fs * (xs - config.xCofR)

  def calculateShearStress(self):
    def shearStressFunc(xx):
      if xx == 0.0:
        return 0.0
      else:
        Rex = config.U * xx / config.nu
        tau = 0.332 * config.rho * config.U**2 * Rex**-0.5
        if np.isnan(tau):
          return 0.0
        else:
          return tau

    x = self.getPts()[0:-1]
    s = np.array([self.interpolator.getSFixedX(xx) for xx in x])
    s = s[-1] - s

    for si, el in zip(s, self.pressureElement):
      el.setShearStress(shearStressFunc(si))

  def getBodyDerivative(self, x, direction='r'):
    if isinstance(x, float):
      x = [x]
    return np.array(map(lambda xx: kp.getDerivative(self.interpolator.getBodyHeight, xx, direction), x))
