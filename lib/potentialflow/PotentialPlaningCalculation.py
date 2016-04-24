import os
import general.config as config

import numpy as np
import general.krampy as kp

from scipy.optimize import fmin, fminbound, fmin_cg, fsolve, fmin_tnc, fmin_slsqp, fmin_l_bfgs_b, leastsq, anneal, fmin_powell, fmin_bfgs

from PressurePatch import PlaningSurface
from PressurePatch import PressureCushion

if config.plot:
    import matplotlib.pyplot as plt


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
