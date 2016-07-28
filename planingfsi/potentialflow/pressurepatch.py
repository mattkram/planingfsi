import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fmin

import planingfsi.config as config
import planingfsi.krampy as kp

import pressureelement as pe

if config.plot:
    import matplotlib.pyplot as plt


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
