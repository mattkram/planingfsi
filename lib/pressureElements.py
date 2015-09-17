import numpy as np
import krampy as kp
from scipy.special import sici
import config
if config.plot:
  import matplotlib.pyplot as plt

#from pressureElementFunctions import getFG, getFunc1, getFunc2, getFunc3, evalLR

def getFG(lam): 
  lam = abs(lam)
  (Si, Ci) = sici(lam)
  (S,  C ) = np.sin(lam), np.cos(lam)
  f = C * (0.5 * np.pi - Si) + S * Ci
  g = S * (0.5 * np.pi - Si) - C * Ci
  return f, g

def getFunc1(lam, fg):
  if lam > 0:
    return (fg[1] + np.log(lam)) / np.pi
  else:
    return (fg[1] + np.log(-lam)) / np.pi + (2*np.sin(lam) - lam)
#  return (fg[1] + np.log(np.abs(lam))) / np.pi + kp.heaviside(-lam) * (2*np.sin(lam) - lam)

def getFunc2(lam, fg):
  return  kp.sign(lam) * fg[0] / np.pi + kp.heaviside(-lam) * (2*np.cos(lam) - 1)

def getFunc3(lam, fg):
  return -kp.sign(lam) * fg[0] / np.pi - 2 * kp.heaviside(-lam) * np.cos(lam) - kp.heaviside(lam)

def evalLR(f, x, dx=1e-6):
  return (f(x+dx) + f(x-dx)) / 2

class PressureElement():
  
  obj = []
  
  @classmethod
  def All(cls):
    return [o for o in cls.obj]
  
  def __init__(self, **kwargs):
    PressureElement.obj.append(self)

    self.p      = kwargs.get('pressure', np.nan)
    self.tau    = kwargs.get('shear',    0.0)
    self.xLoc   = kwargs.get('xLoc',     np.nan)
    self.zLoc   = kwargs.get('zLoc',     np.nan)
    self.width  = kwargs.get('width',    np.nan)
    self.widthR = self.width
    self.widthL = self.width
    self.source = kwargs.get('source',   False)
    self.onBody = False
    self.parent = None
   
  def setParent(self, parent):
    self.parent = parent
        
  def isSource(self):
    return self.source
  
  def setSource(self, source):
    self.source = source
  
  def setWidth(self, width):
    self.width  = width
    self.widthR = self.width
    self.widthL = self.width
      
  def getWidth(self):  
    return self.width
  
  def setXLoc(self, xLoc):
    self.xLoc = xLoc
  
  def getXLoc(self):
    return self.xLoc
  
  def setZLoc(self, zLoc):
    self.zLoc = zLoc
  
  def getZLoc(self):
    return self.zLoc

  def setPressure(self, p):
    self.p = p
    
  def setSourcePressure(self, p):
    self.setPressure(p)
    self.setSource(True)
    self.setZLoc(np.nan)
    
  def setShearStress(self, tau):
    self.tau = tau
    
  def getPressure(self):
    return self.p
  
  def getShearStress(self):
    return self.tau
    
  def isOnBody(self):
    return self.onBody
        
  def getInfluenceCoefficient(self, x):
    if not self.getWidth() == 0:
      return self.getK(x - self.xLoc) / (config.rho * config.g)
    else:
      return 0.0 
    
  def getInfluence(self, x):
    return self.getInfluenceCoefficient(x) * self.getPressure()

  def getPressureFun(self, x):
    if not self.getWidth() == 0:
      return self.pressureFun(x - self.xLoc)
    else:
      return 0.0
    
  def getK(self, x):
    if x == 0.0 or x == self.widthR or x == -self.widthL:
      return evalLR(self.influence, x)
    else:
      return self.influence(x)

  def pressureFun(self, xx):
    return 0.0
  
  def printElement(self):
    print '{0}: (x,z) = ({1}, {2}), width = {3}, source = {4}, p = {5}, onBody = {6}'.format(self.__class__.__name__, self.getXLoc(), self.getZLoc(), self.getWidth(), self.isSource(), self.getPressure(), self.isOnBody())
        
    
class AftHalfTriangularPressureElement(PressureElement):
  
  def __init__(self, **kwargs):
    PressureElement.__init__(self, **kwargs)
    self.onBody = kwargs.get('onBody', True)
    
  def influence(self, xx):
    Lambda0 = config.k0 *  xx
    fg = getFG(Lambda0)
    K  = getFunc1(Lambda0, fg) / (self.width * config.k0)
    K += getFunc2(Lambda0, fg)

    Lambda2 = config.k0 * (xx + self.width)
    fg = getFG(Lambda2)
    K -= getFunc1(Lambda2, fg) / (self.width * config.k0)

    return K

  def pressureFun(self, xx):
    if xx < -self.getWidth() or xx > 0.0:
      return 0.0
    else:
      return self.getPressure() * (1 + xx / self.getWidth())
  
  def plot(self):
    x = self.getXLoc() - np.array([self.getWidth(), 0])
    p = np.array([0, self.p])
    
    col = 'g'
    plt.plot(x, p, col + '-')
    plt.plot([x[1], x[1]], [0.0, p[1]], col + '--')


class ForwardHalfTriangularPressureElement(PressureElement):
  
  def __init__(self, **kwargs):
    PressureElement.__init__(self, **kwargs)
    self.onBody = kwargs.get('onBody', True)
    
  def influence(self, xx):
    Lambda0 = config.k0 *  xx
    fg = getFG(Lambda0)
    K  = getFunc1(Lambda0, fg) / (self.width * config.k0)
    K -= getFunc2(Lambda0, fg)

    Lambda1 = config.k0 * (xx - self.width)
    fg = getFG(Lambda1)
    K -= getFunc1(Lambda1, fg) / (self.width * config.k0)
    return K
  
  def pressureFun(self, xx):
    if xx > self.getWidth() or xx < 0.0:
      return 0.0
    else:
      return self.getPressure() * (1 - xx / self.getWidth())
  
  def plot(self):
    x  = np.ones(2) * self.getXLoc() + np.array([0, self.getWidth()])
    p = np.array([self.p, 0])
    
    col = 'b'
    plt.plot(x, p, col + '-')
    plt.plot([x[0], x[0]], [0.0, p[0]], col + '--')


class NegativeForwardHalfTriangularPressureElement(PressureElement):
  
  def __init__(self, **kwargs):
    PressureElement.__init__(self, **kwargs)
    self.onBody = kwargs.get('onBody', True)
    
  def influence(self, xx):
    Lambda0 = config.k0 *  xx
    fg = getFG(Lambda0)
    K  = getFunc1(Lambda0, fg) / (self.width * config.k0)
    K -= getFunc2(Lambda0, fg)

    Lambda1 = config.k0 * (xx - self.width)
    fg = getFG(Lambda1)
    K -= getFunc1(Lambda1, fg) / (self.width * config.k0)
    return -K
  
  def plot(self):
    x  = np.ones(2) * self.getXLoc() + np.array([0, self.getWidth()])
    p = np.array([self.p, 0])
    
    col = 'b'
    plt.plot(x, p, col + '-')
    plt.plot([x[0], x[0]], [0.0, p[0]], col + '--')


class TransomPressureElement(PressureElement):
  
  def getInfluenceCoefficient(self, x):
    if kp.inRange(x, self.parent.getEndPts()):
      return -1.0 / (config.rho * config.g)
    else:
      return 0.0 
 
  def plot(self):
    return None 

class CompleteTriangularPressureElement(PressureElement):
  
  def __init__(self, **kwargs):
    PressureElement.__init__(self, **kwargs)
    self.onBody = kwargs.get('onBody', True)
    
  def setWidth(self, width):
    PressureElement.setWidth(self, np.sum(width))
    self.widthL = width[0]
    self.widthR = width[1]
  
  def influence(self, xx):
    Lambda0 = config.k0 *  xx
    fg = getFG(Lambda0)
    K  = getFunc1(Lambda0, fg) * self.width / (self.widthR * self.widthL)

    Lambda1 = config.k0 * (xx - self.widthR)
    fg = getFG(Lambda1)
    K -= getFunc1(Lambda1, fg) / self.widthR
    
    Lambda2 = config.k0 * (xx + self.widthL)
    fg = getFG(Lambda2)
    K -= getFunc1(Lambda2, fg) / self.widthL
    return K / config.k0
  
  def pressureFun(self, xx):
    if xx > self.widthR or xx < -self.widthL:
      return 0.0
    elif xx < 0.0:
      return self.getPressure() * (1 + xx / self.widthL)
    else:
      return self.getPressure() * (1 - xx / self.widthR)
  
  def plot(self):
    x  = np.ones(3) * self.getXLoc() + np.array([-self.widthL, 0.0, self.widthR])
    p = np.array([0, self.p, 0])
    
    col = 'r'
    plt.plot(x, p, col + '-')
    plt.plot([x[1], x[1]], [0.0, p[1]], col + '--')


class AftSemiInfinitePressureBand(PressureElement):
  
  def __init__(self, **kwargs):
    PressureElement.__init__(self, **kwargs)
    
  def influence(self, xx):
    Lambda0 = config.k0 * xx
    fg = getFG(Lambda0)
    K = getFunc2(Lambda0, fg)
    return K
  
  def pressureFun(self, xx):
    if xx > 0.0:
      return 0.0
    else:
      return self.getPressure()
  
  def plot(self):
    infinity = 50.0
    
    x  = np.ones(2) * self.getXLoc()
    x += np.array([-infinity, 0])
    
    p  = np.zeros(len(x))
    p += np.array([self.p, self.p])
    
    col = 'r'
    plt.plot(x, p, col + '-')
    plt.plot([x[1], x[1]], [p[0], 0.0], col + '--')
 
 
class ForwardSemiInfinitePressureBand(PressureElement):
  
  def __init__(self, **kwargs):
    PressureElement.__init__(self, **kwargs)
    
  def influence(self, xx):
    Lambda0 = config.k0 * xx
    fg = getFG(Lambda0)
    K = getFunc3(Lambda0, fg)
    return K
  
  def pressureFun(self, xx):
    if xx < 0.0:
      return 0.0
    else:
      return self.getPressure()
  
  def plot(self):
    infinity = 50.0
    x  = np.ones(2) * self.getXLoc()
    x += np.array([0, infinity])
    
    p  = np.zeros(len(x))
    p += np.array([self.p, self.p])
    
    col = 'r'
    plt.plot(x, p, col + '-')
    plt.plot([x[0], x[0]], [p[0], 0.0], col + '--')
 

class CompoundPressureElement(PressureElement):

  def __init__(self, **kwargs):
    PressureElement.__init__(self, **kwargs)
    self.element = [self]
    
  def setWidth(self, width):
    PressureElement.setWidth(self, np.sum(width))
    for i, el in enumerate(self.element):
      el.setWidth(width[i])
  
  def setXLoc(self, xLoc):
    PressureElement.setXLoc(self, xLoc)
    for el in self.element:
      el.setXLoc(xLoc)
  
  def setPressure(self, p):
    PressureElement.setPressure(self, p)
    for el in self.element:
      el.setPressure(p)
  
  def getInfluence(self, x):
    return np.sum([el.getInfluenceCoefficient(x) * el.getPressure() for el in self.element])
    
  def getInfluenceCoefficient(self, x):
    return self.getInfluence(x) / self.getPressure()
    
  def plot(self):
    for el in self.element:
      el.plot()

 
class CompleteTriangularPressureElementNew(CompoundPressureElement):
  
  def __init__(self, **kwargs):
    CompoundPressureElement.__init__(self, **kwargs)
    self.onBody = kwargs.get('onBody', True)
    self.element = [CompleteTriangularPressureElement()]
 
     
class AftFinitePressureBand(CompoundPressureElement):
 
  def __init__(self, **kwargs):
    CompoundPressureElement.__init__(self, **kwargs)
    self.element = [AftSemiInfinitePressureBand() for _ in range(2)]
    self.setWidth(self.width)
    self.setXLoc(self.xLoc)
    self.setPressure(self.p)
        
  def setWidth(self, width):
    PressureElement.setWidth(self, width)
    self.element[1].setXLoc(self.getXLoc() - self.getWidth())
  
  def setXLoc(self, xLoc):
    PressureElement.setXLoc(self, xLoc)
    self.element[0].setXLoc(xLoc)
    self.element[1].setXLoc(xLoc - self.getWidth())
     
  def setPressure(self, p):
    PressureElement.setPressure(self, p)
    self.element[0].setPressure( p)
    self.element[1].setPressure(-p)

  def pressureFun(self, xx):
    if xx > 0.0 or xx < -self.getWidth():
      return 0.0
    else:
      return self.getPressure()
  
  def plot(self):      
    x = np.array([self.element[0].getXLoc(), self.element[1].getXLoc()])
    p = np.array([0.0, self.p])
    
    col = 'm'
    plt.plot(x, np.ones(2)*p[1], col + '-')
    plt.plot(np.ones(2)*x[0], p, col + '--')
    plt.plot(np.ones(2)*x[1], p, col + '--')

    
class AftFinitePressureBandNew(CompoundPressureElement):
 
  def __init__(self, **kwargs):
    CompoundPressureElement.__init__(self, **kwargs)
    self.element = [AftHalfTriangularPressureElement(), \
                    ForwardHalfTriangularPressureElement()]
    self.setWidth(self.width)
    self.setXLoc(self.xLoc)
    self.setPressure(self.p)
        
  def setWidth(self, width):
    PressureElement.setWidth(self, width)
    self.element[1].setXLoc(self.getXLoc() - self.getWidth())
    for el in self.element:
      el.setWidth(width)
  
  def setXLoc(self, xLoc):
    PressureElement.setXLoc(self, xLoc)
    self.element[0].setXLoc(xLoc)
    self.element[1].setXLoc(xLoc - self.getWidth())
     
  def setPressure(self, p):
    PressureElement.setPressure(self, p)
    self.element[0].setPressure(p)
    self.element[1].setPressure(p)

  def plot(self):      
    x = np.array([self.element[0].getXLoc(), self.element[1].getXLoc()])
    p = np.array([0.0, self.p])
    
    col = 'm'
    plt.plot(x, np.ones(2)*p[1], col + '-')
    plt.plot(np.ones(2)*x[0], p, col + '--')
    plt.plot(np.ones(2)*x[1], p, col + '--')

    
class AftFinitePressureBandWithFwdHalfTriangle(CompoundPressureElement):
 
  def __init__(self, **kwargs):
    CompoundPressureElement.__init__(self, **kwargs)
    self.element = [AftFinitePressureBand(), ForwardHalfTriangularPressureElement()]
    

class AftSemiInfinitePressureBandWithFwdHalfTriangle(CompoundPressureElement):
 
  def __init__(self, **kwargs):
    CompoundPressureElement.__init__(self, **kwargs)
    self.element = [AftSemiInfinitePressureBand(), ForwardHalfTriangularPressureElement()]

  def setWidth(self, width):
    PressureElement.setWidth(self, width)
    self.element[1].setWidth(width)

  def setXLoc(self, xLoc):
    PressureElement.setXLoc(self,xLoc)
    self.element[0].setXLoc(xLoc)
    self.element[1].setXLoc(xLoc)
