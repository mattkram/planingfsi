import os

import numpy as np
import matplotlib.pyplot as plt

import planingfsi.config as config
import planingfsi.krampy as kp

class Mesh:
  
    @classmethod
    def getPtByID(cls, ID):
        return Point.findByID(ID)

    def __init__(self):
        self.submesh = []
        self.addPoint(0, 'dir', [0,0])

    def addSubmesh(self, name=''):
        submesh = Submesh(name)
        self.submesh.append(submesh)
        return submesh
        
    def addPoint(self, ID, method, position, **kwargs):
        P = Point()
        P.setID(ID)

        if   method == 'dir':
            P.setPos(np.array(position))
        elif method == 'rel':
            basePtID, ang, R = position
            P.setPos(Point.findByID(basePtID).getPos() + R * kp.ang2vecd(ang))
        elif method == 'con':
            basePtID, dim, val = position
            ang = kwargs.get('angle', 0 if dim == 'x' else 90)

            basePt = Point.findByID(basePtID).getPos()
            if   dim == 'x':
                P.setPos(np.array([val, basePt[1] + (val-basePt[0])*kp.tand(ang)]))
            elif dim == 'y':
                P.setPos(np.array([basePt[0] + (val-basePt[1])/kp.tand(ang), val]))
            else:
                print('Incorrect dimension specification')
        elif method == 'pct':
            basePtID, endPtID, pct = position
            basePt = Point.findByID(basePtID).getPos()
            endPt  = Point.findByID(endPtID).getPos()
            P.setPos((1 - pct) * basePt + pct * endPt)
        else:
            raise NameError('Incorrect position specification method for point, ID: {0}'.format(ID))

        return P

    def addPointAlongCurve(self, ID, curve, pct):
        P = Point()
        P.setID(ID)
        P.setPos(map(curve.getShapeFunc(), [pct])[0])
        return P

    def addLoad(self, ptID, F):
        Point.findByID(ptID).addFixedLoad(F)

    def fixAllPoints(self):
        for pt in Point.All():
            pt.setFixedDOF('x','y')

    def fixPoints(self, ptIDList):
        for pt in [Point.findByID(ptID) for ptID in ptIDList]:
            pt.setFixedDOF('x','y')

    def rotatePoints(self, basePtID, angle, ptIDList):
        for pt in [Point.findByID(ptID) for ptID in ptIDList]:
            pt.rotate(basePtID, angle)

    def rotateAllPoints(self, basePtID, angle):
        for pt in Point.All():
            pt.rotate(basePtID, angle)

    def movePoints(self, dx, dy, ptIDList):
        for pt in [Point.findByID(ptID) for ptID in ptIDList]:
            pt.move(dx, dy)
    
    def moveAllPoints(self, dx, dy):
        for pt in Point.All():
            pt.move(dx, dy)
    
    def scaleAllPoints(self, sf, basePtID=0):
        basePt = Point.findByID(basePtID).getPos()
        for pt in Point.All():
            pt.setPos((pt.getPos() - basePt) * sf + basePt)

    def getDiff(self, pt0, pt1):
        return Point.findByID(pt1).getPos() - Point.findByID(pt0).getPos()

    def get_length(self, pt0, pt1):
        return np.linalg.norm(self.getDiff(pt0, pt1))

    def display(self, **kwargs):
        if kwargs.get('disp', False):
            Shape.printAll()
            print(('Line count:  {0}'.format(Line.count())))
            print(('Point count: {0}'.format(Point.count())))

    def plot(self, **kwargs):
        show = kwargs.get('show',False)
        save = kwargs.get('save',False)
        plot = show or save
        if plot:
            plt.figure(figsize=(16,14))
            plt.axis('equal')
            plt.xlabel(r'$x$', size=18)
            plt.ylabel(r'$y$', size=18)
            
            Shape.plotAll()
            
            lims = plt.gca().get_xlim()
            ext = (lims[1] - lims[0]) * 0.1
            plt.xlim([lims[0]-ext, lims[1]+ext])
        
            # Process optional arguments and save or show figure
            if save:
                savedFileName = kwargs.get('fileName','meshLayout')
                plt.savefig(savedFileName + '.eps', format='eps')
            if show:
                plt.show()

    def write(self):
        kp.createIfNotExist(config.path.mesh_dir)
        x, y = list(zip(*[pt.getPos() for pt in Point.All()]))
        kp.writeaslist(os.path.join(config.path.mesh_dir, 'nodes.txt'), ['x', x], ['y', y])

        x, y = list(zip(*[pt.getFixedDOF() for pt in Point.All()]))
        kp.writeaslist(os.path.join(config.path.mesh_dir, 'fixedDOF.txt'), ['x', x], ['y', y], headerFormat='>1', dataFormat='>1')

        x, y = list(zip(*[pt.getFixedLoad() for pt in Point.All()]))
        kp.writeaslist(os.path.join(config.path.mesh_dir, 'fixedLoad.txt'), ['x', x], ['y', y], headerFormat='>6', dataFormat='6.4e')

        for sm in self.submesh:
            sm.write()


class Submesh(Mesh):
  
    def __init__(self, name):
        Mesh.__init__(self)
        self.name = name
        self.line = []
      
    def addCurve(self, ptID1, ptID2, **kwargs):
        arcLen = kwargs.get('arcLen', None)
        radius = kwargs.get('radius', None)

        C = Curve(kwargs.get('Nel', 1))
        C.setID(kwargs.get('ID', -1))
        C.setEndPtsByID(ptID1, ptID2)
        
        if arcLen is not None:
            C.setArcLen(arcLen)
        elif radius is not None:
            C.setRadius(radius)
        else:
            C.setArcLen(0.0)

        C.distributePoints()
        for pt in C.pt:
            pt.setUsed()

        self.line += [l for l in C.getLines()]

        return C

    def write(self):
        if len(self.line) > 0:
            ptL, ptR = list(zip(*[[pt.getIndex() for pt in l._get_element_coords()] for l in self.line]))
            kp.writeaslist(os.path.join(config.path.mesh_dir, 'elements_{0}.txt'.format(self.name)), ['ptL', ptL], ['ptR', ptR], headerFormat='<4', dataFormat='>4')


class Shape:
    obj = []

    @classmethod
    def All(cls):
        return [o for o in cls.obj]
    
    @classmethod
    def count(cls):
        return len(cls.All())
    
    @classmethod
    def plotAll(cls):
        for o in cls.obj:
            o.plot()
         
    @classmethod
    def printAll(cls):
        for o in cls.obj:
            o.display()
               
    @classmethod
    def findByID(cls, ID):
        if ID == -1:
            return None
        else:
            obj = [a for a in cls.All() if a.ID == ID]
            if len(obj) == 0:
                return None
            else:
                return obj[0]

    def __init__(self):
        self.setIndex(Shape.count())
        self.ID = -1
        Shape.obj.append(self)
       
    def setID(self, ID):
        existing = self.__class__.findByID(ID)
        if existing is not None:
            existing.setID(-1)
        self.ID = ID

    def setIndex(self, ind):
        self.ind = ind

    def getID(self):
        return self.ID

    def getIndex(self):
        return self.ind

    def plot(self):
        return 0


class Point(Shape):
    obj = []

    def __init__(self):
        Shape.__init__(self)
        self.setIndex(Point.count())
        Point.obj.append(self)
        
        self.pos = np.zeros(2)
        self.fixDOF = [True] * 2
        self.fixedLoad = np.zeros(2)
        self.used = False
    
    def setUsed(self, used=True):
        self.used = True
        self.setFreeDOF('x', 'y')
   
    def getFixedLoad(self):
        return self.fixedLoad

    def addFixedLoad(self, F):
        for i in range(2):
            self.fixedLoad[i] += F[i]

    def setFreeDOF(self, *args):
        for arg in args:
            if arg == 'x':
                self.fixDOF[0] = False
            if arg == 'y':
                self.fixDOF[1] = False
    
    def setFixedDOF(self, *args):
        for arg in args:
            if arg == 'x':
                self.fixDOF[0] = True
            if arg == 'y':
                self.fixDOF[1] = True
    
    def getFixedDOF(self):
        return self.fixDOF

    def move(self, dx, dy):
        self.setPos(self.getPos() + np.array([dx, dy]))

    def setPos(self, pos):
        self.pos = pos

    def getPos(self):
        return self.pos

    def getXPos(self):
        return self.pos[0]

    def getYPos(self):
        return self.pos[1]

    def rotate(self, basePtID, angle):
        basePt = Point.findByID(basePtID).getPos()
        self.setPos(kp.rotateVec(self.getPos() - basePt, angle) + basePt)

    def display(self):
        print(('{0} {1}: ID = {2}, Pos = {3}'.format(self.__class__.__name__, self.getIndex(), self.getID(), self.getPos())))

    def plot(self):
        if self.ID == -1:
            plt.plot(self.pos[0], self.pos[1], 'r*')
        else:
            plt.plot(self.pos[0], self.pos[1], 'ro')
            plt.text(self.pos[0], self.pos[1], ' {0}'.format(self.ID))


class Curve(Shape):
    obj = []

    def __init__(self, Nel=1):
        Shape.__init__(self)
        self.setIndex(Curve.count())
        Curve.obj.append(self)
        self.pt    = []
        self.line  = []
        self._end_pts = []
        self.Nel   = Nel
        self.plotSty = 'm--'

    def setEndPtsByID(self, ptID1, ptID2):
        self.setEndPts([Point.findByID(pid) for pid in [ptID1, ptID2]])

    def getShapeFunc(self):
        xy = [pt.getPos() for pt in self._end_pts]
        if self.radius == 0.0:
            return lambda s: xy[0] * (1 - s) + xy[1] * s
        else:
            x, y = list(zip(*xy))
            gam = np.arctan2(y[1] - y[0], x[1] - x[0])
            alf = self.arcLen / (2 * self.radius)
            return lambda s: self._end_pts[0].getPos() + 2*self.radius*np.sin(s*alf) * kp.ang2vec(gam + (s-1)*alf)
    
    def setRadius(self, R):
        self.radius = R
        self.calculateChord()
        self.calculateArcLength()
       
    def setArcLen(self, arcLen):
        self.calculateChord()
        if self.chord >= arcLen:
            self.arcLen = 0.0
            self.setRadius(0.0)
        else:
            self.arcLen = arcLen
            self.calculateRadius()

    def calculateRadius(self):
        self.radius = 1 / self.calculateCurvature()

    def calculateChord(self):
        x, y = list(zip(*[pt.getPos() for pt in self._end_pts]))
        self.chord = ((x[1] - x[0])**2 + (y[1] - y[0])**2)**0.5

    def calculateArcLength(self):
        if self.radius == 0:
            self.arcLen = self.chord
        else:
            f = lambda s: self.chord / (2 * self.radius) - np.sin(s / (2 * self.radius))
        
            # Keep increasing guess until fsolve finds the first non-zero root
            self.arcLen = kp.fzero(f, self.chord+1e-6)

    def calculateCurvature(self):
        f = lambda x: x * self.chord / 2 - np.sin(x * self.arcLen / 2)

        # Keep increasing guess until fsolve finds the first non-zero root
        kap  = 0.0
        kap0 = 0.0
        while kap <= 1e-6:
            kap = kp.fzero(f, kap0)
            kap0 += 0.02

        return kap

    def distributePoints(self):
        self.pt.append(self._end_pts[0])
        if not self.Nel == 1:
            # Distribute N points along a parametric curve defined by f(s), s in [0,1]
            s = np.linspace(0.0, 1.0, self.Nel + 1)[1:-1]
            for xy in map(self.getShapeFunc(), s):
                P = Point()
                self.pt.append(P)
                P.setPos(xy)
        self.pt.append(self._end_pts[1])
        self.generateLines()

    def generateLines(self):
        for ptSt, ptEnd in zip(self.pt[:-1], self.pt[1:]):
            L = Line()
            L.setEndPts([ptSt, ptEnd])
            self.line.append(L)
    
    def getLines(self):
        return self.line
       
    def setPts(self, pt):
        self.pt = pt
    
    def setEndPts(self, endPt):
        self._end_pts = endPt
    
    def _get_element_coords(self):
        return self.pt
      
    def getPtIDs(self):
        return [pt.getID() for pt in self.pt]
      
    def display(self):
        print(('{0} {1}: ID = {2}, Pt IDs = {3}'.format(self.__class__.__name__, self.getIndex(), self.getID(), self.getPtIDs())))

    def plot(self):
        x, y = list(zip(*[pt.getPos() for pt in self.pt]))
        plt.plot(x, y, self.plotSty)


class Line(Curve):
    obj = []

    def __init__(self):
        Curve.__init__(self)
        self.setIndex(Line.count())
        Line.obj.append(self)
        self.plotSty = 'b-'

    def setEndPts(self, endPt):
        Curve.setEndPts(self, endPt)
        self.setPts(endPt)
