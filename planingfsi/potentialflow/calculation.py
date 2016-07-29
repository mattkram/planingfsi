"""Fundamental module for constructing and solving planing potential flow
problems
"""
#pylint: disable=C0103

from os.path import join

import numpy as np
from scipy.optimize import fmin

import planingfsi.config as config
import planingfsi.krampy as kp

from planingfsi.potentialflow.pressurepatch import PlaningSurface
from planingfsi.potentialflow.pressurepatch import PressureCushion

if config.plot:
    import matplotlib.pyplot as plt


class PotentialPlaningCalculation(object):
    """The top-level object which handles calculation of the potential flow
    problem.

    Pointers to the planing surfaces and pressure elements are stored for
    reference during initialization and problem setup. The potential planing
    calculation is then solved during iteration with the structural solver.
    """

    # TODO: getTotalFreeSurfaceHeight should not be a class method. There should
    # only be one instance of PotentialPlaningCalculation
    obj = []

    @classmethod
    def getTotalFreeSurfaceHeight(cls, x):
        """Class method gets total free surface height by summing the response
        from all elements.
        """
        return cls.obj[0].getFreeSurfaceHeight(x)

    def __init__(self):
        self.X = None
        self.xFS = None
        self.D = None
        self.Dw = None
        self.Dp = None
        self.Df = None
        self.L = None
        self.Lp = None
        self.Lf = None
        self.M = None
        self.p = None
        self.tau = None
        self.xFS = None
        self.zFS = None
        self.x = None
        self.xBar = None
        self.xHist = None
        self.storeLen = None
        self.maxLen = None

        self.planingSurface = []
        self.pressureCushion = []
        self.pressurePatch = []
        self.pressureElement = []
        self.lineFS = None
        self.lineFSi = None
        PotentialPlaningCalculation.obj.append(self)
        self.solver = None

        self.minLen = None
        self.initLen = None

    def addPressurePatch(self, instance):
        """Add pressure patch to the calculation.

        Args
        ----
        instance : PressurePatch
            The pressure patch object to add.

        Returns
        -------
        None
        """
        self.pressurePatch.append(instance)
        self.pressureElement += [el for el in instance.pressureElement]
        return None

    def addPlaningSurface(self, dictName='', **kwargs):
        """Add planing surface to the calculation from a dictionary file name.

        Args
        ----
        dictName : str
            The path to the dictionary file.

        Returns
        -------
        PlaningSurface
            Instance created from dictionary.
        """
        dict_ = kp.ensureDict(dictName)
        instance = PlaningSurface(dict_, **kwargs)
        self.planingSurface.append(instance)
        self.addPressurePatch(instance)
        return instance

    def addPressureCushion(self, dictName='', **kwargs):
        """Add pressure cushion to the calculation from a dictionary file name.

        Args
        ----
        dictName : str
            The path to the dictionary file.

        Returns
        -------
        PressureCushion
            Instance created from dictionary.
        """
        Dict = kp.ensureDict(dictName)
        instance = PressureCushion(Dict, **kwargs)
        self.pressureCushion.append(instance)
        self.addPressurePatch(instance)
        return instance

    def getPlaningSurfaceByName(self, name):
        """Return planing surface by name.

        Args
        ----
        name : str
            Name of planing surface.

        Returns
        -------
        PlaningSurface
            Planing surface object match
        """
        l = [surf for surf in self.planingSurface if surf.patchName == name]
        if len(l) > 0:
            return l[0]
        else:
            return None

    def printElementStatus(self):
        """Print status of each element."""
        for el in self.pressureElement:
            el.printElement()

        return None

    def calculatePressure(self):
        """Calculate pressure of each element to satisfy body boundary
        conditions.
        """
        # Form lists of unknown and source elements
        unknownEl = [el for el in self.pressureElement if not el.isSource() and el.getWidth() > 0.0]
        nanEl = [el for el in self.pressureElement if el.getWidth() <= 0.0]
        sourceEl = [el for el in self.pressureElement if el.isSource()]

        # Form arrays to build linear system
        # x location of unknown elements
        X = np.array([el.getXLoc() for el in unknownEl])

        # influence coefficient matrix (to be reshaped)
        A = np.array([el.getInfluenceCoefficient(x) for x in X for el in unknownEl])

        # height of each unknown element above waterline
        Z = np.array([el.getZLoc() for el in unknownEl]) - config.hWL

        # subtract contribution from source elements
        Z -= np.array([np.sum([el.getInfluence(x) for el in sourceEl]) for x in X])

        # Solve linear system
        dim = len(unknownEl)
        if not dim == 0:
            p = np.linalg.solve(np.reshape(A, (dim, dim)), np.reshape(Z, (dim, 1)))[:, 0]
        else:
            p = np.zeros_like(Z)

        # Apply calculated pressure to the elements
        for pi, el in zip(p, unknownEl):
            el.setPressure(pi)

        for el in nanEl:
            el.setPressure(0.0)

        return None

    def getResidual(self, Lw):
        """Return array of residuals of each planing surface to satisfy Kutta
        condition.

        Args
        ----
        Lw : np.ndarray
            Array of wetted lengths of planing surfaces.

        Returns
        -------
        np.ndarray
            Array of trailing edge residuals of each planing surface.
        """
        # Set length of each planing surface
        for Lwi, p in zip(Lw, self.planingSurface):
            p.setLength(np.min([Lwi, p.maximumLength]))

        # Update bounds of pressure cushions
        for p in self.pressureCushion:
            p.updateEndPts()

        # Solve for unknown pressures and output residual
        self.calculatePressure()

        res = np.array([p.getResidual() for p in self.planingSurface])

        print '      Lw:        ', Lw
        print '      Func value:', res

        return res

    def calculateResponse(self):
        """Calculate response, including pressure and free-surface profiles.

        Will load results from file if specified.
        """
        if config.resultsFromFile:
            self.loadResponse()
        else:
            self.calculateResponseUnknownWettedLength()

    def calculateResponseUnknownWettedLength(self):
        """Calculate potential flow problem via iteration to find wetted length
        of all planing surfaces to satisfy all trailing edge conditions.
        """
        # Reset results so they will be recalculated after new solution
        self.xFS = None
        self.X = None
        self.xHist = []

        # Initialize planing surface lengths and then solve until residual is *zero*
        if len(self.planingSurface) > 0:
            for p in self.planingSurface:
                p.initializeEndPts()

            print '  Solving for wetted length:'

            if self.minLen is None:
                self.minLen = np.array([p.getMinLength() for p in self.planingSurface])
                self.maxLen = np.array([p.maximumLength for p in self.planingSurface])
                self.initLen = np.array([p.initialLength for p in self.planingSurface])
            else:
                for i, p in enumerate(self.planingSurface):
                    L = p.getLength()
                    self.initLen[i] = p.initialLength
                    if ~np.isnan(L) and L - self.minLen[i] > 1e-6:
                        # and self.solver.it < self.solver.maxIt:
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

            self.calculatePressureAndShearProfile()

    def plotResidualsOverRange(self):
        """Plot residuals."""
        self.storeLen = np.array([p.getLength() for p in self.planingSurface])

        xx, yy = zip(*self.xHist)

        N = 10
        L = np.linspace(0.001, 0.25, N)
        X = np.zeros((N, N))
        Y = np.zeros((N, N))
        Z1 = np.zeros((N, N))
        Z2 = np.zeros((N, N))

        for i, Li in enumerate(L):
            for j, Lj in enumerate(L):
                x = np.array([Li, Lj])
                y = self.getResidual(x)

                X[i, j] = x[0]
                Y[i, j] = x[1]
                Z1[i, j] = y[0]
                Z2[i, j] = y[1]

        for i, Zi, seal in zip([2, 3], [Z1, Z2], ['bow', 'stern']):
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

    def calculatePressureAndShearProfile(self):
        """Calculate pressure and shear stress profiles over plate surface."""
        if self.X is None:
            if config.shearCalc:
                for p in self.planingSurface:
                    p.calculateShearStress()

            # Calculate forces on each patch
            for p in self.pressurePatch:
                p.calculateForces()

            # Calculate pressure profile
            if len(self.pressurePatch) > 0:
                self.X = np.sort(np.unique(np.hstack([p.getPts() for p in self.pressurePatch])))
                self.p = np.zeros_like(self.X)
                self.tau = np.zeros_like(self.X)
            else:
                self.X = np.array([-1e6, 1e6])
                self.p = np.zeros_like(self.X)
                self.tau = np.zeros_like(self.X)
            for el in self.pressureElement:
                if el.isOnBody():
                    self.p[el.getXLoc() == self.X] += el.getPressure()
                    self.tau[el.getXLoc() == self.X] += el.getShearStress()

            # Calculate total forces as sum of forces on each patch
            for var in ['D', 'Dp', 'Df', 'Dw', 'L', 'Lp', 'Lf', 'M']:
                setattr(self, var, sum([getattr(p, var) for p in self.pressurePatch]))

            f = self.getFreeSurfaceHeight
            xo = -10.1 * config.lam
            xTrough, = fmin(f, xo, disp=False)
            xCrest, = fmin(lambda x: -f(x), xo, disp=False)
            self.Dw = 0.0625 * config.rho * config.g * (f(xCrest) - f(xTrough))**2

            # Calculate center of pressure
            self.xBar = kp.integrate(self.X, self.p * self.X) / self.L
            if config.plotPressure:
                self.plotPressure()

    def calculateFreeSurfaceProfile(self):
        """Calculate free surface profile."""
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
        """Return free surface height at a given x-position consideringthe
        contributions from all pressure patches.

        Args
        ----
        x : float
            x-position at which free surface height shall be returned.

        Returns
        -------
        float
            Free-surface position at input x-position
        """
        return sum([patch.getFreeSurfaceHeight(x) for patch in self.pressurePatch])

    def getFreeSurfaceDerivative(self, x, direction='c'):
        """Return slope (derivative) of free-surface profile.

        Args
        ----
        x : float
            x-position

        Returns
        -------
        float
            Derivative or slope of free-surface profile
        """
        ddx = kp.getDerivative(self.getFreeSurfaceHeight, x, direction='c')
        return ddx

    def writeResults(self):
        """Write resultsto file."""
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
        """Write pressure and shear stress profiles to data file."""
        if len(self.pressureElement) > 0:
            kp.writeaslist(
                join(config.itDir, 'pressureAndShear.{0}'.format(config.dataFormat)),
                ['x [m]', self.X],
                ['p [Pa]', self.p],
                ['tau [Pa]', self.tau])

    def writeFreeSurface(self):
        """Write free-surface profile to file."""
        if len(self.pressureElement) > 0:
            kp.writeaslist(join(config.itDir, 'freeSurface.{0}'.format(config.dataFormat)),
                           ['x [m]', self.xFS],
                           ['y [m]', self.zFS])

    def writeForces(self):
        """Write forces to file."""
        if len(self.pressureElement) > 0:
            kp.writeasdict(join(config.itDir, 'forces_total.{0}'.format(config.dataFormat)),
                           ['Drag', self.D],
                           ['WaveDrag', self.Dw],
                           ['PressDrag', self.Dp],
                           ['FricDrag', self.Df],
                           ['Lift', self.L],
                           ['PressLift', self.Lp],
                           ['FricLift', self.Lf],
                           ['Moment', self.M])

        for patch in self.pressurePatch:
            patch.writeForces()

    def loadResponse(self):
        """Load results from file."""
        self.loadForces()
        self.loadPressureAndShear()
        self.loadFreeSurface()

    def loadPressureAndShear(self):
        """Load pressure and shear stress from file."""
        self.x, self.p, self.tau = np.loadtxt(
            join(config.itDir, 'pressureAndShear.{0}'.format(config.dataFormat)),
            unpack=True)
        for el in [el for patch in self.planingSurface for el in patch.pressureElement]:
            compare = np.abs(self.x - el.getXLoc()) < 1e-6
            if any(compare):
                el.setPressure(self.p[compare][0])
                el.setShearStress(self.tau[compare][0])

        for p in self.planingSurface:
            p.calculateForces()

    def loadFreeSurface(self):
        """Load free surface coordinates from file."""
        try:
            Data = np.loadtxt(join(config.itDir, 'freeSurface.{0}'.format(config.dataFormat)))
            self.xFS = Data[:, 0]
            self.zFS = Data[:, 1]
        except IOError:
            self.zFS = np.zeros_like(self.xFS)
        return None

    def loadForces(self):
        """Load forces from file."""
        K = kp.Dictionary(join(config.itDir, 'forces_total.{0}'.format(config.dataFormat)))
        self.D = K.readOrDefault('Drag', 0.0)
        self.Dw = K.readOrDefault('WaveDrag', 0.0)
        self.Dp = K.readOrDefault('PressDrag', 0.0)
        self.Df = K.readOrDefault('FricDrag', 0.0)
        self.L = K.readOrDefault('Lift', 0.0)
        self.Lp = K.readOrDefault('PressLift', 0.0)
        self.Lf = K.readOrDefault('FricLift', 0.0)
        self.M = K.readOrDefault('Moment', 0.0)

        for patch in self.pressurePatch:
            patch.loadForces()

    def plotPressure(self):
        """Create a plot of the pressure and shear stress profiles."""
        plt.figure(figsize=(5.0, 5.0))
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

        plt.xlim([-1.0, 1.0])
    #    plt.xlim(kp.minMax(self.X / config.Lref * 2))
        plt.ylim([0.0, np.min([1.0, 1.2 * np.max(self.p / config.pStag)])])
        plt.savefig('pressureElements.{0}'.format(config.figFormat), format=config.figFormat)
        plt.figure(1)

        return None

    def plotFreeSurface(self):
        """Create a plot of the free surface profile."""
        self.calculateFreeSurfaceProfile()
        if self.lineFS is not None:
            self.lineFS.set_data(self.xFS, self.zFS)
        endPts = np.array([config.xFSMin, config.xFSMax])
        if self.lineFSi is not None:
            self.lineFSi.set_data(endPts, config.hWL * np.ones_like(endPts))
        return None
