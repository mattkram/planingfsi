"""Classes representing a pressure patch on the free surface.

Consists of the abstract base class `PressurePatch` and two derived classes,
representing a `PressureCushion`, which consists of all source terms, and
`PlaningSurface`, where the pressure profile is solved-for to meet the body
 boundary condition and trailing edge condition.
"""
import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fmin

import planingfsi.config as config
import planingfsi.krampy as kp

# from planingfsi.potentialflow.solver import PotentialPlaningCalculation
import planingfsi.potentialflow.pressureelement as pe


class PressurePatch(object):
    """Abstract base class representing a patch of pressure elements on the
    free surface.

    Args
    ----
    neighbor_up
        PressurePatch instance upstream of this one.


    neighbor_up
        PressurePatch instance downstream of this one.
    """

    def __init__(self, neighbor_up=None, neighbor_down=None):
        self.pressure_elements = []
        self.endPt = np.zeros(2)
        self.kuttaUnknown = False

        self.D = 0.0
        self.Dw = 0.0
        self.Dp = 0.0
        self.Df = 0.0
        self.L = 0.0
        self.Lp = 0.0
        self.Lf = 0.0
        self.M = 0.0

        self.neighborDown = neighbor_down
        self.neighborUp = neighbor_up

    def setNeighbor(self, up=None, down=None):
        """Assign an upstream or downstream neighbor to this patch, and assign
        this patch as the appropriate neighbor of the other.

        Args
        ----
        down : , optional
            Downstream neighbor.

        up : , optional
            Upstream neighbor.
        """
        # TODO: Clean up this logic, it is terrible. Use this method in
        #       constructor instead of assigning explicitly.
        if down is not None:
            self.neighborDown = down
        if self.neighborDown is not None:
            self.neighborDown.neighborUp = self

        if up is not None:
            self.neighborUp = up
        if self.neighborUp is not None:
            self.neighborUp.neighborDown = self

    def setEndPts(self, endPt):
        """Set the location of the end points.

        Args
        -----
        endPt : numpy.ndarray
            Array of end point locations. [trailing, leading] edge
        """
        self.endPt = endPt

    def getEndPts(self):
        """Get array of end point locations."""
        return self.endPt

    def getBasePt(self):
        """Get location of trailing edge, or left-most point.

        Returns
        -------
        float : x-location of base point.
        """
        return self.endPt[0]

    def get_length(self):
        """Return length of `PressurePatch` instance.

        Returns
        -------
        float : Length of pressure patch.
        """
        return self.endPt[1] - self.endPt[0]

    def setKuttaUnknown(self, unknown):
        """Set trailing eldge to be unknown.

        Args
        ----
        unknown : bool
            True of Kutta condition must be satisfied.
        """
        self.kuttaUnknown = unknown

    def isKuttaUnknown(self):
        """Return True if trailing edge pressure is unknown.

        Returns
        -------
        bool
        """
        return self.kuttaUnknown

    def resetElements(self):
        """Reset position of elements given the length and end points of the
        `PressurePatch` instance.

        Returns
        -------
        None, -1 if element type is invalid
        """
        x = self.getPts()
        for i, el in enumerate(self.pressure_elements):
            el.x_coord = x[i]
            if not el.is_source:
                el.z_coord = self.interpolator.getBodyHeight(x[i])

            if isinstance(el, pe.CompleteTriangularPressureElement):
                el.width = [x[i] - x[i - 1], x[i + 1] - x[i]]
            elif isinstance(el, pe.ForwardHalfTriangularPressureElement):
                el.width = x[i + 1] - x[i]
            elif isinstance(el, pe.AftHalfTriangularPressureElement):
                el.width = x[i] - x[i - 1]
            elif isinstance(el, pe.AftSemiInfinitePressureBand):
                el.width = np.inf
            else:
                print 'Invalid Element Type!'
                return -1
        return None

    def get_free_surface_height(self, x):
        """Get free surface height at a position x due to the elements on this
        patch.

        Args
        ----
        x : float
            x-position at which to calculate free-surface height.

        Returns
        -------
        float : Free-surface height.
        """
        return sum([el.get_influence(x) for el in self.pressure_elements])

    def calculateWaveDrag(self):
        """Calculate wave drag of patch.

        Returns
        -------
        None
        """
        f = self.get_free_surface_height
        xo = -10.1 * config.lam
        xTrough = fmin(f, xo, disp=False)[0]
        xCrest = fmin(lambda x: -f(x), xo, disp=False)[0]
        self.Dw = 0.0625 * config.rho * config.g * (f(xCrest) - f(xTrough))**2

    def printForces(self):
        """Print forces to screen."""
        print 'Forces and Moment for {0}:'.format(self.patchName)
        print '    Total Drag [N]      : {0:6.4e}'.format(self.D)
        print '    Wave Drag [N]       : {0:6.4e}'.format(self.Dw)
        print '    Pressure Drag [N]   : {0:6.4e}'.format(self.Dp)
        print '    Frictional Drag [N] : {0:6.4e}'.format(self.Df)
        print '    Total Lift [N]      : {0:6.4e}'.format(self.L)
        print '    Pressure Lift [N]   : {0:6.4e}'.format(self.Lp)
        print '    Frictional Lift [N] : {0:6.4e}'.format(self.Lf)
        print '    Moment [N-m]        : {0:6.4e}'.format(self.M)

    def write_forces(self):
        """Write forces to file."""
        kp.writeasdict(os.path.join(config.it_dir, 'forces_{0}.{1}'.format(
            self.patchName,
            config.data_format)),
            ['Drag', self.D],
            ['WaveDrag', self.Dw],
            ['PressDrag', self.Dp],
            ['FricDrag', self.Df],
            ['Lift', self.L],
            ['PressLift', self.Lp],
            ['FricLift', self.Lf],
            ['Moment', self.M],
            ['BasePt', self.getBasePt()],
            ['Length', self.get_length()])

    def load_forces(self):
        """Load forces from file."""
        K = kp.Dictionary(os.path.join(
            config.it_dir, 'forces_{0}.{1}'.format(self.patchName,
                                                   config.data_format)))
        self.D = K.read('Drag', 0.0)
        self.Dw = K.read('WaveDrag', 0.0)
        self.Dp = K.read('PressDrag', 0.0)
        self.Df = K.read('FricDrag', 0.0)
        self.L = K.read('Lift', 0.0)
        self.Lp = K.read('PressLift', 0.0)
        self.Lf = K.read('FricLift', 0.0)
        self.M = K.read('Moment', 0.0)
        self.setBasePt(K.read('BasePt', 0.0))
        self.set_length(K.read('Length', 0.0))


class PressureCushion(PressurePatch):
    count = 0

    def __init__(self, dict_, **kwargs):
        PressurePatch.__init__(self, **kwargs)
        self.index = PressureCushion.count
        PressureCushion.count += 1

        self.Dict = dict_

        self.patchName = self.Dict.read(
            'pressureCushionName', 'pressureCushion{0}'.format(self.index))
        self.cushionType = self.Dict.read('cushionType', '')
        self.cushionPressure = kwargs.get(
            'Pc', self.Dict.readLoadOrDefault('cushionPressure', 0.0))

        upstreamSurf = PlaningSurface.findByName(
            self.Dict.read('upstreamPlaningSurface', ''))
        downstreamSurf = PlaningSurface.findByName(
            self.Dict.read('downstreamPlaningSurface', ''))

        self.setNeighbor(down=downstreamSurf, up=upstreamSurf)

        if self.neighborDown is not None:
            self.neighborDown.setUpstreamPressure(self.cushionPressure)
        if self.neighborUp is not None:
            self.neighborUp.setKuttaPressure(self.cushionPressure)

        if self.cushionType == 'infinite':
            # Dummy element, will have 0 pressure
            self.pressure_elements += [
                pe.AftSemiInfinitePressureBand(is_source=True, is_on_body=False)]
            self.pressure_elements += [
                pe.AftSemiInfinitePressureBand(is_source=True, is_on_body=False)]
            self.downstreamLoc = -1000.0  # doesn't matter where

        else:
            self.pressure_elements += [
                pe.ForwardHalfTriangularPressureElement(is_source=True, is_on_body=False)]

            Nfl = self.Dict.read('numElements', 10)
            self.sf = self.Dict.read('smoothingFactor', np.nan)
            for n in [self.neighborDown, self.neighborUp]:
                if n is None and ~np.isnan(self.sf):
                    self.pressure_elements += [
                        pe.CompleteTriangularPressureElement(is_source=True) for _ in range(Nfl)]

            self.upstreamLoc = self.Dict.readLoadOrDefault(
                'upstreamLoc',   0.0)
            self.downstreamLoc = self.Dict.readLoadOrDefault(
                'downstreamLoc', 0.0)

            self.pressure_elements += [
                pe.AftHalfTriangularPressureElement(is_source=True, is_on_body=False)]

        self.update_end_pts()

    def setBasePt(self, x):
        self.endPt[1] = x

    def set_pressure(self, p):
        self.cushionPressure = p

    def get_pressure(self):
        return self.cushionPressure

    def getPts(self):
        if self.cushionType == 'smoothed':
            if np.isnan(self.sf):
                return np.array([self.downstreamLoc, self.upstreamLoc])
            else:
                N = len(self.pressure_elements) + 2
                addWidth = np.arctanh(0.99) * self.get_length() / (2 * self.sf)
                addL = np.linspace(-addWidth, addWidth, N / 2)
                x = np.hstack(
                    (self.downstreamLoc + addL, self.upstreamLoc + addL))
                return x
        else:
            return self.getEndPts()

    def getBasePt(self):
        if self.neighborUp is None:
            return PressurePatch.getBasePt(self)
        else:
            return self.neighborUp.getPts()[0]

    def update_end_pts(self):
        if self.neighborDown is not None:
            self.downstreamLoc = self.neighborDown.getPts()[-1]
        if self.neighborUp is not None:
            self.upstreamLoc = self.neighborUp.getPts()[0]

        self.setEndPts([self.downstreamLoc, self.upstreamLoc])

        self.resetElements()

        for elNum, el in enumerate(self.pressure_elements):
            if self.cushionType == 'smoothed' and ~np.isnan(self.sf):
                alf = 2 * self.sf / self.get_length()
                el.set_source_pressure(0.5 * self.get_pressure() * (
                    np.tanh(alf * el.get_xloc()) -
                    np.tanh(alf * (el.get_xloc() - self.get_length()))))
            # for infinite pressure cushion, first element is dummy, set to
            # zero, second is semiInfinitePressureBand and set to cushion
            # pressure
            elif self.cushionType == 'infinite':
                if elNum == 0:
                    el.set_source_pressure(0)
                else:
                    el.set_source_pressure(self.get_pressure())
            else:
                el.set_source_pressure(self.get_pressure())

    def set_length(self, length):
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

    def __init__(self, dict_, **kwargs):
        PressurePatch.__init__(self)
        self.index = PlaningSurface.count
        PlaningSurface.count += 1
        PlaningSurface.obj.append(self)

        self.Dict = dict_
        self.patchName = self.Dict.read('substructureName', '')
        Nfl = self.Dict.read('Nfl', 0)
        self.pointSpacing = self.Dict.read('pointSpacing', 'linear')
        self.initial_length = self.Dict.read('initialLength', None)
        self.minimumLength = self.Dict.read('minimumLength', 0.0)
        self.maximum_length = self.Dict.read(
            'maximum_length', float('Inf'))
        self.springConstant = self.Dict.read('springConstant', 1e4)
        self.kuttaPressure = kwargs.get(
            'kuttaPressure', self.Dict.readLoadOrDefault('kuttaPressure', 0.0))
        self.upstreamPressure = kwargs.get(
            'upstreamPressure', self.Dict.readLoadOrDefault('upstreamPressure', 0.0))
        self.upstreamPressureCushion = None

        self.isInitialized = False
        self.active = True
        self.kuttaUnknown = True
        self.sprung = self.Dict.read('sprung', False)
        if self.sprung:
            self.initial_length = 0.0
            self.minimumLength = 0.0
            self.maximum_length = 0.0

        self.get_residual = self.getPressureResidual
        self.pressure_elements += [
            pe.ForwardHalfTriangularPressureElement(is_on_body=False)]

        self.pressure_elements += [pe.ForwardHalfTriangularPressureElement(
            is_source=True, pressure=self.kuttaPressure)]
        self.pressure_elements += [pe.CompleteTriangularPressureElement()
                                   for _ in range(Nfl - 1)]
        self.pressure_elements += [pe.AftHalfTriangularPressureElement(
            is_source=True, pressure=self.upstreamPressure)]

        for el in self.pressure_elements:
            el.parent = self

        # Define point spacing
        if self.pointSpacing == 'cosine':
            self.pct = 0.5 * (1 - kp.cosd(np.linspace(0, 180, Nfl + 1)))
        else:
            self.pct = np.linspace(0.0, 1.0, Nfl + 1)
        self.pct /= self.pct[-2]
        self.pct = np.hstack((0.0, self.pct))

        self.xBar = 0.0

    def set_interpolator(self, interpolator):
        self.interpolator = interpolator

    def get_min_length(self):
        return self.minimumLength

    def initialize_end_pts(self):
        self.setBasePt(self.interpolator.getSeparationPoint()[0])

        if not self.isInitialized:
            if self.initial_length is None:
                self.set_length(
                    self.interpolator.getImmersedLength() - self.getBasePt())
            else:
                self.set_length(self.initial_length)
            self.isInitialized = True

    def resetElements(self):
        PressurePatch.resetElements(self)
        x = self.getPts()
        self.pressure_elements[0].width = x[2] - x[0]

    def setKuttaPressure(self, p):
        self.kuttaPressure = p

    def setUpstreamPressure(self, p):
        self.upstreamPressure = p
        el = self.pressure_elements[-1]
        el.pressure = p
        el.is_source = True
        el.z_coord = np.nan

    def getKuttaPressure(self):
        return self.kuttaPressure

    def set_length(self, Lw):
        Lw = np.max([Lw, 0.0])

        x0 = self.interpolator.getSeparationPoint()[0]
        self.setEndPts([x0, x0 + Lw])

    def setBasePt(self, x):
        self.endPt[0] = x

    def setEndPts(self, endPt):
        self.endPt = endPt
        self.resetElements()

    def getPts(self):
        return self.pct * self.get_length() + self.endPt[0]

    def getPressureResidual(self):
        if self.get_length() <= 0.0:
            return 0.0
        else:
            return self.pressure_elements[0].pressure / config.pStag

    def getTransomHeightResidual(self):
        if self.get_length() <= 0.0:
            return 0.0
        else:
            el = self.pressure_elements[0]
            return ((el.z_coord - el.pressure) *
                    config.rho * config.g / config.pStag)

    def getUpstreamPressure(self):
        if self.neighborUp is not None:
            return self.neighborUp.get_pressure()
        else:
            return 0.0

    def getDownstreamPressure(self):
        if self.neighborDown is not None:
            return self.neighborDown.get_pressure()
        else:
            return 0.0

    def calculateForces(self):
        if self.get_length() > 0.0:
            el = [el for el in self.pressure_elements if el.is_on_body]
            self.x = np.array([eli.x_coord for eli in el])
            self.p = np.array([eli.pressure for eli in el])
            self.p += self.pressure_elements[0].pressure
            self.shear_stress = np.array(
                [eli.shear_stress for eli in el])
            self.s = np.array(
                [self.interpolator.getSFixedX(xx) for xx in self.x])

            self.fP = interp1d(
                self.x, self.p,   bounds_error=False, fill_value=0.0)
            self.fTau = interp1d(
                self.x, self.shear_stress, bounds_error=False, fill_value=0.0)

            AOA = kp.atand(self.getBodyDerivative(self.x))

            self.Dp = kp.integrate(self.s, self.p * kp.sind(AOA))
            self.Df = kp.integrate(self.s, self.shear_stress * kp.cosd(AOA))
            self.Lp = kp.integrate(self.s, self.p * kp.cosd(AOA))
            self.Lf = -kp.integrate(self.s, self.shear_stress * kp.sind(AOA))
            self.D = self.Dp + self.Df
            self.L = self.Lp + self.Lf
            self.M = kp.integrate(
                self.x, self.p * kp.cosd(AOA) * (self.x - config.xCofR))
        else:
            self.Dp = 0.0
            self.Df = 0.0
            self.Lp = 0.0
            self.Lf = 0.0
            self.D = 0.0
            self.L = 0.0
            self.M = 0.0
            self.x = []
        if self.sprung:
            self.applySpring()

        self.calculateWaveDrag()

    def get_loads_in_range(self, x0, x1):
        # Get indices within range unless length is zero
        if self.get_length() > 0.0:
            ind = np.nonzero((self.x > x0) * (self.x < x1))[0]
        else:
            ind = []

        # Output all values within range
        if not ind == []:
            x = np.hstack((x0,    self.x[ind],           x1))
            p = np.hstack((self.fP(x0),   self.p[ind],   self.fP(x1)))
            tau = np.hstack(
                (self.fTau(x0), self.shear_stress[ind], self.fTau(x1)))
        else:
            x = np.array([x0, x1])
            p = np.zeros_like(x)
            tau = np.zeros_like(x)
        return x, p, tau

#     def applySpring(self):
#         xs = self.pressure_elements[0].get_xloc()
#         zs = self.pressure_elements[0].get_zloc()
#         disp = zs - PotentialPlaningCalculation.getTotalFreeSurfaceHeight(xs)
#         Fs = -self.springConstant * disp
#         self.L += Fs
#         self.M += Fs * (xs - config.xCofR)

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

        for si, el in zip(s, self.pressure_elements):
            el.set_shear_stress(shearStressFunc(si))

    def getBodyDerivative(self, x, direction='r'):
        if isinstance(x, float):
            x = [x]
        return np.array(map(lambda xx: kp.getDerivative(self.interpolator.getBodyHeight, xx, direction), x))
