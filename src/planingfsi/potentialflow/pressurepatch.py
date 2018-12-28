"""Classes representing a pressure patch on the free surface."""
import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fmin

import planingfsi.config as config

# import planingfsi.krampy as kp
import planingfsi.potentialflow.pressureelement as pe


class PressurePatch(object):
    """Abstract base class representing a patch of pressure elements on the
    free surface.

    Attributes
    ----------
    pressure_elements : list
        List of pressure elements.

    base_pt : float
        x-location of base point.

    length : float
        Length of pressure patch.

    is_kutta_unknown : bool
        True of trailing edge pressure unknown

    neighbor_up : PressurePatch
        PressurePatch instance upstream of this one.

    neighbor_down : PressurePatch
        PressurePatch instance downstream of this one.

    interpolator : FSIInterpolator
        Object to get interpolated body position if a `PlaningSurface`

    force_dict : dict
        Dictionary of global force values derived from element pressure/shear
        stress.

    parent : PotentialPlaningCalculation
        Parent solver this patch belongs to.
    """

    def __init__(self, neighbor_up=None, neighbor_down=None, parent=None):
        self.pressure_elements = []
        self._end_pts = np.zeros(2)
        self.is_kutta_unknown = False
        self._neighbor_up = None
        self._neighbor_down = None
        self.neighbor_up = neighbor_up
        self.neighbor_down = neighbor_down
        self.interpolator = None
        self.force_dict = {}
        self.parent = parent

    @property
    def base_pt(self):
        return self._end_pts[0]

    @base_pt.setter
    def base_pt(self, x):
        self._end_pts[0] = x

    @property
    def length(self):
        return self._end_pts[1] - self._end_pts[0]

    @property
    def neighbor_up(self):
        return self._neighbor_up

    @neighbor_up.setter
    def neighbor_up(self, obj):
        if obj is not None:
            self._neighbor_up = obj
            self._neighbor_up.neighbor_down = self

    @property
    def neighbor_down(self):
        return self._neighbor_down

    @neighbor_down.setter
    def neighbor_down(self, obj):
        if obj is not None:
            self._neighbor_down = obj
            self._neighbor_down.neighbor_up = self

    def _reset_element_coords(self):
        """Re-distribute pressure element positions given the length and end
        points of this patch.
        """
        x = self._get_element_coords()
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
                raise ValueError("Invalid Element Type!")

    def get_free_surface_height(self, x):
        """Get free surface height at a position x due to the elements on this
        patch.

        Args
        ----
        x : float
            x-position at which to calculate free-surface height.
        """
        return sum([el.get_influence(x) for el in self.pressure_elements])

    def calculate_wave_drag(self):
        """Calculate wave drag of patch.

        Returns
        -------
        None
        """
        xo = -10.1 * config.flow.lam
        xTrough, = fmin(self.get_free_surface_height, xo, disp=False)
        xCrest, = fmin(lambda x: -self.get_free_surface_height(x), xo, disp=False)
        self.Dw = (
            0.0625
            * config.flow.density
            * config.flow.gravity
            * (
                self.get_free_surface_height(xCrest)
                - self.get_free_surface_height(xTrough)
            )
            ** 2
        )

    def print_forces(self):
        """Print forces to screen."""
        print(("Forces and Moment for {0}:".format(self.patch_name)))
        print(("    Total Drag [N]      : {0:6.4e}".format(self.D)))
        print(("    Wave Drag [N]       : {0:6.4e}".format(self.Dw)))
        print(("    Pressure Drag [N]   : {0:6.4e}".format(self.Dp)))
        print(("    Frictional Drag [N] : {0:6.4e}".format(self.Df)))
        print(("    Total Lift [N]      : {0:6.4e}".format(self.L)))
        print(("    Pressure Lift [N]   : {0:6.4e}".format(self.Lp)))
        print(("    Frictional Lift [N] : {0:6.4e}".format(self.Lf)))
        print(("    Moment [N-m]        : {0:6.4e}".format(self.M)))

    def write_forces(self):
        """Write forces to file."""
        kp.writeasdict(
            os.path.join(
                config.it_dir,
                "forces_{0}.{1}".format(self.patch_name, config.io.data_format),
            ),
            ["Drag", self.D],
            ["WaveDrag", self.Dw],
            ["PressDrag", self.Dp],
            ["FricDrag", self.Df],
            ["Lift", self.L],
            ["PressLift", self.Lp],
            ["FricLift", self.Lf],
            ["Moment", self.M],
            ["BasePt", self.base_pt],
            ["Length", self.length],
        )

    def load_forces(self):
        """Load forces from file."""
        K = kp.Dictionary(
            os.path.join(
                config.it_dir,
                "forces_{0}.{1}".format(self.patch_name, config.io.data_format),
            )
        )
        self.D = K.read("Drag", 0.0)
        self.Dw = K.read("WaveDrag", 0.0)
        self.Dp = K.read("PressDrag", 0.0)
        self.Df = K.read("FricDrag", 0.0)
        self.L = K.read("Lift", 0.0)
        self.Lp = K.read("PressLift", 0.0)
        self.Lf = K.read("FricLift", 0.0)
        self.M = K.read("Moment", 0.0)
        self.base_pt = K.read("BasePt", 0.0)
        self.length = K.read("Length", 0.0)


class PressureCushion(PressurePatch):
    """Pressure Cushion consisting solely of source elements.

    Args
    ----
    dict_ : krampy.Dictionary or str
        Dictionary instance or string with dictionary path to load properties.

    **kwargs : dict
        Keyword arguments.

    Attributes
    ----------
    patch_name : str
        Name of patch

    base_pt : float
        Right end of `PressurePatch`.
    """

    count = 0

    def __init__(self, dict_, **kwargs):
        super(self.__class__, self).__init__(self, **kwargs)
        self.index = PressureCushion.count
        PressureCushion.count += 1

        self.patch_name = dict_.read(
            "pressureCushionName", "pressureCushion{0}".format(self.index)
        )

        # TODO: cushion_type variability should be managed by sub-classing
        # PressurePatch
        self.cushion_type = dict_.read("cushionType", "")
        self.cushion_pressure = kwargs.get(
            "Pc", dict_.read_load_or_default("cushionPressure", 0.0)
        )

        self.neighbor_up = PlaningSurface.find_by_name(
            dict_.read("upstreamPlaningSurface", "")
        )
        self.neighbor_down = PlaningSurface.find_by_name(
            dict_.read("downstreamPlaningSurface", "")
        )

        if self.neighbor_down is not None:
            self.neighbor_down.upstream_pressure = self.cushion_pressure
        if self.neighbor_up is not None:
            self.neighbor_up.kutta_pressure = self.cushion_pressure

        if self.cushion_type == "infinite":
            # Dummy element, will have 0 pressure
            self.pressure_elements += [
                pe.AftSemiInfinitePressureBand(is_source=True, is_on_body=False)
            ]
            self.pressure_elements += [
                pe.AftSemiInfinitePressureBand(is_source=True, is_on_body=False)
            ]
            self._end_pts[0] = -1000.0  # doesn't matter where

        else:
            self.pressure_elements += [
                pe.ForwardHalfTriangularPressureElement(
                    is_source=True, is_on_body=False
                )
            ]

            Nfl = dict_.read("numElements", 10)
            self.smoothing_factor = dict_.read("smoothingFactor", np.nan)
            for n in [self.neighbor_down, self.neighbor_up]:
                if n is None and ~np.isnan(self.smoothing_factor):
                    self.pressure_elements += [
                        pe.CompleteTriangularPressureElement(
                            is_source=True, is_on_body=False
                        )
                        for __ in range(Nfl)
                    ]

            self.end_pts = [
                dict_.read_load_or_default(key, 0.0)
                for key in ["downstreamLoc", "upstreamLoc"]
            ]

            self.pressure_elements += [
                pe.AftHalfTriangularPressureElement(is_source=True, is_on_body=False)
            ]

        self._update_end_pts()

    @property
    def base_pt(self):
        if self.neighbor_up is None:
            return self._end_pts[1]
        else:
            return self.neighbor_up.base_pt

    @base_pt.setter
    def base_pt(self, x):
        self._end_pts[1] = x

    def _get_element_coords(self):
        """Return x-locations of all elements."""
        if not self.cushion_type == "smoothed" or np.isnan(self.smoothing_factor):
            return self.end_pts
        else:
            add_width = np.arctanh(0.99) * self.length / (2 * self.smoothing_factor)
            addL = np.linspace(
                -add_width, add_width, len(self.pressure_elements) / 2 + 1
            )
            x = np.hstack((self.end_pts[0] + addL, self.end_pts[1] + addL))
            return x

    def _update_end_pts(self):
        if self.neighbor_down is not None:
            self._end_pts[0] = self.neighbor_down.end_pts[1]
        if self.neighbor_up is not None:
            self._end_pts[1] = self.neighbor_up.end_pts[0]

        self._reset_element_coords()

        # Set source pressure of elements
        for elNum, el in enumerate(self.pressure_elements):
            el.is_source = True
            if self.cushion_type == "smoothed" and not np.isnan(self.smoothing_factor):
                alf = 2 * self.smoothing_factor / self.length
                el.set_source_pressure(
                    0.5
                    * self.cushion_pressure
                    * (
                        np.tanh(alf * el.x_coord)
                        - np.tanh(alf * (el.x_coord - self.length))
                    )
                )
            # for infinite pressure cushion, first element is dummy, set to
            # zero, second is semiInfinitePressureBand and set to cushion
            # pressure
            elif self.cushion_type == "infinite":
                if elNum == 0:
                    el.pressure = 0.0
                else:
                    el.pressure = self.cushion_pressure
            else:
                el.pressure = self.cushion_pressure

    @PressurePatch.length.setter
    def length(self, length):
        self._end_pts[0] = self._end_pts[1] - length

    def calculate_forces(self):
        self.calculate_wave_drag()


class PlaningSurface(PressurePatch):
    """Planing Surface consisting of unknown elements."""

    count = 0
    all = []

    @classmethod
    def find_by_name(cls, name):
        """Return first planing surface matching provided name.

        Returns
        -------
        PlaningSurface instance or None of no match found
        """
        if not name == "":
            matches = [o for o in cls.all if o.patch_name == name]
            if len(matches) > 0:
                return matches[0]
        return None

    def __init__(self, dict_, **kwargs):
        PressurePatch.__init__(self)
        self.index = PlaningSurface.count
        PlaningSurface.count += 1
        PlaningSurface.all.append(self)

        self.patch_name = dict_.read("substructureName", "")
        Nfl = dict_.read("Nfl", 0)
        self.point_spacing = dict_.read("pointSpacing", "linear")
        self.initial_length = dict_.read("initialLength", None)
        self.minimum_length = dict_.read("minimumLength", 0.0)
        self.maximum_length = dict_.read("maximum_length", float("Inf"))

        self.spring_constant = dict_.read("springConstant", 1e4)
        self.kutta_pressure = kwargs.get(
            "kuttaPressure", dict_.read_load_or_default("kuttaPressure", 0.0)
        )
        self._upstream_pressure = kwargs.get(
            "upstreamPressure", dict_.read_load_or_default("upstreamPressure", 0.0)
        )

        self.is_initialized = False
        self.is_active = True
        self.is_kutta_unknown = True
        self.is_sprung = dict_.read("isSprung", False)
        if self.is_sprung:
            self.initial_length = 0.0
            self.minimum_length = 0.0
            self.maximum_length = 0.0

        self.pressure_elements += [
            pe.ForwardHalfTriangularPressureElement(is_on_body=False)
        ]

        self.pressure_elements += [
            pe.ForwardHalfTriangularPressureElement(
                is_source=True, pressure=self.kutta_pressure
            )
        ]
        self.pressure_elements += [
            pe.CompleteTriangularPressureElement() for __ in range(Nfl - 1)
        ]
        self.pressure_elements += [
            pe.AftHalfTriangularPressureElement(
                is_source=True, pressure=self.upstream_pressure
            )
        ]

        for el in self.pressure_elements:
            el.parent = self

        # Define point spacing
        if self.point_spacing == "cosine":
            self.pct = 0.5 * (1 - kp.cosd(np.linspace(0, 180, Nfl + 1)))
        else:
            self.pct = np.linspace(0.0, 1.0, Nfl + 1)
        self.pct /= self.pct[-2]
        self.pct = np.hstack((0.0, self.pct))

    @property
    def upstream_pressure(self):
        if self.neighbor_up is not None:
            return self.neighbor_up.cushion_pressure
        else:
            return 0.0

    @property
    def downstream_pressure(self):
        if self.neighbor_down is not None:
            return self.neighbor_down.cushion_pressure
        else:
            return 0.0

    @upstream_pressure.setter
    def upstream_pressure(self, pressure):
        self._upstream_pressure = pressure
        el = self.pressure_elements[-1]
        el.pressure = pressure
        el.is_source = True
        el.z_coord = np.nan

    @PressurePatch.length.setter
    def length(self, length):
        length = np.max([length, 0.0])

        x0 = self.interpolator.get_separation_point()[0]
        self._end_pts = [x0, x0 + length]
        self._reset_element_coords()

    def initialize_end_pts(self):
        """Initialize end points to be at separation point and wetted length
        root.
        """
        self.base_pt = self.interpolator.get_separation_point()[0]

        if not self.is_initialized:
            if self.initial_length is None:
                self.length = self.interpolator.immersed_length - self.base_pt
            else:
                self.length = self.initial_length
            self.is_initialized = True

    def _reset_element_coords(self):
        """Set width of first element to be twice as wide."""
        PressurePatch._reset_element_coords(self)
        x = self._get_element_coords()
        self.pressure_elements[0].width = x[2] - x[0]

    def _get_element_coords(self):
        """Get position of pressure elements."""
        return self.pct * self.length + self._end_pts[0]

    def get_residual(self):
        """Get residual to drive first element pressure to zero."""
        if self.length <= 0.0:
            return 0.0
        else:
            return self.pressure_elements[0].pressure / config.flow.stagnation_pressure

    def calculate_forces(self):
        # TODO: Re-factor, this is messy."
        if self.length > 0.0:
            el = [el for el in self.pressure_elements if el.is_on_body]
            self.x = np.array([eli.x_coord for eli in el])
            self.p = np.array([eli.pressure for eli in el])
            self.p += self.pressure_elements[0].pressure
            self.shear_stress = np.array([eli.shear_stress for eli in el])
            self.s = np.array([self.interpolator.getSFixedX(xx) for xx in self.x])

            self.fP = interp1d(self.x, self.p, bounds_error=False, fill_value=0.0)
            self.fTau = interp1d(
                self.x, self.shear_stress, bounds_error=False, fill_value=0.0
            )

            AOA = kp.atand(self._get_body_derivative(self.x))

            self.Dp = kp.integrate(self.s, self.p * kp.sind(AOA))
            self.Df = kp.integrate(self.s, self.shear_stress * kp.cosd(AOA))
            self.Lp = kp.integrate(self.s, self.p * kp.cosd(AOA))
            self.Lf = -kp.integrate(self.s, self.shear_stress * kp.sind(AOA))
            self.D = self.Dp + self.Df
            self.L = self.Lp + self.Lf
            self.M = kp.integrate(
                self.x, self.p * kp.cosd(AOA) * (self.x - config.body.xCofR)
            )
        else:
            self.Dp = 0.0
            self.Df = 0.0
            self.Lp = 0.0
            self.Lf = 0.0
            self.D = 0.0
            self.L = 0.0
            self.M = 0.0
            self.x = []
        if self.is_sprung:
            self._apply_spring()

        self.calculate_wave_drag()

    def get_loads_in_range(self, x0, x1):
        """Return pressure and shear stress values at points between two
        arguments.
        """
        # Get indices within range unless length is zero
        if self.length > 0.0:
            ind = np.nonzero((self.x > x0) * (self.x < x1))[0]
        else:
            ind = []

        # Output all values within range
        if not ind == []:
            x = np.hstack((x0, self.x[ind], x1))
            p = np.hstack((self.fP(x0), self.p[ind], self.fP(x1)))
            tau = np.hstack((self.fTau(x0), self.shear_stress[ind], self.fTau(x1)))
        else:
            x = np.array([x0, x1])
            p = np.zeros_like(x)
            tau = np.zeros_like(x)
        return x, p, tau

    def _apply_spring(self):
        xs = self.pressure_elements[0].x_coord
        zs = self.pressure_elements[0].z_coord
        disp = zs - self.parent.get_free_surface_height(xs)
        Fs = -self.spring_constant * disp
        self.L += Fs
        self.M += Fs * (xs - config.body.xCofR)

    def _calculate_shear_stress(self):
        def shear_stress_func(xx):
            if xx == 0.0:
                return 0.0
            else:
                Rex = config.flow.flow_speed * xx / config.flow.kinematic_viscosity
                return (
                    0.332
                    * config.flow.density
                    * config.flow.flow_speed ** 2
                    * Rex ** -0.5
                )

        x = self._get_element_coords()[0:-1]
        s = np.array([self.interpolator.getSFixedX(xx) for xx in x])
        s = s[-1] - s

        for si, el in zip(s, self.pressure_elements):
            el.shear_stress = shear_stress_func(si)

    def _get_body_derivative(self, x, direction="r"):
        if isinstance(x, float):
            x = [x]
        return np.array(
            [
                kp.getDerivative(self.interpolator.getBodyHeight, xx, direction)
                for xx in x
            ]
        )
