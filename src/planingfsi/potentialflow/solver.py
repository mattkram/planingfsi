"""Fundamental module for constructing and solving planing potential flow problems."""
import weakref
from typing import List, Dict, Any, Optional

import numpy as np
from scipy.optimize import fmin

from .pressureelement import PressureElement
from .pressurepatch import PlaningSurface, PressureCushion, PressurePatch
from .. import solver, config, general, logger
from ..dictionary import load_dict_from_file
from ..fsi import simulation as fsi_simulation, figure


class PotentialPlaningSolver:
    """The top-level object which handles calculation of the potential flow problem.

    Pointers to the planing surfaces and pressure elements are stored for
    reference during initialization and problem setup. The potential planing
    calculation is then solved during iteration with the structural solver.
    """

    def __init__(self, simulation: "fsi_simulation.Simulation"):
        self._simulation = weakref.ref(simulation)
        self.X = None
        self.D = None
        self.Dw = None
        self.Dp = None
        self.Df = None
        self.L = None
        self.Lp = None
        self.Lf = None
        self.M = None
        self.p = None
        self.shear_stress = None
        self.xFS = None
        self.zFS = None
        self.xBar: Optional[float] = None

        self.planing_surfaces: List[PlaningSurface] = []
        self.pressure_cushions: List[PressureCushion] = []
        self.pressure_patches: List[PressurePatch] = []
        self.pressure_elements: List[PressureElement] = []
        self.solver: Optional[solver.RootFinder] = None

        self.min_len = None
        self.max_len = None
        self.init_len = None

    @property
    def simulation(self) -> "fsi_simulation.Simulation":
        """A reference to the simulation object by resolving the weak reference."""
        simulation = self._simulation()
        if simulation is None:
            raise ReferenceError("Simulation object cannot be accessed.")
        return simulation

    def _add_pressure_patch(self, instance: PressurePatch) -> None:
        """Add pressure patch to the calculation.

        Args:
            instance: The pressure patch object to add.

        """
        self.pressure_patches.append(instance)
        self.pressure_elements.extend([el for el in instance.pressure_elements])

    def add_planing_surface(self, dict_: Dict[str, Any]) -> PlaningSurface:
        """Add planing surface to the calculation from a dictionary file name.

        Args:
            dict_: The dictionary file.

        Returns:
            Instance created from dictionary.

        """
        instance = PlaningSurface(dict_)
        self.planing_surfaces.append(instance)
        self._add_pressure_patch(instance)
        return instance

    def add_pressure_cushion(self, dict_: Dict[str, Any]) -> PressureCushion:
        """Add pressure cushion to the calculation from a dictionary file name.

        Args:
            dict_: The dictionary file.

        Returns:
            Instance created from dictionary.

        """
        instance = PressureCushion(dict_)
        self.pressure_cushions.append(instance)
        self._add_pressure_patch(instance)
        return instance

    def calculate_pressure(self) -> None:
        """Calculate pressure of each element to satisfy body boundary conditions."""
        # Form lists of unknown and source elements
        unknown_el = [el for el in self.pressure_elements if not el.is_source and el.width > 0.0]
        nan_el = [el for el in self.pressure_elements if el.width <= 0.0]
        source_el = [el for el in self.pressure_elements if el.is_source]
        # Form arrays to build linear system
        # x location of unknown elements
        X = np.array([el.x_coord for el in unknown_el])

        # influence coefficient matrix (to be reshaped)
        A = np.array([el.get_influence_coefficient(x) for x in X for el in unknown_el])

        # height of each unknown element above waterline
        Z = np.array([el.z_coord for el in unknown_el]) - config.flow.waterline_height

        # subtract contribution from source elements
        Z -= np.array([np.sum([el.get_influence(x) for el in source_el]) for x in X])

        # Solve linear system
        dim = len(unknown_el)
        if not dim == 0:
            p = np.linalg.solve(np.reshape(A, (dim, dim)), np.reshape(Z, (dim, 1)))[:, 0]
        else:
            p = np.zeros_like(Z)

        # Apply calculated pressure to the elements
        for pi, el in zip(p, unknown_el):
            el.pressure = pi

        for el in nan_el:
            el.pressure = 0.0

    def _calculate_residual(self, wetted_length: np.ndarray) -> np.ndarray:
        """Return array of residuals of each planing surface to satisfy Kutta condition.

        Args:
            wetted_length: Array of wetted lengths of planing surfaces.

        Returns:
            Array of trailing edge residuals of each planing surface.

        """
        # Set length of each planing surface
        for Lwi, planing_surface in zip(wetted_length, self.planing_surfaces):
            planing_surface.length = np.min([Lwi, planing_surface.maximum_length])

        # Update bounds of pressure cushions
        for pressure_cushion in self.pressure_cushions:
            pressure_cushion.update_end_pts()

        # Solve for unknown pressures and output residual
        self.calculate_pressure()

        residual = np.array([p.get_residual() for p in self.planing_surfaces])

        def array_to_string(array: np.ndarray) -> str:
            """Convert an array to a string."""
            return ", ".join(["{0:11.4e}".format(a) for a in array]).join("[]")

        logger.info(f"      Lw:       {array_to_string(wetted_length)}")
        logger.info(f"      Residual: {array_to_string(residual)}")
        logger.info("\n")

        return residual

    def calculate_response(self) -> None:
        """Calculate response, including pressure and free-surface profiles.

        Will load results from file if specified.
        """
        if config.io.results_from_file:
            self.load_response()
        else:
            self.calculate_response_unknown_wetted_length()

    def calculate_response_unknown_wetted_length(self) -> None:
        """Calculate potential flow problem via iteration to find wetted length
        of all planing surfaces to satisfy all trailing edge conditions.
        """
        # Reset results so they will be recalculated after new solution
        self.xFS = None
        self.X = None

        # Initialize planing surface lengths and then solve until residual is
        # *zero*
        if len(self.planing_surfaces) > 0:
            for p in self.planing_surfaces:
                p.initialize_end_pts()

            print("  Solving for wetted length:")

            if self.min_len is None:
                self.min_len = np.array([p.minimum_length for p in self.planing_surfaces])
                self.max_len = np.array([p.maximum_length for p in self.planing_surfaces])
                self.init_len = np.array([p.initial_length for p in self.planing_surfaces])
            else:
                for i, p in enumerate(self.planing_surfaces):
                    L = p.length
                    self.init_len[i] = p.initial_length
                    if ~np.isnan(L) and L - self.min_len[i] > 1e-6:
                        # and self.solver.it < self.solver.maxIt:
                        self.init_len[i] = L * 1.0
                    else:
                        self.init_len[i] = p.initial_length

            dxMaxDec = config.solver.wetted_length_max_step_pct_dec * (self.init_len - self.min_len)
            dxMaxInc = config.solver.wetted_length_max_step_pct_inc * (self.init_len - self.min_len)

            for i, p in enumerate(self.planing_surfaces):
                if p.initial_length == 0.0:
                    self.init_len[i] = 0.0

            if self.solver is None:
                self.solver = solver.RootFinder(
                    self._calculate_residual,
                    self.init_len * 1.0,
                    config.solver.wetted_length_solver,
                    xMin=self.min_len,
                    xMax=self.max_len,
                    errLim=config.solver.wetted_length_tol,
                    dxMaxDec=dxMaxDec,
                    dxMaxInc=dxMaxInc,
                    firstStep=1e-6,
                    maxIt=config.solver.wetted_length_max_it_0,
                    maxJacobianResetStep=config.solver.wetted_length_max_jacobian_reset_step,
                    relax=config.solver.wetted_length_relax,
                )
            else:
                self.solver.max_it = config.solver.wetted_length_max_it
                self.solver.reinitialize(self.init_len * 1.0)
                self.solver.dx_max_increase = dxMaxInc
                self.solver.dx_max_decrease = dxMaxDec

            if any(self.init_len > 0.0):
                self.solver.solve()

            self.calculate_pressure_and_shear_profile()

    def calculate_pressure_and_shear_profile(self) -> None:
        """Calculate pressure and shear stress profiles over plate surface."""
        if self.X is None:
            if config.flow.include_friction:
                for p in self.planing_surfaces:
                    p._calculate_shear_stress()

            # Calculate forces on each patch
            for p in self.pressure_patches:
                p.calculate_forces()

            # Calculate pressure profile
            if len(self.pressure_patches) > 0:
                self.X = np.sort(
                    np.unique(np.hstack([p._get_element_coords() for p in self.pressure_patches]))
                )
                self.p = np.zeros_like(self.X)
                self.shear_stress = np.zeros_like(self.X)
            else:
                self.X = np.array([-1e6, 1e6])
                self.p = np.zeros_like(self.X)
                self.shear_stress = np.zeros_like(self.X)
            for el in self.pressure_elements:
                if el.is_on_body:
                    self.p[el.x_coord == self.X] += el.pressure
                    self.shear_stress[el.x_coord == self.X] += el.shear_stress

            # Calculate total forces as sum of forces on each patch
            for var in ["D", "Dp", "Df", "Dw", "L", "Lp", "Lf", "M"]:
                setattr(self, var, sum([getattr(p, var) for p in self.pressure_patches]))

            f = self.get_free_surface_height
            xo = -10.1 * config.flow.lam
            (xTrough,) = fmin(f, xo, disp=False)
            (xCrest,) = fmin(lambda x: -f(x), xo, disp=False)
            self.Dw = (
                0.0625 * config.flow.density * config.flow.gravity * (f(xCrest) - f(xTrough)) ** 2
            )

            # Calculate center of pressure
            self.xBar = general.integrate(self.X, self.p * self.X) / self.L
            if config.plotting.show_pressure:
                figure.plot_pressure(self)

    def calculate_free_surface_profile(self) -> None:
        """Calculate free surface profile."""
        if self.xFS is None:
            xFS = []
            # Grow points upstream and downstream from first and last plate
            for surf in self.planing_surfaces:
                if surf.length > 0:
                    pts = surf._get_element_coords()
                    xFS.append(
                        general.growPoints(
                            pts[1], pts[0], config.plotting.x_fs_min, config.plotting.growth_rate,
                        )
                    )
                    xFS.append(
                        general.growPoints(
                            pts[-2], pts[-1], config.plotting.x_fs_max, config.plotting.growth_rate,
                        )
                    )

            # Add points from each planing surface
            fsList = [
                patch._get_element_coords() for patch in self.planing_surfaces if patch.length > 0
            ]
            if len(fsList) > 0:
                xFS.append(np.hstack(fsList))

            # Add points from each pressure cushion
            xFS.append(np.linspace(config.plotting.x_fs_min, config.plotting.x_fs_max, 100))
            #             for patch in self.pressure_cushions:
            #                 if patch.neighborDown is not None:
            #                     ptsL = patch.neighborDown._get_element_coords()
            #                 else:
            #                     ptsL = np.array([patch.endPt[0] - 0.01, patch.endPt[0]])
            #
            #                 if patch.neighborDown is not None:
            #                     ptsR = patch.neighborUp._get_element_coords()
            #                 else:
            #                     ptsR = np.array([patch.endPt[1], patch.endPt[1] + 0.01])
            #
            #                 xEnd = ptsL[-1] + 0.5 * patch.get_length()
            #                 xFS.append(
            #                     kp.growPoints(ptsL[-2], ptsL[-1],
            #                                   xEnd, config.growthRate))
            #                 xFS.append(
            # kp.growPoints(ptsR[1],  ptsR[0],  xEnd, config.growthRate))

            # Sort x locations and calculate surface heights
            if len(xFS) > 0:
                self.xFS = np.sort(np.unique(np.hstack(xFS)))
                self.zFS = np.array(list(map(self.get_free_surface_height, self.xFS)))
            else:
                self.xFS = np.array([config.plotting.xFSMin, config.plotting.xFSMax])
                self.zFS = np.zeros_like(self.xFS)

    def get_free_surface_height(self, x: float) -> float:
        """Return free surface height at a given x-position considering the
        contributions from all pressure patches.

        Args:
            x: x-position at which free surface height shall be returned.

        Returns:
            Free-surface position at input x-position.

        """
        return sum([patch.get_free_surface_height(x) for patch in self.pressure_patches])

    def get_free_surface_derivative(self, x: float, direction: str = "c") -> float:
        """Return slope (derivative) of free-surface profile.

        Args:
            x: x-position.
            direction: The differencing direction, c: central, r: right, l: left.

        Returns:
            Derivative or slope of free-surface profile.

        """
        return general.getDerivative(self.get_free_surface_height, x, direction=direction)

    def write_results(self) -> None:
        """Write results to file."""
        # Post-process results from current solution
        #    self.calculate_pressure_and_shear_profile()
        self.calculate_free_surface_profile()

        if self.D is not None:
            self.write_forces()
        if self.X is not None:
            self.write_pressure_and_shear()
        if self.xFS is not None:
            self.write_free_surface()

    def write_pressure_and_shear(self) -> None:
        """Write pressure and shear stress profiles to data file."""
        if self.pressure_elements:
            general.writeaslist(
                self.simulation.it_dir / f"pressureAndShear.{config.io.data_format}",
                ["x [m]", self.X],
                ["p [Pa]", self.p],
                ["shear_stress [Pa]", self.shear_stress],
            )

    def write_free_surface(self) -> None:
        """Write free-surface profile to file."""
        if self.pressure_elements:
            general.writeaslist(
                self.simulation.it_dir / f"freeSurface.{config.io.data_format}",
                ["x [m]", self.xFS],
                ["y [m]", self.zFS],
            )

    def write_forces(self) -> None:
        """Write forces to file."""
        if self.pressure_elements:
            general.writeasdict(
                self.simulation.it_dir / f"forces_total.{config.io.data_format}",
                ["Drag", self.D],
                ["WaveDrag", self.Dw],
                ["PressDrag", self.Dp],
                ["FricDrag", self.Df],
                ["Lift", self.L],
                ["PressLift", self.Lp],
                ["FricLift", self.Lf],
                ["Moment", self.M],
            )

        for patch in self.pressure_patches:
            patch.write_forces()

    def load_response(self) -> None:
        """Load results from file."""
        self.load_forces()
        self.load_pressure_and_shear()
        self.load_free_surface()

    def load_pressure_and_shear(self) -> None:
        """Load pressure and shear stress from file."""
        self.X, self.p, self.shear_stress = np.loadtxt(
            str(self.simulation.it_dir / f"pressureAndShear.{config.io.data_format}"), unpack=True,
        )
        for el in [el for patch in self.planing_surfaces for el in patch.pressure_elements]:
            compare = np.abs(self.X - el.get_xloc()) < 1e-6
            if any(compare):
                el.set_pressure(self.p[compare][0])
                el.set_shear_stress(self.shear_stress[compare][0])

        for p in self.planing_surfaces:
            p.calculate_forces()

    def load_free_surface(self) -> None:
        """Load free surface coordinates from file."""
        try:
            data = np.loadtxt(str(self.simulation.it_dir / f"freeSurface.{config.io.data_format}"))
            self.xFS = data[:, 0]
            self.zFS = data[:, 1]
        except IOError:
            self.zFS = np.zeros_like(self.xFS)

    def load_forces(self) -> None:
        """Load forces from file."""
        dict_ = load_dict_from_file(
            self.simulation.it_dir / f"forces_total.{config.io.data_format}"
        )
        self.D = dict_.get("Drag", 0.0)
        self.Dw = dict_.get("WaveDrag", 0.0)
        self.Dp = dict_.get("PressDrag", 0.0)
        self.Df = dict_.get("FricDrag", 0.0)
        self.L = dict_.get("Lift", 0.0)
        self.Lp = dict_.get("PressLift", 0.0)
        self.Lf = dict_.get("FricLift", 0.0)
        self.M = dict_.get("Moment", 0.0)

        for patch in self.pressure_patches:
            patch.load_forces()
