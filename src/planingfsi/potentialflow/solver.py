"""Fundamental module for constructing and solving planing potential flow problems."""
from __future__ import annotations

import weakref
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from scipy.optimize import fmin

from planingfsi import figure
from planingfsi import logger
from planingfsi import math_helpers
from planingfsi import solver
from planingfsi import writers
from planingfsi.config import Config
from planingfsi.potentialflow import pressurepatch
from planingfsi.potentialflow.pressureelement import PressureElement

if TYPE_CHECKING:
    from planingfsi.simulation import Simulation


class PotentialPlaningSolver:
    """The top-level object which handles calculation of the potential flow problem.

    Pointers to the planing surfaces and pressure elements are stored for
    reference during initialization and problem setup. The potential planing
    calculation is then solved during iteration with the structural solver.
    """

    def __init__(self, simulation: Simulation):
        self._simulation = weakref.ref(simulation)

        self.planing_surfaces: list["pressurepatch.PlaningSurface"] = []
        self.pressure_cushions: list["pressurepatch.PressureCushion"] = []
        self.pressure_patches: list["pressurepatch.PressurePatch"] = []
        self.pressure_elements: list[PressureElement] = []

        self.x_coord = np.array([])
        self.pressure = np.array([])
        self.shear_stress = np.array([])
        self.x_coord_fs = np.array([])
        self.z_coord_fs = np.array([])

        self.solver: solver.RootFinder | None = None
        self.fluid_it = 0

        self.min_len = np.array([])
        self.max_len = np.array([])
        self.init_len = np.array([])

    @property
    def config(self) -> Config:
        """A reference to the simulation configuration."""
        return self.simulation.config

    @property
    def simulation(self) -> Simulation:
        """A reference to the simulation object by resolving the weak reference."""
        simulation = self._simulation()
        if simulation is None:
            raise ReferenceError("Simulation object cannot be accessed.")
        return simulation

    @property
    def drag_wave(self) -> float:
        """The wave drag, calculated from downstream wave amplitude."""
        f = self.get_free_surface_height
        x_init = -10 * self.config.flow.lam
        (x_trough,) = fmin(f, x_init, disp=False)
        (x_crest,) = fmin(lambda x: -f(x), x_init, disp=False)
        return (
            0.0625
            * self.config.flow.density
            * self.config.flow.gravity
            * (f(x_crest) - f(x_trough)) ** 2
        )

    @property
    def x_bar(self) -> float:
        """The center of lift."""
        return math_helpers.integrate(self.x_coord, self.pressure * self.x_coord) / self.lift_total

    def __getattr__(self, item: str) -> Any:
        """Calculate total forces as sum of forces on each patch."""
        if item in [
            "drag_total",
            "drag_pressure",
            "drag_friction",
            "drag_wave",
            "lift_total",
            "lift_pressure",
            "lift_friction",
            "moment_total",
        ]:
            return sum([getattr(p, item) for p in self.pressure_patches])
        raise AttributeError

    def _add_pressure_patch(self, instance: "pressurepatch.PressurePatch") -> None:
        """Add pressure patch to the calculation.

        Args:
            instance: The pressure patch object to add.

        """
        self.pressure_patches.append(instance)
        self.pressure_elements.extend([el for el in instance.pressure_elements])

    def add_planing_surface(
        self, dict_or_instance: dict[str, Any] | pressurepatch.PlaningSurface | None, /
    ) -> "pressurepatch.PlaningSurface":
        """Add planing surface to the calculation from a dictionary file name.

        Args:
            dict_or_instance: A dictionary of values, or a PlaningSurface instance.

        Returns:
            Instance created from dictionary.

        """
        if isinstance(dict_or_instance, pressurepatch.PlaningSurface):
            instance = dict_or_instance
        elif dict_or_instance is not None:
            instance = pressurepatch.PlaningSurface(**dict_or_instance)
        else:
            instance = pressurepatch.PlaningSurface()
        instance.parent = self
        self.planing_surfaces.append(instance)
        self._add_pressure_patch(instance)
        return instance

    def add_pressure_cushion(
        self,
        dict_or_instance: dict[str, Any] | pressurepatch.PressureCushion | None,
    ) -> "pressurepatch.PressureCushion":
        """Add pressure cushion to the calculation from a dictionary file name.

        Args:
            dict_or_instance: A dictionary of values, or a PressureCushion instance.

        Returns:
            Instance created from dictionary.

        """
        if isinstance(dict_or_instance, pressurepatch.PressureCushion):
            instance = dict_or_instance
        elif dict_or_instance is not None:
            instance = pressurepatch.PressureCushion(**dict_or_instance)
        else:
            instance = pressurepatch.PressureCushion()
        instance.parent = self
        self.pressure_cushions.append(instance)
        self._add_pressure_patch(instance)
        return instance

    def calculate_pressure(self) -> None:
        """Calculate pressure of each element to satisfy body boundary conditions."""
        # TODO: This calculation may be sped up by using vectorized functions
        # Form lists of unknown and source elements
        unknown_el = [el for el in self.pressure_elements if not el.is_source and el.width > 0.0]
        nan_el = [el for el in self.pressure_elements if el.width <= 0.0]
        source_el = [el for el in self.pressure_elements if el.is_source]
        # Form arrays to build linear system
        # x location of unknown elements
        x_array = np.array([el.x_coord for el in unknown_el])

        # influence coefficient matrix (to be reshaped)
        influence_matrix = np.array(
            [el.get_influence_coefficient(x) for x in x_array for el in unknown_el]
        )

        # height of each unknown element above waterline
        z_array = np.array([el.z_coord for el in unknown_el]) - self.config.flow.waterline_height

        # subtract contribution from source elements
        z_array -= np.array([np.sum([el.get_influence(x) for el in source_el]) for x in x_array])

        # Solve linear system
        dim = len(unknown_el)
        if not dim == 0:
            p = np.linalg.solve(
                np.reshape(influence_matrix, (dim, dim)), np.reshape(z_array, (dim, 1))
            )[:, 0]
        else:
            p = np.zeros_like(z_array)

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
            planing_surface.length = Lwi

        # Update bounds of pressure cushions
        for pressure_cushion in self.pressure_cushions:
            pressure_cushion.update_end_pts()

        # Solve for unknown pressures and output residual
        self.calculate_pressure()

        residual = np.array([p.get_residual() for p in self.planing_surfaces])

        def array_to_string(array: np.ndarray) -> str:
            """Convert an array to a string."""
            return ", ".join(["{0:11.4e}".format(a) for a in array]).join("[]")

        logger.info(f"    Wetted length iteration: {self.fluid_it}")
        logger.info(f"      Lw:       {array_to_string(wetted_length)}")
        logger.info(f"      Residual: {array_to_string(residual)}\n")

        self.fluid_it += 1

        return residual

    def _initialize_solver(self) -> None:
        """Initialize the solver before each call."""
        for p in self.planing_surfaces:
            p.initialize_end_pts()

        if self.solver is None:
            self.init_len = np.array([p.initial_length for p in self.planing_surfaces])
            self.min_len = np.array([p.minimum_length for p in self.planing_surfaces])
            self.max_len = np.array([p.maximum_length for p in self.planing_surfaces])
            self.solver = solver.RootFinder(
                self._calculate_residual,
                self.init_len,
                self.config.solver.wetted_length_solver,
                xMin=self.min_len,
                xMax=self.max_len,
                errLim=self.config.solver.wetted_length_tol,
                firstStep=1e-6,
                maxIt=self.config.solver.wetted_length_max_it_0,
                maxJacobianResetStep=self.config.solver.wetted_length_max_jacobian_reset_step,
                relax=self.config.solver.wetted_length_relax,
            )
        else:
            self.solver.max_it = self.config.solver.wetted_length_max_it
            for i, p in enumerate(self.planing_surfaces):
                length = p.length
                if np.isnan(length) or length - self.min_len[i] < 1e-6:
                    self.init_len[i] = p.initial_length
                else:
                    self.init_len[i] = length

        self.solver.reinitialize(self.init_len)
        self.solver.dx_max_decrease = self.config.solver.wetted_length_max_step_pct_dec * (
            self.init_len - self.min_len
        )
        self.solver.dx_max_increase = self.config.solver.wetted_length_max_step_pct_inc * (
            self.init_len - self.min_len
        )

    def calculate_response(self) -> None:
        """Calculate response, including pressure and free-surface profiles.

        Will load results from file if specified. Otherwise, calculate potential flow problem via
        iteration to find wetted length of all planing surfaces to satisfy all trailing edge
        conditions.

        """
        if self.config.io.results_from_file:
            self.load_results()
            return

        if not self.planing_surfaces:
            # Return early if there are no planing surfaces
            return

        # Initialize planing surface lengths and then solve until residual is *zero*
        self._initialize_solver()
        assert self.solver is not None

        if (self.init_len > 0.0).any():
            logger.info("  Solving for wetted length:")
            self.fluid_it = 0
            self.solver.solve()

        # Post-process results from current solution
        self.calculate_free_surface_profile()
        self.calculate_pressure_and_shear_profile()

        # Plot the pressure profile if specified
        if self.config.plotting.show_pressure:
            figure.plot_pressure(self)

    def calculate_pressure_and_shear_profile(self) -> None:
        """Calculate pressure and shear stress profiles over plate surface."""
        # Calculate forces on each patch
        for p in self.pressure_patches:
            p.calculate_forces()

        # Calculate pressure profile
        if self.pressure_patches:
            self.x_coord = np.unique(
                np.hstack([p.get_element_coords() for p in self.pressure_patches])
            )
        else:
            self.x_coord = np.array([-1e6, 1e6])
        self.pressure = np.zeros_like(self.x_coord)
        self.shear_stress = np.zeros_like(self.x_coord)

        for el in self.pressure_elements:
            if el.is_on_body:
                ind = el.x_coord == self.x_coord
                self.pressure[ind] += el.pressure
                self.shear_stress[ind] += el.shear_stress

    def calculate_free_surface_profile(self) -> None:
        """Calculate free surface profile."""
        x_fs: list[float] = []
        for surf in self.planing_surfaces:
            if surf.length > 0:
                # Add points from each planing surface
                x_fs.extend(surf.get_element_coords())

        # Grow points upstream and downstream from first and last plate
        x_fs_u = np.unique(x_fs)
        x_fs.extend(
            _grow_points(
                x_fs_u[-2],
                x_fs_u[-1],
                self.config.plotting.x_fs_max,
                self.config.plotting.growth_rate,
            )
        )
        x_fs.extend(
            _grow_points(
                x_fs_u[1],
                x_fs_u[0],
                self.config.plotting.x_fs_min,
                self.config.plotting.growth_rate,
            )
        )

        # Add points from each pressure cushion
        for patch in self.pressure_cushions:
            if patch.neighbor_down is not None:
                pts_down = np.unique(patch.neighbor_down.get_element_coords())
            else:
                pts_down = np.array([patch._end_pts[0] - 0.01, patch._end_pts[0]])

            if patch.neighbor_up is not None:
                pts_up = np.unique(patch.neighbor_up.get_element_coords())
            else:
                pts_up = np.array([patch._end_pts[1], patch._end_pts[1] + 0.01])

            x_mid = 0.5 * (pts_down[-1] + pts_up[0])
            x_fs.extend(
                _grow_points(pts_down[-2], pts_down[-1], x_mid, self.config.plotting.growth_rate)
            )
            x_fs.extend(_grow_points(pts_up[1], pts_up[0], x_mid, self.config.plotting.growth_rate))

        # Sort x locations and calculate surface heights
        self.x_coord_fs = np.unique(x_fs)
        self.z_coord_fs = np.array(list(map(self.get_free_surface_height, self.x_coord_fs)))

    def get_free_surface_height(self, x: float) -> float:
        """Return free surface height at a given x-position considering the
        contributions from all pressure patches.

        Args:
            x: x-position at which free surface height shall be returned.

        Returns:
            Free-surface position at input x-position.

        """
        return sum([patch.get_free_surface_height(x) for patch in self.pressure_patches])

    def write_results(self) -> None:
        """Write results to files."""
        self._write_forces()
        self._write_pressure_and_shear()
        self._write_free_surface()

    def _write_forces(self) -> None:
        """Write forces to file."""
        writers.write_as_dict(
            self.simulation.it_dir / f"forces_total.{self.config.io.data_format}",
            ["Drag", self.drag_total],
            ["WaveDrag", self.drag_wave],
            ["PressDrag", self.drag_pressure],
            ["FricDrag", self.drag_friction],
            ["Lift", self.lift_total],
            ["PressLift", self.lift_pressure],
            ["FricLift", self.lift_friction],
            ["Moment", self.moment_total],
        )

        for patch in self.pressure_patches:
            patch.write_forces()

    def _write_pressure_and_shear(self) -> None:
        """Write pressure and shear stress profiles to data file."""
        writers.write_as_list(
            self.simulation.it_dir / f"pressureAndShear.{self.config.io.data_format}",
            ["x [m]", self.x_coord],
            ["p [Pa]", self.pressure],
            ["shear_stress [Pa]", self.shear_stress],
        )

    def _write_free_surface(self) -> None:
        """Write free-surface profile to file."""
        writers.write_as_list(
            self.simulation.it_dir / f"freeSurface.{self.config.io.data_format}",
            ["x [m]", self.x_coord_fs],
            ["y [m]", self.z_coord_fs],
        )

    def load_results(self) -> None:
        """Load results from file."""
        self._load_forces()
        self._load_pressure_and_shear()
        self._load_free_surface()

    def _load_forces(self) -> None:
        """Load forces from file."""
        for patch in self.pressure_patches:
            patch.load_forces()

    def _load_pressure_and_shear(self) -> None:
        """Load pressure and shear stress from file."""
        self.x_coord, self.pressure, self.shear_stress = np.loadtxt(
            str(self.simulation.it_dir / f"pressureAndShear.{self.config.io.data_format}"),
            unpack=True,
        )
        for el in [el for patch in self.planing_surfaces for el in patch.pressure_elements]:
            compare = np.abs(self.x_coord - el.x_coord) < 1e-6
            if any(compare):
                el.pressure = self.pressure[compare][0]
                el.shear_stress = self.shear_stress[compare][0]

        for p in self.planing_surfaces:
            p.calculate_forces()

    def _load_free_surface(self) -> None:
        """Load free surface coordinates from file."""
        try:
            self.x_coord_fs, self.z_coord_fs = np.loadtxt(
                str(self.simulation.it_dir / f"freeSurface.{self.config.io.data_format}"),
                unpack=True,
            )
        except IOError:
            self.z_coord_fs = np.zeros_like(self.x_coord_fs)


def _grow_points(x0: float, x1: float, x_max: float, rate: float = 1.1) -> np.ndarray:
    """Grow points exponentially from two starting points assuming a growth rate.

    Args:
        x0: The first point.
        x1: The second point.
        x_max: The maximum distance.
        rate: The growth rate of spacing between subsequent points.

    """
    # TODO: Check this function, is first point included?
    dx = x1 - x0
    x = [x1]

    if dx > 0:

        def done(xt: float) -> bool:
            return xt > x_max

    elif dx < 0:

        def done(xt: float) -> bool:
            return xt < x_max

    else:

        def done(xt: float) -> bool:
            _ = xt
            return True

    while not done(x[-1]):
        x.append(x[-1] + dx)
        dx *= rate

    return np.array(x[1:])
