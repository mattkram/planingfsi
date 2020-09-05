import os
from pathlib import Path
from typing import Optional, List

import numpy as np

from .figure import FSIFigure
from .interpolator import Interpolator
from .. import config, logger
from ..dictionary import load_dict_from_file
from ..fe.structure import StructuralSolver
from ..general import write_as_dict
from ..potentialflow.solver import PotentialPlaningSolver


class Simulation:
    """Simulation object to manage the FSI problem. Handles the iteration
    between the fluid and solid solvers.

    Attributes:
        solid_solver (StructuralSolver): The structural solver.
        fluid_solver (PotentialPlaningSolver): The fluid solver.

    """

    it_dirs: List[Path]

    def __init__(self) -> None:
        self.solid_solver = StructuralSolver(self)
        self.fluid_solver = PotentialPlaningSolver(self)
        self._figure: Optional[FSIFigure] = None
        self.it = 0
        self.ramp = 1.0

    @property
    def figure(self) -> Optional[FSIFigure]:
        """Use a property for the figure object to initialize lazily."""
        if self._figure is None and config.plotting.plot_any:
            self._figure = FSIFigure(self)
        return self._figure

    def update_fluid_response(self) -> None:
        """Update the fluid response and apply to the structural solver."""
        self.fluid_solver.calculate_response()
        self.solid_solver.update_fluid_forces()

    def update_solid_response(self) -> None:
        """Update the structural response."""
        self.solid_solver.calculate_response()

    @property
    def it_dir(self) -> Path:
        """A path to the directory for the current iteration."""
        return Path(config.path.case_dir, str(self.it))

    def create_dirs(self) -> None:
        config.path.fig_dir_name = os.path.join(config.path.case_dir, config.path.fig_dir_name)

        if self.check_output_interval() and not config.io.results_from_file:
            Path(config.path.case_dir).mkdir(exist_ok=True)
            Path(self.it_dir).mkdir(exist_ok=True)

        if config.plotting.save:
            Path(config.path.fig_dir_name).mkdir(exist_ok=True)
            if self.it == 0:
                for f in os.listdir(config.path.fig_dir_name):
                    os.remove(os.path.join(config.path.fig_dir_name, f))

    def load_input_files(self) -> None:
        """Load all of the input files."""
        self._load_rigid_bodies()
        self._load_substructures()
        self._load_pressure_cushions()

    def _load_rigid_bodies(self) -> None:
        """Load all rigid bodies from files."""
        if Path(config.path.body_dict_dir).exists():
            for dict_path in Path(config.path.body_dict_dir).glob("*"):
                dict_ = load_dict_from_file(str(dict_path))
                self.solid_solver.add_rigid_body(dict_)
        else:
            # Add a dummy rigid body that cannot move
            self.solid_solver.add_rigid_body()
        print(f"Rigid Bodies: {self.solid_solver.rigid_body}")

    def _load_substructures(self) -> None:
        """Load all substructures from files."""
        for dict_path in Path(config.path.input_dict_dir).glob("*"):
            dict_ = load_dict_from_file(dict_path)
            substructure = self.solid_solver.add_substructure(dict_)

            if dict_.get("hasPlaningSurface", False):
                planing_surface = self.fluid_solver.add_planing_surface(dict_)
                interpolator = Interpolator(substructure, planing_surface, dict_)
                interpolator.solid_position_function = substructure.get_coordinates
                interpolator.fluid_pressure_function = planing_surface.get_loads_in_range
        print(f"Substructures: {self.solid_solver.substructure}")

    def _load_pressure_cushions(self) -> None:
        """Load all pressure cushions from files."""
        if Path(config.path.cushion_dict_dir).exists():
            for dict_path in Path(config.path.cushion_dict_dir).glob("*"):
                dict_ = load_dict_from_file(dict_path)
                self.fluid_solver.add_pressure_cushion(dict_)
        print(f"Pressure Cushions: {self.fluid_solver.pressure_cushions}")

    def run(self) -> None:
        """Run the fluid-structure interaction simulation by iterating
        between the fluid and solid solvers.
        """
        self.reset()

        if config.io.results_from_file:
            self.create_dirs()
            self.apply_ramp()
            self.it_dirs = sorted(Path(".").glob("[0-9]*"), key=lambda x: int(x.name))

        self.initialize_solvers()

        # Iterate between solid and fluid solvers until equilibrium
        while self.it <= config.solver.num_ramp_it or (
            self.get_residual() >= config.solver.max_residual and self.it <= config.solver.max_it
        ):

            # Calculate response
            if config.has_free_structure:
                self.apply_ramp()
                self.update_solid_response()
                self.update_fluid_response()
                self.solid_solver.get_residual()
            else:
                self.solid_solver.res = 0.0

            # Write, print, and plot results
            self.create_dirs()
            self.write_results()
            self.print_status()
            self.update_figure()

            # Increment iteration count
            self.increment()

        if (
            self.figure is not None
            and config.plotting.save
            and not config.plotting.fig_format == "png"
        ):
            config.plotting.fig_format = "png"
            self.figure.save()

        if self.figure is not None and config.io.write_time_histories:
            self.figure.write_time_histories()

        logger.info("Execution complete")

        if self.figure is not None and config.plotting.show:
            self.figure.show()

    def reset(self) -> None:
        """Reset iteration counter and load the structural mesh."""
        self.it = 0
        self.solid_solver.load_mesh()

    def initialize_solvers(self) -> None:
        """Initialize body at specified trim and draft and solve initial fluid problem."""
        self.solid_solver.initialize_rigid_bodies()
        self.update_fluid_response()

    def increment(self) -> None:
        """Increment iteration counter. If loading from files, use the next stored iteration."""
        if config.io.results_from_file:
            old_ind = np.nonzero(self.it == self.it_dirs)[0][0]
            if not old_ind == len(self.it_dirs) - 1:
                self.it = int(self.it_dirs[old_ind + 1])
            else:
                self.it = config.solver.max_it + 1
            self.create_dirs()
        else:
            self.it += 1

    def update_figure(self) -> None:
        if self.figure is not None and config.plotting.plot_any:
            self.figure.update()

    def get_body_res(self, x: np.array) -> np.array:
        # self.solid_solver.get_pt_disp_rb(x[0], x[1])
        # self.solid_solver.update_nodal_positions()
        self.update_fluid_response()

        # Write, print, and plot results
        self.create_dirs()
        self.write_results()
        self.print_status()
        self.update_figure()

        # Update iteration number depending on whether loading existing or
        # simply incrementing by 1
        if config.io.results_from_file:
            if self.it < len(self.it_dirs) - 1:
                self.it = int(str(self.it_dirs[self.it + 1]))
            else:
                self.it = config.solver.max_it
            self.it += 1
        else:
            self.it += 1

        config.res_l = self.solid_solver.rigid_body[0].get_res_lift()
        config.res_m = self.solid_solver.rigid_body[0].get_res_moment()

        logger.info("Rigid Body Residuals:")
        logger.info("  Lift:   {0:0.4e}".format(config.res_l))
        logger.info("  Moment: {0:0.4e}\n".format(config.res_m))

        return np.array([config.res_l, config.res_m])

    def apply_ramp(self) -> None:
        if config.io.results_from_file:
            self.load_results()
        else:
            if config.solver.num_ramp_it == 0:
                ramp = 1.0
            else:
                ramp = np.min((self.it / float(config.solver.num_ramp_it), 1.0))

            # TODO: Remove reference to config.ramp eventually
            config.ramp = self.ramp = ramp
            config.solver.relax_FEM = (
                1 - ramp
            ) * config.solver.relax_initial + ramp * config.solver.relax_final

    def get_residual(self) -> float:
        if config.io.results_from_file:
            return 1.0
            return load_dict_from_file(os.path.join(self.it_dir, "overallQuantities.txt")).get(
                "Residual", 0.0
            )
        else:
            return self.solid_solver.res

    def print_status(self) -> None:
        logger.info(
            "Residual after iteration {1:>4d}: {0:5.3e}".format(self.get_residual(), self.it)
        )

    def check_output_interval(self) -> bool:
        return (
            self.it >= config.solver.max_it
            or self.get_residual() < config.solver.max_residual
            or np.mod(self.it, config.io.write_interval) == 0
        )

    def write_results(self) -> None:
        if self.check_output_interval() and not config.io.results_from_file:
            write_as_dict(
                os.path.join(self.it_dir, "overallQuantities.txt"),
                ["Ramp", config.ramp],
                ["Residual", self.solid_solver.res],
            )

            self.fluid_solver.write_results()
            self.solid_solver.write_results()

    def load_results(self) -> None:
        dict_ = load_dict_from_file(os.path.join(self.it_dir, "overallQuantities.txt"))
        config.ramp = dict_.get("Ramp", 0.0)
        self.solid_solver.res = dict_.get("Residual", 0.0)
