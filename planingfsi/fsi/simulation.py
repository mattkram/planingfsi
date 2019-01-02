"""The ``planingfsi.fsi.simulation`` module houses the main class controlling the interaction of the fluid
and structural solvers.

"""
import json
import os
from glob import glob
from typing import List

import numpy

from planingfsi import config, logger
from planingfsi.dictionary import Dictionary, load_dict_from_file
from planingfsi.fe.structure import FEStructure
from planingfsi.fsi.interpolator import Interpolator
from planingfsi.potentialflow.solver import PotentialPlaningSolver
from .figure import FSIFigure


class Simulation(object):
    """Simulation object to manage the FSI problem. Handles the iteration
    between the fluid and solid solvers.

    Attributes:
        solid_solver (.FEStructure): Solid solver.
        fluid_solver (.PotentialPlaningSolver): Fluid solver.
        it_dirs (List[str]): A list of existing iteration directories.
        res_l (float): Lift residual.
        res_m (float): Moment residual.

    """

    def __init__(self):
        self.solid_solver = FEStructure()
        self.fluid_solver = PotentialPlaningSolver()
        self.it_dirs: List[str] = []
        self.res_l = 1.0
        self.res_m = 1.0

    #     def setFluidPressureFunc(self, func):
    #         self.fluidPressureFunc = func
    #
    #     def setSolidPositionFunc(self, func):
    #         self.solidPositionFunc = func

    def load_input_files(self):
        """Load input files defining the solid and fluid problems."""
        self._load_rigid_bodies()

        # Add all substructures
        for dict_name in glob(os.path.join(config.path.input_dict_dir, "*")):
            dict_ = Dictionary(os.path.join(config.path.input_dict_dir, dict_name))

            substructure = self.solid_solver.add_substructure(dict_)

            if dict_.get("hasPlaningSurface", False):
                planing_surface = self.fluid_solver.add_planing_surface(dict_)
                interpolator = Interpolator(substructure, planing_surface, dict_)
                interpolator.set_solid_position_function(substructure.get_coordinates)
                interpolator.set_fluid_pressure_function(
                    planing_surface.get_loads_in_range
                )

        # Add all pressure cushions
        if os.path.exists(config.path.cushion_dict_dir):
            for dict_name in glob(os.path.join(config.path.cushion_dict_dir, "*")):
                dict_ = load_dict_from_file(
                    os.path.join(config.path.cushion_dict_dir, dict_name)
                )
                self.fluid_solver.add_pressure_cushion(dict_)

    def _load_rigid_bodies(self):
        """Load rigid bodies from input files. If file doesn't exist, create default body."""
        if os.path.exists(config.path.body_dict_dir):
            for dict_name in glob(os.path.join(config.path.body_dict_dir, "*")):
                dict_ = load_dict_from_file(
                    os.path.join(config.path.body_dict_dir, dict_name)
                )
                self.solid_solver.add_rigid_body(dict_)
        else:
            self.solid_solver.add_rigid_body()

    def run(self):
        """Run the fluid-structure interaction simulation by iterating between the fluid and solid solvers."""
        config.it = 0
        self.solid_solver.load_mesh()

        if config.io.results_from_file:
            self.create_dirs()
            self.apply_ramp()
            self.it_dirs = glob("[0-9]*")

        if config.plotting.plot_any:
            self.figure = FSIFigure(self.solid_solver, self.fluid_solver)

        # Initialize body at specified trim and draft
        self.solid_solver.initialize_rigid_bodies()
        self.update_fluid_response()

        # Iterate between solid and fluid solvers until equilibrium
        while config.it <= config.solver.num_ramp_it or (
            self.get_residual() >= config.solver.max_residual
            and config.it <= config.solver.max_it
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

        if config.plotting.save and not config.plotting.fig_format == "png":
            config.plotting.fig_format = "png"
            self.figure.save()

        if config.io.write_time_histories:
            self.figure.write_time_histories()

        print("Execution complete")
        if config.plotting.show:
            self.figure.show()

    def update_fluid_response(self):
        """Update the fluid solver and transfer forces to solid solver."""
        self.fluid_solver.calculate_response()
        self.solid_solver.update_fluid_forces()

    def update_solid_response(self):
        """Calculate the solid solver response."""
        self.solid_solver.calculate_response()

    def create_dirs(self):
        """Create iteration and figure directories.

        Todo: Many of these commands should be moved elsewhere.

        """
        config.it_dir = os.path.join(config.path.case_dir, "{0}".format(config.it))
        config.path.fig_dir_name = os.path.join(
            config.path.case_dir, config.path.fig_dir_name
        )

        if self.check_output_interval() and not config.io.results_from_file:
            os.makedirs(config.path.case_dir, exist_ok=True)
            os.makedirs(config.it_dir, exist_ok=True)

        if config.plotting.save:
            os.makedirs(config.path.fig_dir_name, exist_ok=True)
            if config.it == 0:
                for f in os.listdir(config.path.fig_dir_name):
                    os.remove(os.path.join(config.path.fig_dir_name, f))

    def increment(self):
        if config.io.results_from_file:
            old_ind = numpy.nonzero(config.it == self.it_dirs)[0][0]
            if not old_ind == len(self.it_dirs) - 1:
                config.it = int(self.it_dirs[old_ind + 1])
            else:
                config.it = config.solver.max_it + 1
            self.create_dirs()
        else:
            config.it += 1

    def update_figure(self):
        if config.plotting.plot_any:
            self.figure.update()

    def get_body_res(self, x):
        self.solid_solver.get_pt_disp_rb(x[0], x[1])
        self.solid_solver.update_nodal_positions()
        self.update_fluid_response()

        # Write, print, and plot results
        self.create_dirs()
        self.write_results()
        self.print_status()
        self.update_figure()

        # Update iteration number depending on whether loading existing or
        # simply incrementing by 1
        if config.io.results_from_file:
            if config.it < len(self.it_dirs) - 1:
                config.it = int(self.it_dirs[config.it + 1])
            else:
                config.it = config.solver.max_it
            config.it += 1
        else:
            config.it += 1

        self.res_l = self.solid_solver.rigid_body[0].get_res_l()
        self.res_m = self.solid_solver.rigid_body[0].get_res_moment()

        print("Rigid Body Residuals:")
        print(("  Lift:   {0:0.4e}".format(self.res_l)))
        print(("  Moment: {0:0.4e}\n".format(self.res_m)))

        return numpy.array([self.res_l, self.res_m])

    def apply_ramp(self):
        if config.io.results_from_file:
            self.loadResults()
        else:
            if config.solver.num_ramp_it == 0:
                ramp = 1.0
            else:
                ramp = numpy.min((config.it / float(config.solver.num_ramp_it), 1.0))

            config.ramp = ramp
            config.relax_FEM = (
                1 - ramp
            ) * config.solver.relax_initial + ramp * config.solver.relax_final

    def get_residual(self):
        if config.io.results_from_file:
            return 1.0
            # return kp.Dictionary(
            #     os.path.join(config.it_dir, "overallQuantities.txt")
            # ).read("Residual", 0.0)
        else:
            return self.solid_solver.res

    def print_status(self):
        logger.info(
            "Residual after iteration {1:>4d}: {0:5.3e}".format(
                self.get_residual(), config.it
            )
        )

    def check_output_interval(self):
        return (
            config.it >= config.solver.max_it
            or self.get_residual() < config.solver.max_residual
            or numpy.mod(config.it, config.io.write_interval) == 0
        )

    def write_results(self):
        """Write the results to file."""
        if self.check_output_interval() and not config.io.results_from_file:
            dict_ = {"Ramp": config.ramp, "Residual": self.solid_solver.res}
            with open(os.path.join(config.it_dir, "overallQuantities.txt"), "w") as fp:
                json.dump(dict_, fp)

            self.fluid_solver.write_results()
            self.solid_solver.write_results()

    def load_results(self):
        dict_ = load_dict_from_file(
            os.path.join(config.it_dir, "overallQuantities.txt")
        )
        config.ramp = dict_.get("Ramp", 0.0)
        self.solid_solver.res = dict_.get("Residual", 0.0)
