"""High-level control of a `planingfsi` simulation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from planingfsi import logger
from planingfsi import writers

# TODO: There is an import cycle making this noreorder line necessary
from planingfsi.config import Config
from planingfsi.dictionary import load_dict_from_file
from planingfsi.potentialflow.solver import PotentialPlaningSolver

if TYPE_CHECKING:
    from planingfsi.fe.rigid_body import RigidBody


class Simulation:
    """Simulation object to manage the FSI problem.

    Handles the iteration between the fluid and solid solvers.

    Attributes:
        solid_solver (StructuralSolver): The structural solver.
        fluid_solver (PotentialPlaningSolver): The fluid solver.
        it (int): The current iteration.
        ramp (float): The current ramping coefficient, used when applying loads
            and moving nodes. The value is between 1.0 and 0.0.

    """

    it_dirs: list[Path]

    def __init__(self) -> None:
        # TODO: Remove after circular dependencies resolved
        from planingfsi.fe.structure import StructuralSolver  # noqa: F811

        self.config = Config()
        self.solid_solver = StructuralSolver(self)
        self.fluid_solver = PotentialPlaningSolver(self)
        self._figure: FSIFigure | None = None
        self.it = 0
        self.ramp = 1.0

    @classmethod
    def from_input_files(cls, config_filename: Path | str) -> "Simulation":
        """Construct a `Simulation` object by loading input files."""
        simulation = cls()
        simulation.load_input_files(config_filename)
        return simulation

    @property
    def figure(self) -> FSIFigure | None:
        """Use a property for the figure object to initialize lazily."""
        from planingfsi.fsi.figure import FSIFigure  # noreorder

        if self._figure is None and self.config.plotting.plot_any:
            self._figure = FSIFigure(simulation=self, config=self.config)
        return self._figure

    def add_rigid_body(self, rigid_body: dict[str, Any] | None = None) -> RigidBody:
        """Add a rigid body to the simulation and solid solver.

        Args:
            rigid_body: An optional dictionary of values to construct the rigid body.

        """
        return self.solid_solver.add_rigid_body(rigid_body)

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
        return Path(self.config.path.case_dir, str(self.it))

    def create_dirs(self) -> None:
        self.config.path.fig_dir_name = os.path.join(
            self.config.path.case_dir, self.config.path.fig_dir_name
        )

        if self.check_output_interval() and not self.config.io.results_from_file:
            Path(self.config.path.case_dir).mkdir(exist_ok=True)
            Path(self.it_dir).mkdir(exist_ok=True)

        if self.config.plotting.save:
            Path(self.config.path.fig_dir_name).mkdir(exist_ok=True)
            if self.it == 0:
                for f in os.listdir(self.config.path.fig_dir_name):
                    os.remove(os.path.join(self.config.path.fig_dir_name, f))

    def load_input_files(self, config_filename: Path | str) -> None:
        """Load all of the input files."""
        self.config.load_from_file(config_filename)
        self._load_rigid_bodies()
        self._load_substructures()
        self._load_pressure_cushions()

    def _load_rigid_bodies(self) -> None:
        """Load all rigid bodies from files."""
        if Path(self.config.path.body_dict_dir).exists():
            for dict_path in Path(self.config.path.body_dict_dir).glob("*"):
                dict_ = load_dict_from_file(str(dict_path))
                self.add_rigid_body(dict_)
        else:
            # Add a dummy rigid body that cannot move
            self.add_rigid_body()
        print(f"Rigid Bodies: {self.solid_solver.rigid_body}")

    def _load_substructures(self) -> None:
        """Load all substructures from files."""
        for dict_path in Path(self.config.path.input_dict_dir).glob("*"):
            dict_ = load_dict_from_file(
                dict_path,
                key_map={
                    "substructureName": "name",
                    "initialLength": "initial_length",
                    "Nfl": "num_fluid_elements",
                    "minimumLength": "minimum_length",
                    "maximumLength": "maximum_length",
                    "kuttaPressure": "kutta_pressure",
                    "upstreamPressure": "upstream_pressure",
                    "isSprung": "is_sprung",
                    "springConstant": "spring_constant",
                    "pointSpacing": "point_spacing",
                    "waterlineHeight": "waterline_height",
                    "sSepPctStart": "separation_arclength_start_pct",
                    "sImmPctStart": "immersion_arclength_start_pct",
                    "bodyName": "body_name",
                    "Ps": "seal_pressure",
                    "PsMethod": "seal_pressure_method",
                    "overPressurePct": "seal_over_pressure_pct",
                    "cushionPressureType": "cushion_pressure_type",
                    "tipLoad": "tip_load",
                    "tipLoadPct": "tip_load_pct",
                    "tipConstraintHt": "tip_constraint_height",
                    "structInterpType": "struct_interp_type",
                    "structExtrap": "struct_extrap",
                    "initialAngle": "initial_angle",
                    "basePtPct": "base_pt_pct",
                    "relaxAng": "relaxation_angle",
                    "attachPct": "attach_pct",
                    "minimumAngle": "minimum_angle",
                    "maxAngleStep": "max_angle_step",
                    "attachedSubstructure": "attached_substructure_name",
                    "attachedSubstructureEnd": "attached_substructure_end",
                    "EA": "axial_stiffness",
                },
            )
            # TODO: This default fallback to config could be handled in the PlaningSurface class
            dict_.setdefault("waterline_height", self.config.flow.waterline_height)

            substructure = self.solid_solver.add_substructure(dict_)

            if dict_.get("hasPlaningSurface", False):
                planing_surface = self.fluid_solver.add_planing_surface(dict_)
                substructure.add_planing_surface(planing_surface, **dict_)
        print(f"Substructures: {self.solid_solver.substructure}")

    def _load_pressure_cushions(self) -> None:
        """Load all pressure cushions from files."""
        if Path(self.config.path.cushion_dict_dir).exists():
            for dict_path in Path(self.config.path.cushion_dict_dir).glob("*"):
                dict_ = load_dict_from_file(dict_path)
                self.fluid_solver.add_pressure_cushion(dict_)
        print(f"Pressure Cushions: {self.fluid_solver.pressure_cushions}")

    def run(self) -> None:
        """Run the fluid-structure interaction simulation.

        The fluid and solid solvers are solved iteratively until convergence is reached.

        """
        self.reset()

        if self.config.io.results_from_file:
            self.create_dirs()
            self.apply_ramp()
            self.it_dirs = sorted(Path(".").glob("[0-9]*"), key=lambda x: int(x.name))

        self.initialize_solvers()

        # Iterate between solid and fluid solvers until equilibrium
        while self.it <= self.config.solver.num_ramp_it or (
            self.get_residual() >= self.config.solver.max_residual
            and self.it <= self.config.solver.max_it
        ):

            # Calculate response
            if self.solid_solver.has_free_structure:
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
            and self.config.plotting.save
            and not self.config.plotting.fig_format == "png"
        ):
            self.config.plotting.fig_format = "png"
            self.figure.save()

        if self.figure is not None and self.config.io.write_time_histories:
            self.figure.write_time_histories()

        logger.info("Execution complete")

        if self.figure is not None and self.config.plotting.show:
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
        if self.config.io.results_from_file:
            old_ind = np.nonzero(self.it == self.it_dirs)[0][0]
            if not old_ind == len(self.it_dirs) - 1:
                self.it = int(self.it_dirs[old_ind + 1])
            else:
                self.it = self.config.solver.max_it + 1
            self.create_dirs()
        else:
            self.it += 1

    def update_figure(self) -> None:
        if self.figure is not None and self.config.plotting.plot_any:
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
        if self.config.io.results_from_file:
            if self.it < len(self.it_dirs) - 1:
                self.it = int(str(self.it_dirs[self.it + 1]))
            else:
                self.it = self.config.solver.max_it
            self.it += 1
        else:
            self.it += 1

        res_l = self.solid_solver.res_l
        res_m = self.solid_solver.res_m

        logger.info("Rigid Body Residuals:")
        logger.info("  Lift:   {0:0.4e}".format(res_l))
        logger.info("  Moment: {0:0.4e}\n".format(res_m))

        return np.array([res_l, res_m])

    def apply_ramp(self) -> None:
        if self.config.io.results_from_file:
            self.load_results()
        else:
            if self.config.solver.num_ramp_it == 0:
                self.ramp = 1.0
            else:
                self.ramp = np.min((self.it / float(self.config.solver.num_ramp_it), 1.0))

            self.config.solver.relax_FEM = (
                1 - self.ramp
            ) * self.config.solver.relax_initial + self.ramp * self.config.solver.relax_final

    def get_residual(self) -> float:
        if self.config.io.results_from_file:
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
            self.it >= self.config.solver.max_it
            or self.get_residual() < self.config.solver.max_residual
            or np.mod(self.it, self.config.io.write_interval) == 0
        )

    def write_results(self) -> None:
        if self.check_output_interval() and not self.config.io.results_from_file:
            writers.write_as_dict(
                os.path.join(self.it_dir, "overallQuantities.txt"),
                ["Ramp", self.ramp],
                ["Residual", self.solid_solver.res],
            )

            self.fluid_solver.write_results()
            self.solid_solver.write_results()

    def load_results(self) -> None:
        dict_ = load_dict_from_file(os.path.join(self.it_dir, "overallQuantities.txt"))
        self.ramp = dict_.get("Ramp", 0.0)
        self.solid_solver.res = dict_.get("Residual", 0.0)


if TYPE_CHECKING:
    from planingfsi.fe.structure import StructuralSolver  # noqa: F401
    from planingfsi.fsi.figure import FSIFigure  # noreorder, noqa: F401
