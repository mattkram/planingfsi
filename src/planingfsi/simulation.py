"""High-level control of a `planingfsi` simulation."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from planingfsi import Mesh
from planingfsi import logger
from planingfsi import writers
from planingfsi.config import Config
from planingfsi.dictionary import load_dict_from_file
from planingfsi.fe.structure import StructuralSolver
from planingfsi.figure import Figure
from planingfsi.potentialflow.pressurepatch import PlaningSurface
from planingfsi.potentialflow.solver import PotentialPlaningSolver

if TYPE_CHECKING:
    from planingfsi.fe.rigid_body import RigidBody


class Simulation:
    """Simulation object to manage the FSI problem.

    Handles the iteration between the fluid and solid solvers.

    Attributes:
        config: A reference to the `Config` object, which stores values "global" to this simulation.
            Each `Simulation` instance has its own independent config.
        structural_solver: The structural solver.
        fluid_solver: The fluid solver.
        it: The current iteration.
        ramp: The current ramping coefficient, used when applying loads and moving nodes.
            The value is between 1.0 and 0.0.

    """

    def __init__(self) -> None:
        self.config = Config()
        self.structural_solver = StructuralSolver(self)
        self.fluid_solver = PotentialPlaningSolver(self)
        self.it = 0
        self.ramp = 1.0
        self._figure: Figure | None = None
        self._it_dirs: list[Path] | None = None
        self._case_dir: Path | None = None

    @classmethod
    def from_input_files(cls, config_filename: Path | str) -> Simulation:
        """Construct a `Simulation` object by loading input files."""
        simulation = cls()
        simulation.load_input_files(config_filename)
        return simulation

    @property
    def case_dir(self) -> Path:
        """The base path for the simulation."""
        return self._case_dir or Path(self.config.path.case_dir)

    @case_dir.setter
    def case_dir(self, value: Path) -> None:
        self._case_dir = value

    @property
    def mesh_dir(self) -> Path:
        """A path to the mesh directory."""
        return self.case_dir / self.config.path.mesh_dir_name

    @property
    def fig_dir(self):
        """A path to the directory in which figures will be saved."""
        return self.case_dir / self.config.path.fig_dir_name

    @property
    def figure(self) -> Figure | None:
        """The `FSIFigure` object where results are drawn. Will be None if plotting is disabled."""
        if self._figure is None and self.config.plotting.plot_any:
            self._figure = Figure(simulation=self)
        return self._figure

    @property
    def it_dir(self) -> Path:
        """A path to the directory for the current iteration."""
        return self.case_dir / str(self.it)

    @property
    def it_dirs(self) -> list[Path]:
        if self._it_dirs is None:
            self._it_dirs = sorted(self.case_dir.glob("[0-9]*"), key=lambda x: int(x.name))
        return self._it_dirs

    @property
    def residual(self) -> float:
        return self.structural_solver.residual

    @property
    def is_converged(self) -> bool:
        return (
            self.structural_solver.residual < self.config.solver.max_residual
            and self.it > self.config.solver.num_ramp_it
        )

    @property
    def is_write_iteration(self) -> bool:
        """True if results should be written at the end of the current iteration."""
        return (
            self.it >= self.config.solver.max_it
            or self.residual < self.config.solver.max_residual
            or np.mod(self.it, self.config.io.write_interval) == 0
        )

    def add_rigid_body(self, rigid_body: dict[str, Any] | RigidBody | None = None) -> RigidBody:
        """Add a rigid body to the simulation and solid solver.

        Args:
            rigid_body: A `RigidBody` object, or optional dictionary of values to construct the rigid body.

        Returns:
            The `RigidBody` that was added to the simulation.

        """
        return self.structural_solver.add_rigid_body(rigid_body)

    def _update_fluid_response(self) -> None:
        """Update the fluid response and apply to the structural solver."""
        self.fluid_solver.calculate_response()
        self.structural_solver.update_fluid_forces()

    def _update_solid_response(self) -> None:
        """Update the structural response."""
        self.structural_solver.calculate_response()

    def load_input_files(self, config_filename: Path | str) -> None:
        """Load all the input files."""
        self.config.load_from_file(config_filename)
        self._load_rigid_bodies()
        self._load_substructures()
        self._load_pressure_cushions()
        self.load_mesh()

    def _load_rigid_bodies(self) -> None:
        """Load all rigid bodies from files in the body dict directory.

        If no files are provided, a default stationary rigid body is added.

        """
        body_dict_dir = self.case_dir / self.config.path.body_dict_dir_name
        for dict_path in body_dict_dir.glob("*"):
            # TODO: I may be missing old spellings in the key_map
            dict_ = load_dict_from_file(
                dict_path,
                key_map={
                    "bodyName": "name",
                    "W": "weight",
                    "loadPct": "load_pct",
                    "m": "mass",
                    "Iz": "rotational_inertia",
                    "xCofG": "x_cg",
                    "yCofG": "y_cg",
                    "xCofR": "x_cr",
                    "yCofR": "y_cr",
                },
            )

            # Fallback on self.config.body if these keys are missing from rigid body dict file
            fallback_keys = {
                "x_cg",
                "y_cg",
                "x_cr",
                "y_cr",
                "initial_draft",
                "initial_trim",
                "free_in_draft",
                "free_in_trim",
                "max_draft_step",
                "max_trim_step",
                "relax_draft",
                "relax_trim",
            }
            for key in fallback_keys:
                dict_.setdefault(key, getattr(self.config.body, key))

            dict_.setdefault("weight", self.config.body.seal_load_pct * self.config.body.weight)

            self.add_rigid_body(dict_)

        if not self.structural_solver.rigid_bodies:
            # Add a dummy rigid body that cannot move
            self.add_rigid_body()

        logger.info(f"Rigid Bodies: {self.structural_solver.rigid_bodies}")

    def _load_substructures(self) -> None:
        """Load all substructures from files."""
        input_dict_dir = self.case_dir / self.config.path.input_dict_dir_name
        for dict_path in input_dict_dir.glob("*"):
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
                    "hasPlaningSurface": "has_planing_surface",
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

            substructure = self.structural_solver.add_substructure(dict_)

            if dict_.get("has_planing_surface", False):
                planing_surface = PlaningSurface(**dict_)
                substructure.add_planing_surface(planing_surface, **dict_)
        logger.info(f"Substructures: {self.structural_solver.substructures}")

    def _load_pressure_cushions(self) -> None:
        """Load all pressure cushions from files."""
        cushion_dict_dir = self.case_dir / self.config.path.cushion_dict_dir_name
        for dict_path in cushion_dict_dir.glob("*"):
            dict_ = load_dict_from_file(
                dict_path,
                key_map={
                    "pressureCushionName": "name",
                    "cushionPressure": "cushion_pressure",
                    "upstreamPlaningSurface": "upstream_planing_surface",
                    "downstreamPlaningSurface": "downstream_planing_surface",
                    "upstreamLoc": "upstream_loc",
                    "downstreamLoc": "downstream_loc",
                    "numElements": "num_elements",
                    "cushionType": "cushion_type",
                    "smoothingFactor": "smoothing_factor",
                },
            )
            self.fluid_solver.add_pressure_cushion(dict_)
        logger.info(f"Pressure Cushions: {self.fluid_solver.pressure_cushions}")

    def load_mesh(self, mesh: Path | Mesh | None = None) -> None:
        """Load the mesh from files, or directly. By default, will load from "mesh" directory."""
        self.structural_solver.load_mesh(mesh or self.mesh_dir)

    def run(self) -> None:
        """Run the fluid-structure interaction simulation.

        The fluid and solid solvers are solved iteratively until convergence is reached.

        """
        self.it = 0

        if self.config.io.results_from_file:
            self._update_ramp()

        self.initialize_solvers()

        # Iterate between solid and fluid solvers until equilibrium
        while self.it <= self.config.solver.num_ramp_it or (
            self.residual >= self.config.solver.max_residual
            and self.it <= self.config.solver.max_it
        ):

            # Calculate response
            if self.structural_solver.has_free_structure:
                self._update_ramp()
                self._update_solid_response()
                self._update_fluid_response()

            # Write, print, and plot results
            self.write_results()
            logger.info("Residual after iteration %4s: %5.3e", self.it, self.residual)
            self._update_figure()

            # Increment iteration count
            self.increment()

        self._save_figure()

        logger.info("Execution complete")

        if self.figure is not None and self.config.plotting.show:
            self.figure.show()

    def initialize_solvers(self) -> None:
        """Initialize body at specified trim and draft and solve initial fluid problem."""
        self.structural_solver.initialize_rigid_bodies()
        self._update_fluid_response()

    def increment(self) -> None:
        """Increment iteration counter. If loading from files, use the next stored iteration."""
        if self.config.io.results_from_file:
            stored_iterations = [int(it_dir.name) for it_dir in self.it_dirs]
            current_ind = stored_iterations.index(self.it)
            try:
                self.it = stored_iterations[current_ind + 1]
            except IndexError:
                self.it = self.config.solver.max_it + 1
        else:
            self.it += 1

    def _update_figure(self) -> None:
        if self.figure is not None and self.config.plotting.plot_any:
            self.figure.update()

    def _save_figure(self) -> None:
        """Save the final figure if configuration is correct."""
        if (
            self.figure is not None
            and self.config.plotting.save
            and not self.config.plotting.fig_format == "png"
        ):
            self.config.plotting.fig_format = "png"
            self.figure.save()

        if self.figure is not None and self.config.io.write_time_histories:
            self.figure.write_time_histories()

    def _update_ramp(self) -> None:
        """Update the ramp value based on the current iteration number."""
        if self.config.io.results_from_file:
            self._load_results()
        else:
            if self.config.solver.num_ramp_it == 0:
                self.ramp = 1.0
            else:
                self.ramp = min(self.it / self.config.solver.num_ramp_it, 1.0)

            self.config.solver.relax_FEM = (
                1 - self.ramp
            ) * self.config.solver.relax_initial + self.ramp * self.config.solver.relax_final

    def write_results(self) -> None:
        """Write the current overall results to an iteration directory."""
        if self.is_write_iteration and not self.config.io.results_from_file:
            self.it_dir.mkdir(exist_ok=True, parents=True)
            writers.write_as_dict(
                self.it_dir / "overallQuantities.txt",
                ["Ramp", self.ramp],
                ["Residual", self.structural_solver.residual],
            )

            self.fluid_solver.write_results()
            self.structural_solver.write_results()

    def _load_results(self) -> None:
        """Load the overall quantities from the results file."""
        dict_ = load_dict_from_file(self.it_dir / "overallQuantities.txt")
        self.ramp = dict_.get("Ramp", 0.0)
        self.structural_solver.residual = dict_.get("Residual", 0.0)
