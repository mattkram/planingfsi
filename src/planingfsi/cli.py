"""Command line program for PlaningFSI.

This script is called from the command line with options to plot or load
results from file. It reads the fluid and solid body properties from dictionary
files, assembles the problem, and runs it.

"""
from __future__ import annotations

from pathlib import Path

import click
import click_log

from planingfsi import logger
from planingfsi.config import Config
from planingfsi.fe.femesh import Mesh
from planingfsi.simulation import Simulation


@click.group(name="planingfsi", help="Run the PlaningFSI program")
def cli() -> None:
    click_log.basic_config(logger)


@cli.command(name="run")
@click.option(
    "post_mode",
    "--post",
    is_flag=True,
    help="Run in post-processing mode, loading results from file and generating figures.",
)
@click.option("--plot-save", is_flag=True, help="Save the plots to figures.")
@click.option("--plot-show", is_flag=True, help="Show the plots in a pop-up window.")
@click.option(
    "new_case",
    "--new",
    is_flag=True,
    help="Force generate new case, deleting old results first.",
)
def run_planingfsi(post_mode: bool, plot_save: bool, plot_show: bool, new_case: bool) -> None:
    """Run the planingFSI solver."""
    simulation = Simulation.from_input_files("configDict")

    simulation.config.plotting.save = plot_save
    simulation.config.plotting.show = plot_show

    if post_mode:
        logger.info("Running in post-processing mode")
        simulation.config.plotting.save = True
        simulation.config.io.results_from_file = True

    if new_case:
        logger.info("Removing all time directories")
        for it_dir in Path(simulation.config.path.case_dir).glob("[0-9]*"):
            it_dir.unlink()

    simulation.run()


@cli.command(name="mesh")
@click.argument("mesh_dict", required=False)
@click.option("plot_show", "--show", is_flag=True)
@click.option("plot_save", "--save", is_flag=True)
@click.option("--verbose", is_flag=True)
def generate_mesh(mesh_dict: str | None, plot_show: bool, plot_save: bool, verbose: bool) -> None:
    """Generate the initial mesh."""
    config = Config.from_file("configDict")

    if mesh_dict is not None:
        config.path.mesh_dict_name = mesh_dict

    if not Path(config.path.mesh_dict_name).exists():
        raise FileNotFoundError(f"File {config.path.mesh_dict_name} does not exist")

    mesh = Mesh()
    with Path(config.path.mesh_dict_name).open() as fp:
        exec(fp.read())

    mesh.display(disp=verbose)
    mesh.plot(show=plot_show, save=plot_save)
    mesh.write(mesh_dir=config.path.mesh_dir_name)
