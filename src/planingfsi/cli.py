"""Command line program for PlaningFSI.

This script is called from the command line with options to plot or load
results from file. It reads the fluid and solid body properties from dictionary
files, assembles the problem, and runs it.

"""
from pathlib import Path
from typing import Optional

import click

from . import config
from . import logger
from .fe.femesh import Mesh
from .fsi.simulation import Simulation


@click.group(name="planingfsi", help="Run the PlaningFSI program")
def cli():
    pass


@cli.command(name="run")
@click.option("post_mode", "--post", is_flag=True)
@click.option("--plot_save", is_flag=True)
@click.option("new_case", "--new", is_flag=True)
def run_planingfsi(post_mode: bool, plot_save: bool, new_case: bool) -> None:
    if post_mode:
        logger.info("Running in post-processing mode")
        config.plotting.save = True
        config.plotting.plot_any = True
        config.io.results_from_file = True
    if plot_save:
        config.plotting.save = True
        config.plotting.plot_any = True
    if new_case:
        for it_dir in Path(config.path.case_dir).glob("[0-9]*"):
            it_dir.unlink()

    simulation = Simulation()
    simulation.load_input_files()
    simulation.run()


@cli.command(name="mesh")
@click.argument("mesh_dict", required=False)
@click.option("plot_show", "--show", is_flag=True)
@click.option("plot_save", "--save", is_flag=True)
@click.option("--verbose", is_flag=True)
def generate_mesh(
    mesh_dict: Optional[str], plot_show: bool, plot_save: bool, verbose: bool
) -> None:
    if mesh_dict is not None:
        config.path.mesh_dict_dir = mesh_dict

    if not Path(config.path.mesh_dict_dir).exists():
        raise FileNotFoundError(f"File {config.path.mesh_dict_dir} does not exist")

    mesh = Mesh()
    exec(Path(config.path.mesh_dict_dir).open("r").read())

    mesh.display(disp=verbose)
    mesh.plot(show=plot_show, save=plot_save)
    mesh.write()
