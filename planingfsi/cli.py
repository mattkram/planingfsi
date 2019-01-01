"""Command line program for PlaningFSI.

This script is called from the command line with options to plot or load
results from file. It reads the fluid and solid body properties from dictionary
files, assembles the problem, and runs it.

Todo: Improve docstring

"""
import os
import shutil
from glob import glob

import click

from . import config
from . import logger
from .fe.femesh import Mesh
from .fsi.simulation import Simulation


@click.group(help="Run the PlaningFSI program")
def main():
    pass


@main.command("run")
@click.option(
    "post_mode",
    "--post",
    is_flag=True,
    help="Run in post-processing mode, loading results from file and generating figures.",
)
@click.option("plot_save", "--plot", is_flag=True, help="Save the plots to figures.")
@click.option(
    "new_case",
    "--new",
    is_flag=True,
    help="Force generate new case, deleting old results first.",
)
def run(post_mode, plot_save, new_case):
    """Run the planingFSI solver."""

    config.load_from_dict_file("configDict")

    if post_mode:
        logger.info("Running in post-processing mode")
        config.plotting.save = True
        config.plotting.plot_any = True
        config.io.results_from_file = True
    if plot_save:
        config.plotting.save = True
        config.plotting.plot_any = True
    if new_case:
        logger.info("Removing all time directories")
        for time_dir in glob(os.path.join(config.path.case_dir, "[0.validated-9]*")):
            shutil.rmtree(time_dir)

    simulation = Simulation()
    simulation.load_input_files()
    simulation.run()


@main.command(name="mesh")
@click.argument("mesh_dict", required=False)
@click.option("plot_show", "--show", is_flag=True)
@click.option("plot_save", "--save", is_flag=True)
@click.option("--verbose", is_flag=True)
def generate_mesh(mesh_dict, plot_show, plot_save, verbose):
    """Generate the initial mesh."""
    if not mesh_dict:
        config.path.mesh_dict_dir = mesh_dict

    mesh = Mesh()

    exec(open(config.path.mesh_dict_dir).read())

    mesh.display(disp=verbose)
    mesh.plot(show=plot_show, save=plot_save)
    mesh.write()


if __name__ == "__main__":
    run()
