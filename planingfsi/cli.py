"""Command line program for PlaningFSI.

This script is called from the command line with options to plot or load
results from file. It reads the fluid and solid body properties from dictionary
files, assembles the problem, and runs it.

"""
import os

import click
import matplotlib

from planingfsi import config as config, logger
from planingfsi import krampy
from planingfsi.fe.femesh import Mesh
from planingfsi.fsi.simulation import Simulation

# Use tk by default. Otherwise try Agg. Otherwise, disable plotting.
_fallback_engine = 'Agg'
if os.environ.get('DISPLAY') is None:
    matplotlib.use(_fallback_engine)
else:
    try:
        from matplotlib import pyplot
    except ImportError:
        try:
            matplotlib.use(_fallback_engine)
        except ImportError:
            config.plotting.plot_any = False


@click.group()
@click.option('post_mode', '--post', is_flag=True)
@click.option('--plot_save', is_flag=True)
@click.option('new_case', '--new', is_flag=True)
def planingfsi(post_mode, plot_save, new_case):
    if post_mode:
        logger.info('Running in post-processing mode')
        config.plotting.save = True
        config.plotting.plot_any = True
        config.io.results_from_file = True
    if plot_save:
        config.plotting.save = True
        config.plotting.plot_any = True
    if new_case:
        krampy.rm_rf(krampy.find_files(config.path.case_dir, '[0-9]*'))

    simulation = Simulation()
    simulation.load_input_files()
    simulation.run()


@planingfsi.command(name='mesh')
@click.argument('mesh_dict', required=False)
@click.option('plot_show', '--show', is_flag=True)
@click.option('plot_save', '--save', is_flag=True)
@click.option('--verbose', is_flag=True)
def generate_mesh(mesh_dict, plot_show, plot_save, verbose):
    if not mesh_dict:
        config.path.mesh_dict_dir = mesh_dict

    mesh = Mesh()

    exec(open(config.path.mesh_dict_dir).read())

    mesh.display(disp=verbose)
    mesh.plot(show=plot_show, save=plot_save)
    mesh.write()
