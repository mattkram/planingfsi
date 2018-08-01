"""PlaningFSI solves the general FSI problem of 2-D planing surfaces on a free
surface, where each surface can be rigid or flexible. The substructures are
part of a rigid body, which can be free in sinkage and trim.

This script is called from the command line with options to plot or load
results from file. It reads the fluid and solid body properties from dictionary
files, assembles the problem, and runs it.

"""
import os
import sys

import click

from planingfsi import config as config
from planingfsi import io
from planingfsi import krampy
from planingfsi.fe.femesh import Mesh
from planingfsi.fsi.interpolator import Interpolator
from planingfsi.fsi.simulation import Simulation

# if os.environ.get('DISPLAY') is None:
#     import matplotlib as mpl
#     mpl.use('Agg')

# Use tk by default. Otherwise try Agg. Otherwise, disable plotting.
try:
    import matplotlib.pyplot
except ImportError:
    try:
        import matplotlib as mpl

        mpl.use('Agg')
    except ImportError:
        config.plotting.plot_any = False


@click.group()
@click.option('--post', is_flag=True)
@click.option('--plot_save', is_flag=True)
@click.option('--new', is_flag=True)
def planingfsi(post, plot_save, new):
    if post:
        print('Running in post-processing mode')
        config.plotting.save = True
        config.plotting.plot_any = True
        config.io.results_from_file = True
    if plot_save:
        config.plotting.save = True
        config.plotting.plot_any = True
    if new:
        krampy.rm_rf(krampy.find_files(config.path.case_dir, '[0-9]*'))

    # Create simulation
    sim = Simulation()

    # Add all rigid bodies
    if os.path.exists(config.path.body_dict_dir):
        for dict_name in krampy.listdir_nohidden(config.path.body_dict_dir):
            dict_ = io.Dictionary(os.path.join(config.path.body_dict_dir, dict_name))
            sim.solid.add_rigid_body(dict_)
    else:
        sim.solid.add_rigid_body()

    # Add all substructures
    for dict_name in krampy.listdir_nohidden(config.path.input_dict_dir):
        dict_ = io.Dictionary(os.path.join(config.path.input_dict_dir, dict_name))

        substructure = sim.solid.add_substructure(dict_)

        if dict_.read('hasPlaningSurface', False):
            planing_surface = sim.fluid.add_planing_surface(dict_)
            interpolator = Interpolator(substructure, planing_surface, dict_)
            interpolator.set_solid_position_function(substructure.get_coordinates)
            interpolator.set_fluid_pressure_function(
                planing_surface.get_loads_in_range)

    # Add all pressure cushions
    if os.path.exists(config.path.cushion_dict_dir):
        for dict_name in krampy.listdir_nohidden(config.path.cushion_dict_dir):
            dict_ = io.Dictionary(os.path.join(config.path.cushion_dict_dir, dict_name))
            sim.fluid.add_pressure_cushion(dict_)

    # Run simulation
    sim.run()


if __name__ == '__main__':
    planingfsi()


@click.command()
def mesh():
    # Process input arguments
    plot_show = False
    plot_save = False
    disp = False
    for arg in sys.argv[1:]:
        if not arg[0] == '-':
            config.path.mesh_dict_dir = arg
        if arg == '-show' or arg == 'show':
            plot_show = True
        if arg == '-save' or arg == 'save':
            plot_save = True
        if arg == '-print' or arg == 'print':
            disp = True

    # Create mesh
    M = Mesh()

    # Process mesh dictionary
    exec(open(config.path.mesh_dict_dir).read())

    # Display, plot, and write mesh files
    M.display(disp=disp)
    M.plot(show=plot_show, save=plot_save)
    M.write()
