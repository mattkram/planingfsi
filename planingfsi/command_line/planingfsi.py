#!/usr/bin/env python
"""PlaningFSI solves the general FSI problem of 2-D planing surfaces on a free
surface, where each surface can be rigid or flexible. The substructures are
part of a rigid body, which can be free in sinkage and trim.

This script is called from the command line with options to plot or load
results from file. It reads the fluid and solid body properties from dictionary
files, assembles the problem, and runs it.
"""
import planingfsi.config as config
import planingfsi.krampy as kp
from planingfsi import io
from planingfsi.fsi.fsisimulation import Simulation
from planingfsi.fsi.fsiinterpolator import Interpolator

import os
import sys


def main():
    # Process command line arguments
    for arg in sys.argv[1:]:
        if 'post' in arg:
            print('Running in post-processing mode')
            config.plotting.save = True
            config.plotting.plot_any = True
            config.io.results_from_file = True
        if 'plotSave' in arg:
            config.plotting.save = True
            config.plotting.plot_any = True
        if 'new' in arg:
            kp.rm_rf(kp.find_files(config.path.case_dir, '[0-9]*'))

    # Use tk by default. Otherwise try Agg. Otherwise, disable plotting.
    try:
        import matplotlib.pyplot
    except:
        try:
            import matplotlib as mpl
            mpl.use('Agg')
        except ImportError:
            config.plotting.plot_any = False

    # Create simulation
    sim = Simulation()

    # Add all rigid bodies
    if os.path.exists(config.path.body_dict_dir):
        for dict_name in kp.listdir_nohidden(config.path.body_dict_dir):
            dict_ = io.Dictionary(os.path.join(config.path.body_dict_dir, dict_name))
            sim.solid.add_rigid_body(dict_)
    else:
        sim.solid.add_rigid_body()

    # Add all substructures
    for dict_name in kp.listdir_nohidden(config.path.input_dict_dir):
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
        for dict_name in kp.listdir_nohidden(config.path.cushion_dict_dir):
            dict_ = io.Dictionary(os.path.join(config.path.cushion_dict_dir, dict_name))
            sim.fluid.add_pressure_cushion(dict_)

    # Run simulation
    sim.run()


if __name__ == '__main__':
    main()
