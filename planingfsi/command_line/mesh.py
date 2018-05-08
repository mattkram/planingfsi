#!/usr/bin/env python

import os
import sys

if os.environ.get('DISPLAY') is None:
    import matplotlib as mpl
    mpl.use('Agg')

import numpy as np

import planingfsi.config as config
import planingfsi.unit as unit
import planingfsi.io as io

from planingfsi.fe.femesh import Mesh


def main():
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


if __name__ == '__main__':
    main()
