"""PlaningFSI is a Python package for solving 2-D FSI problems.

PlaningFSI solves the general FSI problem of 2-D planing surfaces on a free
surface, where each surface can be rigid or flexible.
Each planing surface must belong to a substructure.
The substructures are part of a rigid body, which can be free in sinkage and trim.

The fluid solver is based on linearized potential-flow assumptions,
and the structural solver considers a large-deformation simple beam element.

"""
import logging
import os

logger = logging.getLogger("planingfsi")
logger.setLevel(logging.DEBUG)

from . import config
from .__version__ import __version__

# Use tk by default. Otherwise try Agg. Otherwise, disable plotting.
import matplotlib

_fallback_engine = "Agg"
if os.environ.get("DISPLAY") is None:
    matplotlib.use(_fallback_engine)
else:
    try:
        from matplotlib import pyplot
    except ImportError:
        try:
            matplotlib.use(_fallback_engine)
        except ImportError:
            config.plotting.plot_any = False
