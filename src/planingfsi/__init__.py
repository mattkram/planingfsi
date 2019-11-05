"""PlaningFSI is a Python package for solving 2d FSI problems.

PlaningFSI solves the general FSI problem of 2-D planing surfaces on a free
surface, where each surface can be rigid or flexible. The substructures are
part of a rigid body, which can be free in sinkage and trim.

The fluid solver is based on linearized potential-flow assumptions,
and the structural solver considers a large-deformation simple beam element.

"""
import logging

logger = logging.getLogger("planingfsi")
logger.setLevel(logging.INFO)

# # Use tk by default. Otherwise try Agg. Otherwise, disable plotting.
# _fallback_engine = "Agg"
# if os.environ.get("DISPLAY") is None:
#     matplotlib.use(_fallback_engine)
# else:
#     try:
#         from matplotlib import pyplot
#     except ImportError:
#         try:
#             matplotlib.use(_fallback_engine)
#         except ImportError:
#             from .config import plotting
#
#             plotting.plot_any = False
