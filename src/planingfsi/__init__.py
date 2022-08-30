"""PlaningFSI is a Python package for solving 2d FSI problems.

PlaningFSI solves the general FSI problem of 2-D planing surfaces on a free
surface, where each surface can be rigid or flexible. The substructures are
part of a rigid body, which can be free in sinkage and trim.

The fluid solver is based on linearized potential-flow assumptions,
and the structural solver considers a large-deformation simple beam element.

"""
import logging

logging.basicConfig()

logger = logging.getLogger("planingfsi")
logger.setLevel(logging.INFO)

from planingfsi.fe.femesh import Mesh
from planingfsi.fe.substructure import FlexibleMembraneSubstructure
from planingfsi.fe.substructure import RigidSubstructure
from planingfsi.fe.substructure import TorsionalSpringSubstructure
from planingfsi.potentialflow.pressurepatch import PlaningSurface
from planingfsi.potentialflow.pressurepatch import PressureCushion
from planingfsi.simulation import Simulation

__all__ = [
    "logger",
    "Mesh",
    "FlexibleMembraneSubstructure",
    "RigidSubstructure",
    "TorsionalSpringSubstructure",
    "PlaningSurface",
    "PressureCushion",
    "Simulation",
]
