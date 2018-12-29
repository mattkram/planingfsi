"""krampy provides useful utilities for other Python projects."""
import logging

VERSION = (0, 1, 1)

__version__ = ".".join(map(str, VERSION))

logger = logging.getLogger("krampy")
logger.setLevel("INFO")

from .iterator import Iterator

# from solver import RootFinder, fzero
# from trigonometry import *
# from general import *
# import unit
