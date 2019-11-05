"""Load version number from pyproject.toml in project rood directory."""
import os

import toml

here = os.path.dirname(__file__)

pyproject_path = os.path.join(here, "..", "pyproject.toml")

with open(pyproject_path, "r") as f:
    dict_ = toml.load(f)

    __version__ = dict_["tool"]["poetry"]["version"]
