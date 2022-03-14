# PlaningFSI

[![Run Python Tests](https://github.com/mattkram/planingfsi/workflows/Run%20Python%20Tests/badge.svg)](https://github.com/mattkram/planingfsi/actions)
[![Coverage](https://codecov.io/gh/mattkram/planingfsi/branch/develop/graph/badge.svg)](https://codecov.io/gh/mattkram/planingfsi)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Docs](https://readthedocs.org/projects/planingfsi/badge/?version=latest)](https://planingfsi.readthedocs.io/en/latest/?badge=latest)
[![Version](https://img.shields.io/pypi/v/planingfsi.svg)](https://pypi.org/project/planingfsi/)
[![License](https://img.shields.io/pypi/l/planingfsi.svg)](https://pypi.org/project/planingfsi/)

PlaningFSI is a scientific Python program use to calculate the steady-state response of two-dimensional marine structures planing at constant speed on the free surface with consideration for Fluid-Structure Interaction (FSI) and rigid body motion.
It was originally written in 2012-2013 to support my Ph.D. research and has recently (2018) been updated and released as open-source.

## Cautionary Note

I am currently working on releasing this package as open source.
Since this is my first open-source release, the next few releases on PyPI should not be used for production.
I will release version 1.0.0 and remove this note once I feel that I have sufficiently cleaned up and documented the code.

## Required Python version

The code is written in Python and was originally written in Python 2.6.5.
it has since been updated to require Python 3.6+.

## Installation

PlaningFSI can be installed with pip:

```
pip install planingfsi
```

## Contributing

To contribute, you should install the code in developer mode.

```
poetry install --develop=.
```

## Getting Started

The main command-line interface is called `planingFSI` and can be called directly, once appropriate input files have been prepared.
A collection of examples can be found in the `tutorials` directory in the source package.
