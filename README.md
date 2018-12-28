# PlaningFSI

[![image](https://img.shields.io/pypi/v/planingfsi.svg)](https://pypi.org/project/planingfsi/)
[![image](https://img.shields.io/pypi/l/planingfsi.svg)](https://pypi.org/project/planingfsi/)

## Cautionary Note

I am currently working on releasing this package as open source.
Since this is my first open-source release, the next few releases on PyPi should not be used for production.
I will release version 1.0.0 once I feel that I have sufficiently cleaned up and documented the code. 

## Summary

PlaningFSI is a scientific Python program use to calculate the steady-state response of 2-d marine structures subject to planing flow with consideration for Fluid-Structure Interaction (FSI) as well as rigid body motion.
It was originally written in 2012-2013 to support my Ph.D. research and has recently (2018) been updated and released as open source.

## Required Python version

The code is written in Python and was originally written in Python 2.6.5.
it has since been updated to require Python 3.6+.
Although future versions of Python **should** support, the code, I can make no guarantees.

The code requires several Python modules, which are imported at the top of each program file. All of the modules should be included with the standard installation of Python except for the following:
- Numpy (provides support for array structures and other tools for numerical analysis)
- Scipy (provides support for many standard scientific functions, including interpolation)
- Matplotlib (provides support for plotting)


## Installation

PlaningFSI can be installed with pip:

```
pip insall planingfsi
```
