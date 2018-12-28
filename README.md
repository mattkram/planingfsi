# PlaningFSI #


![Bitbucket Pipelines](https://img.shields.io/bitbucket/pipelines/mattkram/planingfsi.svg)
[![image](https://img.shields.io/pypi/v/planingfsi.svg)](https://pypi.org/project/planingfsi/)
[![image](https://img.shields.io/pypi/l/planingfsi.svg)](https://pypi.org/project/planingfsi/)

## Cautionary Note

I am currently working on releasing this package as open source.
Since this is my first open-source release, the next few releases on PyPi should not be used for production.
I will release version 1.0.0 once I feel that I have sufficiently cleaned up and documented the code. 

## Summary ##

This document has been prepared as an informal instruction manual for the code that I wrote for my Ph.D. The program is currently called "rigidBodyFSI" because it is constructed as a means of calculating the steady-state response of 2-d marine structures subject to planing flow with consideration for Fluid-Strucuture Interaction (FSI) as well as rigid body motion.

## Required Software Packages ##

The code is written in Python and has been tested using Python 2.6.5. Although future versions of Python **should** support, the code, I make no guarantees.

The code requires several Python modules, which are imported at the top of each program file. All of the modules should be included with the standard installation of Python except for the following:
- Numpy (provides support for array structures and other tools for numerical analysis)
- Scipy (provides support for many standard scientific functions, including interpolation)
- Matplotlib (provides support for plotting)


## Installation ##

In the current directory, there are two directories and an install file. The install file is a shell script that creates a bashrc file. The bashrc file, once sourced, creates the shell environment variable $KCODEPATH, as well as appends the lib folder to the Python search path, and appends the bin folder to the shell search path. The install command only needs to be run once, however it can be run again if the location of the base folder changes.

Run the install file using the command "sh install". After it runs, check the end of your ~/.bashrc file, where it should have added a line to source the newly-created bashrc file in the kCode folder. In order to complete the installation, either open a new terminal or source your ~/.bashrc file by typing ". ~/.bashrc"

There are two subfolders within the main kCode folder:

The lib folder is used to store Python modules that I have written. These modules house the majority of the code, and are intended to be flexible packages that may be imported into other programs to utilize their classes and functions.

The bin folder contains the main program and some utilities that may be run from the command line. In order to run them from the command line, they must first be made executable. In order to do so, you can either do it manually or type the following command, which requires that the bashrc file has been sourced and that the $KCODEPATH variable is available. A good way to check this is to type a simple echo command, i.e. "echo $KCODEPATH". The command to make the program files executable is:

chmod +x $KCODEPATH/bin/*

Once the files are made executable, the names of the programs should be able to be autocompleted by the bash shell. Since the main program is called rigidBodyFSI, try typing rigi and then hitting tab. If everything is installed properly, it should autocomplete.
