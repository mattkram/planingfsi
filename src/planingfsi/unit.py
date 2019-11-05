"""A simple module for storing unit conversion factors to convert to SI.

Example: A length e.g. can be represented as 1.0 * unit.ft to represent one foot,
    which will return 0.3048 meters.

"""
import math

# Acceleration due to gravity
_gravity = 9.80665

# Convert length to m
m = 1.0
cm = m / 100.0
mm = m / 1000.0
ft = 0.3048
inch = ft / 12.0

# Convert speed to m/s
mps = 1.0
fps = ft
kts = 0.5144444

# Convert mass to kg
kg = 1.0
lbm = kg / 2.20462

# Convert force to N
N = 1.0
lb = _gravity * lbm
lbf = 1.0 * lb  # lbs-force
mtf = _gravity * 1000.0  # metric tonnes-force
stf = N * lb * 2000.0

rad = 1.0
deg = math.pi / 180.0
