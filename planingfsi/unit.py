"""This module is used to store factors for unit conversion.

Usage:
    import planingfsi.unit as unit
    value = 15.0 * unit.ft
"""

import math

# Convert length to m
m = 1.0
cm = m / 100.0
mm = m / 1000.0
ft = 0.3048
inch = ft / 12.0

# Convert speed to m/s
mps = 1.0
fps = ft

# Convert mass to kg
kg = 1.0
lbm = kg / 2.20462

# Convert force to N
N = 1.0
lb = 9.81 * lbm
lbf = 1.0 * lb

rad = 1.0
deg = math.pi / 180.0
