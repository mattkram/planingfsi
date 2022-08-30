# Geometry and Meshing

## Origin & coordinate system

`planingFSI` uses a two-dimensional Cartesian coordinate system, with the primary coordinates being `x` and `y`, positive to the right and up, respectively.
The fluid is assumed to flow in the `-x` direction (equivalently, the body moves in the `+x` direction at steady speed).

There is a point with ID `0` located at the origin, `(0.0, 0.0)`.

## Units

`planingFSI` does not consider units.
Thus, for all coordinates and input values, one must be careful to maintain a consistent set of units.
To do so, it is recommended to use SI units (`kg`, `m`, `s`) or English units (`slug`, `ft`, `s`) for (`mass`, `length`, `time`).

## Creating a mesh

Specification of the computational mesh begins with the instantiation of a `Mesh` object:

```python
from planingfsi import Mesh

mesh = Mesh()
```

The geometry may then be specified, first by adding control points.
These control points are connected with `Curve`s or `Line`s.

There are four methods for adding `Point` objects to the `Mesh`.

### Direct coordinate specification

Points can be placed in the mesh with direct coordinate specification.
The form of this is as follows, where `1` is the new point's ID:

```python
from planingfsi import Mesh

mesh = Mesh()
mesh.add_point(1, "dir", [1.0, 1.0])
```

### Relative coordinate specification

Similarly, points can be placed relative to another existing point.
In this case, the `position` argument is an iterable of `(base point id, angle, radius)`
For example, the following places a point at an angle 45 degrees counter-clockwise from the origin at a distance of 10:

```python
from planingfsi import Mesh

mesh = Mesh()
mesh.add_point(1, "rel", [0, 45.0, 10.0])
```

### Constraining points to a specific `x`- or `y`-coordinate

Sometimes, it is useful to place points relative to another at an unknown distance, where the distance is calculated such that a specific `x`- or `y`-coordinate is achieved.
For example, the code below will place a point at an angle of 45 degrees counter-clockwise from the origin, but the `x`-coordinate will be 10.0, rather than the distance (~14.14).
In this method, the `position` argument is an iterable of `(base point id, constrained coordinate (x or y), value)` and an optional `angle` keyword argument may be provided.
The `angle` defaults to `0.0` if `x` is selected, and `90.0` if `y` is selected as the direction.

```python
from planingfsi import Mesh

mesh = Mesh()
mesh.add_point(1, "con", [0, "x", 10.0], angle=45.0)

```

### Placing points along a line between two other points

Points can be placed along the line between two existing points.
In this case, the `position` argument is an iterable of `(first point id, second point id, fractional distance)`.
For example, the following will place a point 3/4 of the way between two points (at `(7.5, 7.5)`):

```python
from planingfsi import Mesh

mesh = Mesh()
mesh.add_point(1, "dir", [10.0, 10.0])  # Second point
mesh.add_point(2, "pct", [0, 1, 0.75])  # First point is the origin
```
