"""The `femesh` module handles mesh generation for planingFSI cases."""
from __future__ import annotations

import abc
import itertools
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from planingfsi import logger
from planingfsi import trig
from planingfsi.solver import fzero
from planingfsi.writers import write_as_list


class Mesh:
    """A high-level class to store a structural mesh.

    The mesh serves as a container for `Point` instances. It then serves as a container for
    any number of `Submesh` instances, which each consists of a series of `Curve`s and can
    optionally be associated with a planing surface.

    Attributes:
        points: A list of all points in the mesh.
        submesh: A list of submeshes belonging to the parent mesh.

    """

    def __init__(self) -> None:
        self.points: list["Point"] = []
        self.submesh: list["Subcomponent"] = []
        self.add_point(0, "dir", [0, 0])

    @property
    def curves(self) -> list["Curve"]:
        """A list of all component curves from all submeshes."""
        return list(itertools.chain(*(submesh.curves for submesh in self.submesh)))

    def get_point(self, pt_id: int, /) -> "Point":
        """Return a point by ID from the mesh."""
        # TODO: Potentially replace this with a dictionary lookup
        for pt in self.points:
            if pt.id == pt_id:
                return pt

        raise ValueError(f"Cannot find Point object with ID={pt_id}")

    """A method alias, kept for backwards compatibility with old meshDict files."""
    get_pt_by_id = get_point

    def add_submesh(self, name: str = "") -> "Subcomponent":
        """Add a submesh to the mesh."""
        submesh = Subcomponent(name, mesh=self)
        self.submesh.append(submesh)
        return submesh

    def add_point(
        self, id_: int, method: str, position: Iterable[float | int | str], **kwargs: Any
    ) -> "Point":
        """Add a new point to the mesh, returning the created point.

        Args:
            id_: The point ID.
            method: A method by which to specify the point. Depending on the method, the position
                argument is treated differently.

                Options are:
                    * "dir" - Direct coordinate specification
                    * "rel" - Relative to another point, using polar coordinates
                    * "con" - Constrained along a specific direction up to a certain distance or intersection
                    * "pct" - At a specific percentage of the distance between two other points
            position: An iterable of parameters, used for specifying the position.
                The specific meaning of the different parameters differs based on the selected method.
        Returns:
            The point that is created.

        """
        # TODO: Split method into several

        if method == "dir":
            # Direct coordinate specification
            point = Point(id=id_, mesh=self)
            self.points.append(point)
            point.position = np.array(position)
        elif method == "rel":
            # Relative coordinate specification using polar coordinates
            point = Point(id=id_, mesh=self)
            self.points.append(point)
            base_pt_id, ang, radius = position

            # TODO: Consider removing this and fixing static types
            assert isinstance(radius, (float, int)), type(radius)
            assert isinstance(ang, (float, int)), type(ang)

            point.position = self.get_point(int(base_pt_id)).position + radius * trig.angd2vec2d(
                float(ang)
            )
        elif method == "con":
            # Constrained coordinate specification
            # Either extrapolating at an angle or going horizontally or vertically
            point = Point(id=id_, mesh=self)
            self.points.append(point)
            base_pt_id, dim, val = position
            ang = kwargs.get("angle", 0.0 if dim == "x" else 90.0)

            base_pt = self.get_point(int(base_pt_id)).position
            if dim == "x":
                point.position = np.array(
                    [val, base_pt[1] + (val - base_pt[0]) * trig.tand(float(ang))]
                )
            elif dim == "y":
                point.position = np.array(
                    [base_pt[0] + (val - base_pt[1]) / trig.tand(float(ang)), val]
                )
            else:
                raise ValueError("Incorrect dimension specification")
        elif method == "pct":
            # Place a point at a certain percentage along the line between two other points
            base_pt_id, end_pt_id, pct = position
            curve = Curve(
                self.get_point(int(base_pt_id)),
                self.get_point(int(end_pt_id)),
            )
            point = self.add_point_along_curve(id_, curve=curve, pct=float(pct))
        else:
            raise NameError(f"Incorrect position specification method for point, ID: {id_}")

        return point

    def add_point_along_curve(self, id_: int, curve: "Curve", pct: float) -> "Point":
        """Add a point at a certain percentage along a curve."""
        point = Point(id=id_, mesh=self)
        self.points.append(point)
        point.position = curve.get_coords(pct)
        return point

    def add_load(self, pt_id: int, load: np.ndarray) -> None:
        """Add a fixed load at a specific point.

        Args:
            pt_id: The ID of the point at which to add the load.
            load: A 2d vector load to apply at the point.

        """
        self.get_point(pt_id).add_fixed_load(load)

    def fix_points(self, pt_id_list: Iterable[int]) -> None:
        """Fix the position of a list of points in the mesh."""
        for pt in [self.get_point(pt_id) for pt_id in pt_id_list]:
            pt.set_fixed_dof("x", "y")

    def fix_all_points(self) -> None:
        """Fix the position of all points in the mesh."""
        self.fix_points([p.id for p in self.points if p.id is not None])

    def rotate_points(self, base_pt_id: int, angle: float, pt_id_list: Iterable[int]) -> None:
        """Rotate all points in a list by a given angle about a base point, counter-clockwise."""
        base_pt = self.get_point(base_pt_id)
        for pt in [self.get_point(pt_id) for pt_id in pt_id_list]:
            pt.rotate(base_pt, angle)

    def rotate_all_points(self, base_pt_id: int, angle: float) -> None:
        """Rotate all points in the mesh by a given angle about a base point, counter-clockwise."""
        self.rotate_points(base_pt_id, angle, [p.id for p in self.points if p.id is not None])

    def move_points(self, dx: float, dy: float, pt_id_list: Iterable[int]) -> None:
        """Move (translate) a selection of points in the x & y directions."""
        for pt in [self.get_point(pt_id) for pt_id in pt_id_list]:
            pt.move(dx, dy)

    def move_all_points(self, dx: float, dy: float) -> None:
        """Move (translate) all points in the x & y directions."""
        self.move_points(dx, dy, [p.id for p in self.points if p.id is not None])

    def scale_all_points(self, sf: float, base_pt_id: int = 0) -> None:
        """Scale the coordinates of all points in the mesh by a constant scaling factor, relative to some base point.

        Args:
            sf: The scale factor.
            base_pt_id: The point from which coordinates are taken to be relative. Defaults to the origin.

        """
        base_pt = self.get_point(base_pt_id).position
        for pt in self.points:
            pt.position = (pt.position - base_pt) * sf + base_pt

    def get_diff(self, pt0: int, pt1: int) -> np.ndarray:
        """The vector pointing from one point to another.

        Args:
            pt0: The source point.
            pt1: The destination point.

        Returns:
            A two-dimensional vector pointing from source to destination.

        """
        return self.get_point(pt1).position - self.get_point(pt0).position

    def get_length(self, pt0: int, pt1: int) -> float:
        """The Cartesian distance of the line between two points."""
        return float(np.linalg.norm(self.get_diff(pt0, pt1)))

    def display(self, disp: bool = False) -> None:
        """Display information about the mesh.

        Args:
            disp: If True, print to console. Otherwise, do nothing.

        """
        if not disp:
            return

        for pt in self.points:
            pt.display()

        logger.info("Curve count: {0}".format(len(self.curves)))
        logger.info("Point count: {0}".format(len(self.points)))

    def plot(self, show: bool = False, save: bool = False, *, filename: str = "meshLayout") -> None:
        """Plot the mesh if we elect to show or save it. Otherwise, do nothing.

        Args:
            show: If True, the plot window will open.
            save: If True, the plot will be saved to a file.
            filename: A filename which can be used when saving the mesh image.

        """
        if not (show or save):
            return

        plt.figure()
        plt.axis("equal")
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")

        for curve in self.curves:
            curve.plot()

        for point in self.points:
            point.plot()

        lims = plt.gca().get_xlim()
        ext = (lims[1] - lims[0]) * 0.1
        plt.xlim([lims[0] - ext, lims[1] + ext])

        # Process optional arguments and save or show figure
        if save:
            plt.savefig(f"{filename}.eps", format="eps")
        if show:
            plt.show()  # pragma: no cover

    def write(self, mesh_dir: Path | str = Path("mesh")) -> None:
        """Write the mesh to text files."""
        mesh_dir = Path(mesh_dir)
        mesh_dir.mkdir(exist_ok=True)

        x, y = list(zip(*(pt.position for pt in self.points)))
        write_as_list(mesh_dir / "nodes.txt", ["x", x], ["y", y])

        x, y = list(zip(*(pt.is_dof_fixed for pt in self.points)))
        write_as_list(
            mesh_dir / "fixedDOF.txt",
            ["x", x],
            ["y", y],
            header_format=">1",
            data_format=">1",
        )

        x, y = list(zip(*(pt.fixed_load for pt in self.points)))
        write_as_list(
            mesh_dir / "fixedLoad.txt",
            ["x", x],
            ["y", y],
            header_format=">6",
            data_format="6.4e",
        )

        for sm in self.submesh:
            sm.write(mesh_dir)


class Subcomponent:
    """A child component of the mesh used for splitting up different sets of curves.

    The `Subcomponent` will store a set of curves, as well as a set of line segments once the geometry
    is discretized.

    Attributes:
        name: The name of the `Subcomponent`, which must correspond to the name of the associated `Substructure`.
        mesh: A reference to the parent `Mesh` object, to which this `Subcomponent` belongs.
        curves: A list of control `Curve`s that define the `Subcomponent`.
        line_segments: A list of line segments as a result of the curve being discretized.

    """

    def __init__(self, name: str, mesh: Mesh):
        self.name = name
        self.mesh = mesh
        self.curves: list["Curve"] = []
        self.line_segments: list[Curve] = []

    def add_curve(self, pt_id1: int, pt_id2: int, **kwargs: Any) -> "Curve":
        """Add a curve to the submesh.

        Args:
            pt_id1: The starting point ID.
            pt_id2: The end point ID.

        Keyword Args:
            ID: An explicit ID to use for the curve.
            arcLen: The arclength of the curve (if circular).
            radius: The radius of the curve (if circular).
            Nel: The number of line segments to divide the curve into.

        Returns:
            The Curve object.

        """
        curve = Curve(
            self.mesh.get_point(pt_id1),
            self.mesh.get_point(pt_id2),
            mesh=self.mesh,
            id=kwargs.get("ID"),
            arc_length=kwargs.get("arcLen"),
            radius=kwargs.get("radius"),
            num_elements=kwargs.get("Nel", 1),
        )
        self.curves.append(curve)

        points, line_segments = curve.distribute_points()
        self.mesh.points.extend(points)
        self.line_segments.extend(line_segments)

        return curve

    def write(self, mesh_dir: Path) -> None:
        """Write the submesh to file.

        The submesh is stored as a list of Point IDs for each line segment in the submesh.

        """
        if not self.line_segments:
            return  # pragma: no cover

        point_indices = []
        for line in self.line_segments:
            point_indices.append([line.start_point.index, line.end_point.index])

        pt_l, pt_r = list(zip(*point_indices))
        write_as_list(
            mesh_dir / f"elements_{self.name}.txt",
            ["ptL", pt_l],
            ["ptR", pt_r],
            header_format="<4",
            data_format=">4",
        )


class _ShapeBase:
    """An abstract base class for all Shapes."""

    def __init__(self, id: int | None = None, mesh: Mesh | None = None) -> None:
        self.id = id
        self._mesh = mesh

    @abc.abstractmethod
    def plot(self) -> None:
        raise NotImplementedError


class Point(_ShapeBase):
    """A general point in 2D space, used to represent control points and location of FE nodes.

    Attributes:
        position: A 2D array containing the (x, y) coordinates.
        is_dof_fixed: A length-2 list corresponding to True/False of that DOF if fixed.
        fixed_load: A 2D array containing the externally-applied forces in (x, y) directions.

    """

    def __init__(self, id: int | None = None, mesh: Mesh | None = None) -> None:
        super().__init__(id=id, mesh=mesh)
        self.position = np.zeros(2)
        self.is_dof_fixed = [True, True]
        self.fixed_load = np.zeros(2)
        self._is_used = False

    @property
    def index(self) -> int:
        """The index of the point within the mesh point list."""
        if self._mesh is None:
            raise ValueError("Point is not associated with a mesh")
        return self._mesh.points.index(self)

    @property
    def is_used(self) -> bool:
        """If True, the point will be used in the solution and is free to move."""
        return self._is_used

    @is_used.setter
    def is_used(self, value: bool) -> None:
        self._is_used = value
        if value:
            self.set_free_dof("x", "y")
        else:
            self.set_fixed_dof("x", "y")

    @property
    def x_pos(self) -> float:
        """The x-coordinate of the point."""
        return self.position[0]

    @property
    def y_pos(self) -> float:
        """The y-coordinate of the point."""
        return self.position[1]

    def add_fixed_load(self, load: np.ndarray) -> None:
        """Add a fixed load to the point."""
        self.fixed_load += load

    def set_free_dof(self, *args: str) -> None:
        """Set specific degrees of freedom to be free.

        Args:
            *args: Can be "x" and/or "y".

        """
        for arg in args:
            if arg == "x":
                self.is_dof_fixed[0] = False
            if arg == "y":
                self.is_dof_fixed[1] = False

    def set_fixed_dof(self, *args: str) -> None:
        """Set specific degrees of freedom to be fixed.

        Args:
            *args: Can be "x" and/or "y".

        """
        for arg in args:
            if arg == "x":
                self.is_dof_fixed[0] = True
            if arg == "y":
                self.is_dof_fixed[1] = True

    def move(self, dx: float, dy: float) -> None:
        self.position += np.array([dx, dy])

    def get_x_pos(self) -> float:
        # Kept for backwards-compatibility with old meshDicts
        return self.x_pos

    def get_y_pos(self) -> float:
        # Kept for backwards-compatibility with old meshDicts
        return self.y_pos

    def rotate(self, base_pt: Point | int, angle: float) -> None:
        """Rotate this point about another base point.

        Args:
            base_pt: The point around which to rotate.
            angle: The angle to rotate, in degrees, positive counter-clockwise.

        """
        if isinstance(base_pt, int):
            if self._mesh is None:
                raise LookupError(
                    "Only points existing in the mesh with an ID can be rotated by ID."
                )
            base_pt = self._mesh.get_point(base_pt)
        self.position = (
            trig.rotate_vec_2d(self.position - base_pt.position, angle) + base_pt.position
        )

    def display(self) -> None:
        logger.info(
            " ".join(
                [
                    f"{self.__class__.__name__} {self.index}",
                    f"ID = {self.id}, Pos = {self.position}",
                ]
            )
        )

    def plot(self) -> None:
        """Plot the point.

        If it is a control point with an ID, it is a circle with a label. Otherwise, it's a star.

        """
        if self.id is None:
            plt.plot(self.x_pos, self.y_pos, "r*")
        else:
            plt.plot(self.x_pos, self.y_pos, "ro")
            plt.text(self.x_pos, self.y_pos, f" {self.id}")


class Curve(_ShapeBase):
    """A curve connecting two `Point`s.

    The curve is characterized by its curvature, which is zero by default (a straight line).

    Attributes:
        id: A unique ID for the curve.
        start_point: The starting `Point` of the curve.
        end_point: The end `Point` of the curve.
        curvature: The curvature of the curve, i.e. the inverse of the radius. A value of zero is a straight line.
        num_elements: The number of elements to split the curve into during mesh generation.

    """

    def __init__(
        self,
        start_point: Point,
        end_point: Point,
        *,
        id: int | None = None,
        mesh: Mesh | None = None,
        curvature: float | None = None,
        arc_length: float | None = None,
        radius: float | None = None,
        num_elements: int = 1,
    ):
        super().__init__(id=id, mesh=mesh)
        self.start_point = start_point
        self.end_point = end_point
        self.num_elements = num_elements

        if sum(item is not None for item in (curvature, arc_length, radius)) > 1:
            raise ValueError("Can only set one of 'curvature', 'arc_length', or 'radius'")

        self.curvature = 0.0
        if curvature is not None:
            self.curvature = curvature
        elif arc_length is not None:
            self.arc_length = arc_length
        elif radius is not None:
            self.radius = radius

    @property
    def chord(self) -> float:
        """The chord length, i.e. the distance between start and end points."""
        return float(np.linalg.norm(self.end_point.position - self.start_point.position))

    @property
    def radius(self) -> float:
        """The radius of the curve. A flat curve has an infinite radius."""
        return 1 / self.curvature if self.curvature != 0 else np.inf

    @radius.setter
    def radius(self, value: float) -> None:
        self.curvature = 1 / value if ~np.isinf(value) else 0.0

    def _get_arc_length_residual(self, curvature: float, arc_length: float) -> float:
        """Return the trigonometric residual for the relationship between curvature and arclength."""
        return 0.5 * curvature * self.chord - np.sin(0.5 * curvature * arc_length)

    @property
    def arc_length(self) -> float:
        """The arclength of the curve."""
        if self.curvature == 0:
            return self.chord
        else:

            return fzero(
                lambda s: self._get_arc_length_residual(curvature=self.curvature, arc_length=s),
                self.chord + 1e-6,
            )

    @arc_length.setter
    def arc_length(self, value: float) -> None:
        if self.chord >= value:
            self.curvature = 0.0
        else:
            # Keep increasing guess until fsolve finds the first non-zero root.
            # The criterion here is 1e-6, because the zero case is already handled above
            # and if we set kap = 0, then the residual is always zero, but that case is
            # singular.
            kap0 = 0.02
            while (
                kap := fzero(
                    lambda x: self._get_arc_length_residual(curvature=x, arc_length=value), kap0
                )
            ) <= 1e-6:
                kap0 += 0.02

            self.curvature = kap

    def get_coords(self, s: float) -> np.ndarray:
        """Get the coordinates of any point along the curve.

        Args:
            s: The position along the curve (as a fraction of total arclength).

        Returns:
            The (x,y) coordinates of the point, as a numpy array.

        """
        if self.curvature == 0.0:
            return self.start_point.position * (1 - s) + self.end_point.position * s
        else:
            diff = self.end_point.position - self.start_point.position
            gam = np.arctan2(diff[1], diff[0])
            alf = self.arc_length / (2 * self.radius)
            return (
                self.start_point.position
                + 2.0 * self.radius * np.sin(s * alf) * trig.ang2vec(gam + (s - 1.0) * alf)[:2]
            )

    def distribute_points(self) -> tuple[list[Point], list[Curve]]:
        """Distribute points along the curve during mesh generation.

        Returns:
             A list of `Point`s and straight `Curve` objects connecting them.

        """
        points = [self.start_point]
        if self.num_elements > 1:
            # Distribute N points along a parametric curve defined by f(s), s in [0,1]
            for s in np.linspace(0.0, 1.0, self.num_elements + 1)[1:-1]:
                point = Point(mesh=self._mesh)
                point.is_used = True
                point.position = self.get_coords(s)
                points.append(point)
        points.append(self.end_point)

        lines = []
        for ptSt, ptEnd in zip(points[:-1], points[1:]):
            line = Curve(ptSt, ptEnd, mesh=self._mesh)
            lines.append(line)
        return points, lines

    def plot(self) -> None:
        """Plot the curve as a line by chaining all component points together."""
        points, _ = self.distribute_points()
        x, y = list(zip(*(pt.position for pt in points)))
        plt.plot(x, y, "k-")
