import abc
import os
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from .. import logger
from .. import trig
from ..solver import fzero
from ..writers import write_as_list


class Mesh:
    """A high-level class to store a structural mesh.

    The mesh serves as a container for `Point` instances. It then serves as a container for
    any number of `Submesh` instances, which each consists of a series of `Curve`s and can
    optionally be associated with a planing surface.

    Attributes:
        mesh_dir: The directory in which to store the mesh files on export.
        submesh: A list of submeshes belonging to the parent mesh.

    """

    def __init__(self, mesh_dir: Union[Path, str] = "mesh") -> None:
        self.mesh_dir = Path(mesh_dir)
        self.points: List["Point"] = []
        self.submesh: List["Submesh"] = []
        self.add_point(0, "dir", [0, 0])

    def get_point(self, pt_id: int, /) -> "Point":
        """Return a point by ID from the mesh."""
        return Point.find_by_id(pt_id)

    def add_submesh(self, name: str = "") -> "Submesh":
        """Add a submesh to the mesh."""
        submesh = Submesh(name)
        self.submesh.append(submesh)
        return submesh

    def add_point(
        self, id_: int, method: str, position: Iterable[Union[float, int, str]], **kwargs: Any
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

        Returns:
            The point that is created.

        """
        # TODO: Split method into several
        point = Point(id=id_)
        self.points.append(point)

        if method == "dir":
            # Direct coordinate specification
            point.set_position(np.array(position))
        elif method == "rel":
            # Relative coordinate specification using polar coordinates
            base_pt_id, ang, radius = position
            point.set_position(
                Point.find_by_id(int(base_pt_id)).get_position()
                + radius * trig.angd2vec2d(float(ang))
            )
        elif method == "con":
            # Constrained coordinate specification
            # Either extrapolating at an angle or going horizontally or vertically
            base_pt_id, dim, val = position
            ang = kwargs.get("angle", 0.0 if dim == "x" else 90.0)

            base_pt = Point.find_by_id(int(base_pt_id)).get_position()
            if dim == "x":
                point.set_position(
                    np.array([val, base_pt[1] + (val - base_pt[0]) * trig.tand(float(ang))])
                )
            elif dim == "y":
                point.set_position(
                    np.array([base_pt[0] + (val - base_pt[1]) / trig.tand(float(ang)), val])
                )
            else:
                raise ValueError("Incorrect dimension specification")
        elif method == "pct":
            # Place a point at a certain percentage along the line between two other points
            base_pt_id, end_pt_id, pct = position
            base_pt = Point.find_by_id(int(base_pt_id)).get_position()
            end_pt = Point.find_by_id(int(end_pt_id)).get_position()
            point.set_position((1.0 - float(pct)) * base_pt + float(pct) * end_pt)
        else:
            raise NameError(f"Incorrect position specification method for point, ID: {id_}")

        return point

    def add_point_along_curve(self, id_: int, curve: "Curve", pct: float) -> "Point":
        """Add a point at a certain percentage along a curve."""
        point = Point(id=id_)
        self.points.append(point)
        point.set_position(curve.get_shape_func()(pct))
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
        for pt in [Point.find_by_id(pt_id) for pt_id in pt_id_list]:
            pt.set_fixed_dof("x", "y")

    def fix_all_points(self) -> None:
        """Fix the position of all points in the mesh."""
        self.fix_points([p.ind for p in Point.all()])

    def rotate_points(self, base_pt_id: int, angle: float, pt_id_list: Iterable[int]) -> None:
        """Rotate all points in a list by a given angle about a base point, counter-clockwise."""
        for pt in [Point.find_by_id(pt_id) for pt_id in pt_id_list]:
            pt.rotate(base_pt_id, angle)

    def rotate_all_points(self, base_pt_id: int, angle: float) -> None:
        self.rotate_points(base_pt_id, angle, [p.ind for p in Point.all()])

    def move_points(self, dx: float, dy: float, pt_id_list: Iterable[int]) -> None:
        for pt in [Point.find_by_id(pt_id) for pt_id in pt_id_list]:
            pt.move(dx, dy)

    def move_all_points(self, dx: float, dy: float) -> None:
        self.move_points(dx, dy, [p.ind for p in Point.all()])

    def scale_all_points(self, sf: float, base_pt_id: int = 0) -> None:
        base_pt = Point.find_by_id(base_pt_id).get_position()
        for pt in Point.all():
            pt.set_position((pt.get_position() - base_pt) * sf + base_pt)

    def get_diff(self, pt0: int, pt1: int) -> np.ndarray:
        return Point.find_by_id(pt1).get_position() - Point.find_by_id(pt0).get_position()

    def get_length(self, pt0: int, pt1: int) -> float:
        return np.linalg.norm(self.get_diff(pt0, pt1))

    def display(self, disp: bool = False) -> None:
        """Display information about the mesh.

        Args:
            disp: If True, print to console. Otherwise, do nothing.

        """
        if not disp:
            return
        Shape.print_all()
        logger.info("Line count:  {0}".format(Line.count()))
        logger.info("Point count: {0}".format(Point.count()))

    def plot(self, **kwargs: Any) -> None:
        show = kwargs.get("show", False)
        save = kwargs.get("save", False)
        plot = show or save
        if plot:
            plt.figure(figsize=(16, 14))
            plt.axis("equal")
            plt.xlabel(r"$x$", size=18)
            plt.ylabel(r"$y$", size=18)

            Shape.plot_all()

            lims = plt.gca().get_xlim()
            ext = (lims[1] - lims[0]) * 0.1
            plt.xlim([lims[0] - ext, lims[1] + ext])

            # Process optional arguments and save or show figure
            if save:
                saved_file_name = kwargs.get("fileName", "meshLayout")
                plt.savefig(saved_file_name + ".eps", format="eps")
            if show:
                plt.show()

    def write(self) -> None:
        """Write the mesh to text files."""
        Path(self.mesh_dir).mkdir(exist_ok=True)
        x, y = list(zip(*[pt.get_position() for pt in Point.all()]))
        write_as_list(os.path.join(self.mesh_dir, "nodes.txt"), ["x", x], ["y", y])

        x, y = list(zip(*[pt.get_fixed_dof() for pt in Point.all()]))
        write_as_list(
            os.path.join(self.mesh_dir, "fixedDOF.txt"),
            ["x", x],
            ["y", y],
            header_format=">1",
            data_format=">1",
        )

        x, y = list(zip(*[pt.get_fixed_load() for pt in Point.all()]))
        write_as_list(
            os.path.join(self.mesh_dir, "fixedLoad.txt"),
            ["x", x],
            ["y", y],
            header_format=">6",
            data_format="6.4e",
        )

        for sm in self.submesh:
            sm.write()


class Submesh(Mesh):
    def __init__(self, name: str, **kwargs: Any):
        Mesh.__init__(self, **kwargs)
        self.name = name
        self.line: List["Line"] = []

    def add_curve(self, pt_id1: int, pt_id2: int, **kwargs: Any) -> "Curve":
        """Add a curve to the submesh."""
        arc_length = kwargs.get("arcLen")
        radius = kwargs.get("radius")

        curve = Curve(kwargs.get("Nel", 1))
        curve.ID = kwargs.get("ID", -1)
        curve.set_end_pts_by_id(pt_id1, pt_id2)

        if arc_length is not None:
            curve.set_arc_length(arc_length)
        elif radius is not None:
            curve.set_radius(radius)
        else:
            curve.set_arc_length(0.0)

        curve.distribute_points()
        for pt in curve.pt:
            pt.set_used()

        self.line += [line for line in curve.get_lines()]

        return curve

    def write(self) -> None:
        """Write the submesh to file."""
        if self.line:
            pt_l, pt_r = list(
                zip(*[[pt.ind for pt in line.get_element_coords()] for line in self.line])
            )
            write_as_list(
                os.path.join(self.mesh_dir, "elements_{0}.txt".format(self.name)),
                ["ptL", pt_l],
                ["ptR", pt_r],
                header_format="<4",
                data_format=">4",
            )


T = TypeVar("T", bound="Shape")


class Shape:
    __all: List["Shape"] = []

    @classmethod
    def all(cls: Type[T]) -> List[T]:
        all_return: List[T] = []
        for o in cls.__all:
            if isinstance(o, cls):
                all_return.append(o)
        return all_return

    @classmethod
    def count(cls) -> int:
        return len(cls.all())

    @classmethod
    def plot_all(cls) -> None:
        for o in cls.__all:
            o.plot()

    @classmethod
    def print_all(cls) -> None:
        for o in cls.__all:
            o.display()

    @classmethod
    def find_by_id(cls: Type[T], id_: int) -> T:
        if id_ is not None:
            for a in cls.all():
                if a.ID == id_:
                    return a
        raise ValueError(f"Cannot find {cls.__name__} object with ID={id_}")

    def __init__(self, id: Optional[int] = None) -> None:
        self.ind = self.count()
        self.ID: Optional[int] = id
        Shape.__all.append(self)

    @property
    def ID(self) -> Optional[int]:
        return self._id

    @ID.setter
    def ID(self, value: Optional[int]) -> None:
        # TODO: Not sure why we need this
        if value is not None:
            try:
                existing = self.find_by_id(value)
            except ValueError:
                pass
            else:
                existing.ID = None
        self._id = value

    @abc.abstractmethod
    def display(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def plot(self) -> None:
        raise NotImplementedError


class Point(Shape):
    __all: List["Point"] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        Point.__all.append(self)

        self.pos = np.zeros(2)
        self.is_dof_fixed = [True, True]
        self.fixed_load = np.zeros(2)
        self.is_used = False

    def set_used(self) -> None:
        self.is_used = True
        self.set_free_dof("x", "y")

    def get_fixed_load(self) -> np.ndarray:
        return self.fixed_load

    def add_fixed_load(self, load: np.ndarray) -> None:
        """Add a fixed load to the point."""
        self.fixed_load += load

    def set_free_dof(self, *args: str) -> None:
        for arg in args:
            if arg == "x":
                self.is_dof_fixed[0] = False
            if arg == "y":
                self.is_dof_fixed[1] = False

    def set_fixed_dof(self, *args: str) -> None:
        for arg in args:
            if arg == "x":
                self.is_dof_fixed[0] = True
            if arg == "y":
                self.is_dof_fixed[1] = True

    def get_fixed_dof(self) -> List[bool]:
        return self.is_dof_fixed

    def move(self, dx: float, dy: float) -> None:
        self.set_position(self.get_position() + np.array([dx, dy]))

    def set_position(self, pos: np.ndarray) -> None:
        self.pos = pos

    def get_position(self) -> np.ndarray:
        return self.pos

    def get_x_pos(self) -> float:
        return self.pos[0]

    def get_y_pos(self) -> float:
        return self.pos[1]

    def rotate(self, base_pt_id: int, angle: float) -> None:
        base_pt = Point.find_by_id(base_pt_id).get_position()
        self.set_position(trig.rotate_vec_2d(self.get_position() - base_pt, angle) + base_pt)

    def display(self) -> None:
        logger.info(
            " ".join(
                [
                    f"{self.__class__.__name__} {self.ind}",
                    f"ID = {self.ID}, Pos = {self.get_position()}",
                ]
            )
        )

    def plot(self) -> None:
        if self.ID is None:
            plt.plot(self.pos[0], self.pos[1], "r*")
        else:
            plt.plot(self.pos[0], self.pos[1], "ro")
            plt.text(self.pos[0], self.pos[1], " {0}".format(self.ID))


class Curve(Shape):
    __all: List["Curve"] = []

    def __init__(self, Nel: int = 1, **kwargs: Any):
        super().__init__(**kwargs)
        Curve.__all.append(self)
        self.pt: List[Point] = []
        self.line: List["Line"] = []
        self._end_pts: List[Point] = []
        self.Nel = Nel
        self.plot_sty = "m--"

        self.radius = 0.0
        self.arc_length = 0.0
        self.chord = 0.0

    def set_end_pts_by_id(self, pt_id1: int, pt_id2: int) -> None:
        self.set_end_pts([Point.find_by_id(pid) for pid in [pt_id1, pt_id2]])

    def get_shape_func(self) -> Callable[[float], np.ndarray]:
        xy = [pt.get_position() for pt in self._end_pts]
        assert self.radius is not None or self.arc_length is not None
        if self.radius == 0.0:
            return lambda s: xy[0] * (1 - s) + xy[1] * s
        else:
            x, y = list(zip(*xy))
            gam = np.arctan2(y[1] - y[0], x[1] - x[0])
            assert self.radius is not None
            assert self.arc_length is not None
            alf = self.arc_length / (2 * self.radius)
            assert self.radius is not None
            return (
                lambda s: self._end_pts[0].get_position()
                + 2.0 * self.radius * np.sin(s * alf) * trig.ang2vec(gam + (s - 1.0) * alf)[:2]
            )

    def set_radius(self, radius: float) -> None:
        self.radius = radius
        self.calculate_chord()
        self.calculate_arc_length()

    def set_arc_length(self, arc_length: float) -> None:
        self.calculate_chord()
        if self.chord >= arc_length:
            self.arc_length = 0.0
            self.set_radius(0.0)
        else:
            self.arc_length = arc_length
            self.calculate_radius()

    def calculate_radius(self) -> None:
        self.radius = 1 / self.calculate_curvature()

    def calculate_chord(self) -> None:
        x, y = list(zip(*[pt.get_position() for pt in self._end_pts]))
        self.chord = ((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2) ** 0.5

    def calculate_arc_length(self) -> None:
        if self.radius == 0:
            self.arc_length = self.chord
        else:

            def f(s: float) -> float:
                return self.chord / (2 * self.radius) - np.sin(s / (2 * self.radius))

            # Keep increasing guess until fsolve finds the first non-zero root
            self.arc_length = fzero(f, self.chord + 1e-6)

    def calculate_curvature(self) -> float:
        def f(x: float) -> float:
            return x * self.chord / 2 - np.sin(x * self.arc_length / 2)

        # Keep increasing guess until fsolve finds the first non-zero root
        kap = 0.0
        kap0 = 0.0
        while kap <= 1e-6:
            kap = fzero(f, kap0)
            kap0 += 0.02

        return kap

    def distribute_points(self) -> None:
        self.pt.append(self._end_pts[0])
        if not self.Nel == 1:
            # Distribute N points along a parametric curve defined by f(s), s in [0,1]
            s = np.linspace(0.0, 1.0, self.Nel + 1)[1:-1]
            for xy in map(self.get_shape_func(), s):
                P = Point()
                self.pt.append(P)
                P.set_position(xy)
        self.pt.append(self._end_pts[1])
        self.generate_lines()

    def generate_lines(self) -> None:
        for ptSt, ptEnd in zip(self.pt[:-1], self.pt[1:]):
            L = Line()
            L.set_end_pts([ptSt, ptEnd])
            self.line.append(L)

    def get_lines(self) -> List["Line"]:
        return self.line

    def set_pts(self, pt: List[Point]) -> None:
        self.pt = pt

    def set_end_pts(self, end_pt: List[Point]) -> None:
        self._end_pts = end_pt

    def get_element_coords(self) -> List[Point]:
        return self.pt

    def get_pt_ids(self) -> List[int]:
        out: List[int] = []
        for pt in self.pt:
            id_ = pt.ID
            if id_ is not None:
                out.append(id_)
        return out

    def display(self) -> None:
        logger.info(
            " ".join(
                [
                    f"{self.__class__.__name__} {self.ind}:",
                    f"ID = {self.ID}, Pt IDs = {self.get_pt_ids()}",
                ]
            )
        )

    def plot(self) -> None:
        x, y = list(zip(*[pt.get_position() for pt in self.pt]))
        plt.plot(x, y, self.plot_sty)


class Line(Curve):
    __all: List["Line"] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        Line.__all.append(self)
        self.plot_sty = "b-"

    def set_end_pts(self, end_pt: List[Point]) -> None:
        super().set_end_pts(end_pt)
        self.set_pts(end_pt)
