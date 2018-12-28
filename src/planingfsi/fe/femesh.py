import os

import numpy as np
import matplotlib.pyplot as plt

import planingfsi.config as config
import krampy as kp


class Mesh:
    @classmethod
    def get_pt_by_id(cls, ID):
        return Point.find_by_id(ID)

    def __init__(self):
        self.submesh = []
        self.add_point(0, "dir", [0, 0])

    def add_submesh(self, name=""):
        submesh = Submesh(name)
        self.submesh.append(submesh)
        return submesh

    def add_point(self, ID, method, position, **kwargs):
        P = Point()
        P.set_id(ID)

        if method == "dir":
            P.set_position(np.array(position))
        elif method == "rel":
            base_pt_id, ang, R = position
            P.set_position(
                Point.find_by_id(base_pt_id).get_position() + R * kp.ang2vecd(ang)
            )
        elif method == "con":
            base_pt_id, dim, val = position
            ang = kwargs.get("angle", 0 if dim == "x" else 90)

            base_pt = Point.find_by_id(base_pt_id).get_position()
            if dim == "x":
                P.set_position(
                    np.array([val, base_pt[1] + (val - base_pt[0]) * kp.tand(ang)])
                )
            elif dim == "y":
                P.set_position(
                    np.array([base_pt[0] + (val - base_pt[1]) / kp.tand(ang), val])
                )
            else:
                print("Incorrect dimension specification")
        elif method == "pct":
            base_pt_id, end_pt_id, pct = position
            base_pt = Point.find_by_id(base_pt_id).get_position()
            end_pt = Point.find_by_id(end_pt_id).get_position()
            P.set_position((1 - pct) * base_pt + pct * end_pt)
        else:
            raise NameError(
                "Incorrect position specification method for point, ID: {0}".format(ID)
            )

        return P

    def add_point_along_curve(self, ID, curve, pct):
        P = Point()
        P.set_id(ID)
        P.set_position(map(curve.get_shape_func(), [pct])[0])
        return P

    def add_load(self, pt_id, F):
        Point.find_by_id(pt_id).add_fixed_load(F)

    def fix_all_points(self):
        for pt in Point.All():
            pt.set_fixed_dof("x", "y")

    def fix_points(self, pt_id_list):
        for pt in [Point.find_by_id(pt_id) for pt_id in pt_id_list]:
            pt.set_fixed_dof("x", "y")

    def rotate_points(self, base_pt_id, angle, pt_id_list):
        for pt in [Point.find_by_id(pt_id) for pt_id in pt_id_list]:
            pt.rotate(base_pt_id, angle)

    def rotate_all_points(self, base_pt_id, angle):
        for pt in Point.All():
            pt.rotate(base_pt_id, angle)

    def move_points(self, dx, dy, pt_id_list):
        for pt in [Point.find_by_id(pt_id) for pt_id in pt_id_list]:
            pt.move(dx, dy)

    def move_all_points(self, dx, dy):
        for pt in Point.All():
            pt.move(dx, dy)

    def scale_all_points(self, sf, base_pt_id=0):
        base_pt = Point.find_by_id(base_pt_id).get_position()
        for pt in Point.All():
            pt.set_position((pt.get_position() - base_pt) * sf + base_pt)

    def get_diff(self, pt0, pt1):
        return (
            Point.find_by_id(pt1).get_position() - Point.find_by_id(pt0).get_position()
        )

    def get_length(self, pt0, pt1):
        return np.linalg.norm(self.get_diff(pt0, pt1))

    def display(self, **kwargs):
        if kwargs.get("disp", False):
            Shape.print_all()
            print(("Line count:  {0}".format(Line.count())))
            print(("Point count: {0}".format(Point.count())))

    def plot(self, **kwargs):
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

    def write(self):
        kp.createIfNotExist(config.path.mesh_dir)
        x, y = list(zip(*[pt.get_position() for pt in Point.All()]))
        kp.writeaslist(
            os.path.join(config.path.mesh_dir, "nodes.txt"), ["x", x], ["y", y]
        )

        x, y = list(zip(*[pt.get_fixed_dof() for pt in Point.All()]))
        kp.writeaslist(
            os.path.join(config.path.mesh_dir, "fixedDOF.txt"),
            ["x", x],
            ["y", y],
            headerFormat=">1",
            dataFormat=">1",
        )

        x, y = list(zip(*[pt.get_fixed_load() for pt in Point.All()]))
        kp.writeaslist(
            os.path.join(config.path.mesh_dir, "fixedLoad.txt"),
            ["x", x],
            ["y", y],
            headerFormat=">6",
            dataFormat="6.4e",
        )

        for sm in self.submesh:
            sm.write()


class Submesh(Mesh):
    def __init__(self, name):
        Mesh.__init__(self)
        self.name = name
        self.line = []

    def add_curve(self, pt_id1, pt_id2, **kwargs):
        arc_length = kwargs.get("arcLen", None)
        radius = kwargs.get("radius", None)

        C = Curve(kwargs.get("Nel", 1))
        C.set_id(kwargs.get("ID", -1))
        C.set_end_pts_by_id(pt_id1, pt_id2)

        if arc_length is not None:
            C.set_arc_length(arc_length)
        elif radius is not None:
            C.set_radius(radius)
        else:
            C.set_arc_length(0.0)

        C.distribute_points()
        for pt in C.pt:
            pt.set_used()

        self.line += [l for l in C.get_lines()]

        return C

    def write(self):
        if len(self.line) > 0:
            ptL, ptR = list(
                zip(
                    *[
                        [pt.get_index() for pt in l._get_element_coords()]
                        for l in self.line
                    ]
                )
            )
            kp.writeaslist(
                os.path.join(
                    config.path.mesh_dir, "elements_{0}.txt".format(self.name)
                ),
                ["ptL", ptL],
                ["ptR", ptR],
                headerFormat="<4",
                dataFormat=">4",
            )


class Shape:
    obj = []

    @classmethod
    def All(cls):
        return [o for o in cls.obj]

    @classmethod
    def count(cls):
        return len(cls.All())

    @classmethod
    def plot_all(cls):
        for o in cls.obj:
            o.plot()

    @classmethod
    def print_all(cls):
        for o in cls.obj:
            o.display()

    @classmethod
    def find_by_id(cls, ID):
        if ID == -1:
            return None
        else:
            obj = [a for a in cls.All() if a.ID == ID]
            if len(obj) == 0:
                return None
            else:
                return obj[0]

    def __init__(self):
        self.set_index(Shape.count())
        self.ID = -1
        Shape.obj.append(self)

    def set_id(self, ID):
        existing = self.__class__.find_by_id(ID)
        if existing is not None:
            existing.set_id(-1)
        self.ID = ID

    def set_index(self, ind):
        self.ind = ind

    def get_id(self):
        return self.ID

    def get_index(self):
        return self.ind

    def plot(self):
        return 0


class Point(Shape):
    obj = []

    def __init__(self):
        Shape.__init__(self)
        self.set_index(Point.count())
        Point.obj.append(self)

        self.pos = np.zeros(2)
        self.is_dof_fixed = [True] * 2
        self.fixed_load = np.zeros(2)
        self.is_used = False

    def set_used(self, used=True):
        self.is_used = True
        self.set_free_dof("x", "y")

    def get_fixed_load(self):
        return self.fixed_load

    def add_fixed_load(self, F):
        for i in range(2):
            self.fixed_load[i] += F[i]

    def set_free_dof(self, *args):
        for arg in args:
            if arg == "x":
                self.is_dof_fixed[0] = False
            if arg == "y":
                self.is_dof_fixed[1] = False

    def set_fixed_dof(self, *args):
        for arg in args:
            if arg == "x":
                self.is_dof_fixed[0] = True
            if arg == "y":
                self.is_dof_fixed[1] = True

    def get_fixed_dof(self):
        return self.is_dof_fixed

    def move(self, dx, dy):
        self.set_position(self.get_position() + np.array([dx, dy]))

    def set_position(self, pos):
        self.pos = pos

    def get_position(self):
        return self.pos

    def get_x_pos(self):
        return self.pos[0]

    def get_y_pos(self):
        return self.pos[1]

    def rotate(self, base_pt_id, angle):
        base_pt = Point.find_by_id(base_pt_id).get_position()
        self.set_position(kp.rotateVec(self.get_position() - base_pt, angle) + base_pt)

    def display(self):
        print(
            (
                "{0} {1}: ID = {2}, Pos = {3}".format(
                    self.__class__.__name__,
                    self.get_index(),
                    self.get_id(),
                    self.get_position(),
                )
            )
        )

    def plot(self):
        if self.ID == -1:
            plt.plot(self.pos[0], self.pos[1], "r*")
        else:
            plt.plot(self.pos[0], self.pos[1], "ro")
            plt.text(self.pos[0], self.pos[1], " {0}".format(self.ID))


class Curve(Shape):
    obj = []

    def __init__(self, Nel=1):
        Shape.__init__(self)
        self.set_index(Curve.count())
        Curve.obj.append(self)
        self.pt = []
        self.line = []
        self._end_pts = []
        self.Nel = Nel
        self.plot_sty = "m--"

    def set_end_pts_by_id(self, pt_id1, pt_id2):
        self.set_end_pts([Point.find_by_id(pid) for pid in [pt_id1, pt_id2]])

    def get_shape_func(self):
        xy = [pt.get_position() for pt in self._end_pts]
        if self.radius == 0.0:
            return lambda s: xy[0] * (1 - s) + xy[1] * s
        else:
            x, y = list(zip(*xy))
            gam = np.arctan2(y[1] - y[0], x[1] - x[0])
            alf = self.arc_length / (2 * self.radius)
            return lambda s: self._end_pts[0].get_position() + 2 * self.radius * np.sin(
                s * alf
            ) * kp.ang2vec(gam + (s - 1) * alf)

    def set_radius(self, R):
        self.radius = R
        self.calculate_chord()
        self.calculate_arc_length()

    def set_arc_length(self, arc_length):
        self.calculate_chord()
        if self.chord >= arc_length:
            self.arc_length = 0.0
            self.set_radius(0.0)
        else:
            self.arc_length = arc_length
            self.calculate_radius()

    def calculate_radius(self):
        self.radius = 1 / self.calculate_curvature()

    def calculate_chord(self):
        x, y = list(zip(*[pt.get_position() for pt in self._end_pts]))
        self.chord = ((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2) ** 0.5

    def calculate_arc_length(self):
        if self.radius == 0:
            self.arc_length = self.chord
        else:
            f = lambda s: self.chord / (2 * self.radius) - np.sin(s / (2 * self.radius))

            # Keep increasing guess until fsolve finds the first non-zero root
            self.arc_length = kp.fzero(f, self.chord + 1e-6)

    def calculate_curvature(self):
        f = lambda x: x * self.chord / 2 - np.sin(x * self.arc_length / 2)

        # Keep increasing guess until fsolve finds the first non-zero root
        kap = 0.0
        kap0 = 0.0
        while kap <= 1e-6:
            kap = kp.fzero(f, kap0)
            kap0 += 0.02

        return kap

    def distribute_points(self):
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

    def generate_lines(self):
        for ptSt, ptEnd in zip(self.pt[:-1], self.pt[1:]):
            L = Line()
            L.set_end_pts([ptSt, ptEnd])
            self.line.append(L)

    def get_lines(self):
        return self.line

    def set_pts(self, pt):
        self.pt = pt

    def set_end_pts(self, end_pt):
        self._end_pts = end_pt

    def _get_element_coords(self):
        return self.pt

    def getPtIDs(self):
        return [pt.get_id() for pt in self.pt]

    def display(self):
        print(
            (
                "{0} {1}: ID = {2}, Pt IDs = {3}".format(
                    self.__class__.__name__,
                    self.get_index(),
                    self.get_id(),
                    self.getPtIDs(),
                )
            )
        )

    def plot(self):
        x, y = list(zip(*[pt.get_position() for pt in self.pt]))
        plt.plot(x, y, self.plot_sty)


class Line(Curve):
    obj = []

    def __init__(self):
        Curve.__init__(self)
        self.set_index(Line.count())
        Line.obj.append(self)
        self.plot_sty = "b-"

    def set_end_pts(self, endPt):
        Curve.set_end_pts(self, endPt)
        self.set_pts(endPt)
