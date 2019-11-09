import abc
import os

import numpy as np
from planingfsi import config, general, trig
from planingfsi.fe import felib as fe
from scipy.interpolate import interp1d


class Substructure(abc.ABC):
    count = 0
    obj = []

    @classmethod
    def All(cls):
        return [o for o in cls.obj]

    @classmethod
    def find_by_name(cls, name):
        return [o for o in cls.obj if o.name == name][0]

    def __init__(self, dict_):
        self.index = Substructure.count
        Substructure.count += 1
        Substructure.obj.append(self)

        self.dict_ = dict_
        self.name = self.dict_.get("substructureName", "")
        self.type_ = self.dict_.get("substructureType", "rigid")
        self.interpolator = None

        self.seal_pressure = self.get_or_config("Ps", 0.0)
        self.seal_pressure_method = self.dict_.get("PsMethod", "constant")

        self.seal_over_pressure_pct = self.get_or_config("overPressurePct", 1.0)
        self.cushion_pressure_type = self.dict_.get("cushionPressureType", None)
        self.tip_load = self.get_or_config("tipLoad", 0.0)
        self.tip_constraint_height = self.dict_.get("tipConstraintHt", None)
        self.struct_interp_type = self.dict_.get("structInterpType", "linear")
        self.struct_extrap = self.dict_.get("structExtrap", True)
        self.line_fluid_pressure = None
        self.line_air_pressure = None
        self.fluidS = []
        self.fluidP = []
        self.airS = []
        self.airP = []
        self.U = 0.0
        self.node = []

    def add_parent(self, parent):
        self.parent = parent

    def get_residual(self):
        return 0.0

    def set_element_properties(self):
        for el in self.el:
            el.set_properties(length=self.get_arc_length() / len(self.el))

    def load_mesh(self):
        ndSt, ndEnd = np.loadtxt(
            os.path.join(config.path.mesh_dir, "elements_{0}.txt".format(self.name)), unpack=True,
        )
        if isinstance(ndSt, float):
            ndSt = [int(ndSt)]
            ndEnd = [int(ndEnd)]
        else:
            ndSt = [int(nd) for nd in ndSt]
            ndEnd = [int(nd) for nd in ndEnd]
        ndInd = ndSt + [ndEnd[-1]]

        # Generate Element list
        self.node = [fe.Node.get_index(i) for i in ndInd]

        self.set_interp_function()
        self.el = [self.element_type() for _ in ndSt]
        self.set_element_properties()
        for ndSti, ndEndi, el in zip(ndSt, ndEnd, self.el):
            el.set_nodes([fe.Node.get_index(ndSti), fe.Node.get_index(ndEndi)])
            el.set_parent(self)

    def set_interp_function(self):
        self.node_arc_length = np.zeros(len(self.node))
        for i, nd0, nd1 in zip(list(range(len(self.node) - 1)), self.node[:-1], self.node[1:]):
            self.node_arc_length[i + 1] = (
                self.node_arc_length[i] + ((nd1.x - nd0.x) ** 2 + (nd1.y - nd0.y) ** 2) ** 0.5
            )

        if len(self.node_arc_length) == 2:
            self.struct_interp_type = "linear"
        elif len(self.node_arc_length) == 3 and not self.struct_interp_type == "linear":
            self.struct_interp_type = "quadratic"

        x, y = [np.array(xx) for xx in zip(*[(nd.x, nd.y) for nd in self.node])]
        self.interp_func_x, self.interp_func_y = (
            interp1d(self.node_arc_length, x),
            interp1d(self.node_arc_length, y, kind=self.struct_interp_type),
        )

        if self.struct_extrap:
            self.interp_func_x, self.interp_func_y = self.extrap_coordinates(
                self.interp_func_x, self.interp_func_y
            )

    def extrap_coordinates(self, fxi, fyi):
        def extrap1d(interpolator):
            xs = interpolator.x
            ys = interpolator.y

            def pointwise(xi):
                if xi < xs[0]:
                    return ys[0] + (xi - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
                elif xi > xs[-1]:
                    return ys[-1] + (xi - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
                else:
                    return interpolator(xi)

            def ufunclike(xs):
                return np.array(list(map(pointwise, np.array([xs]))))[0]

            return ufunclike

        return extrap1d(fxi), extrap1d(fyi)

    def get_coordinates(self, si):
        return self.interp_func_x(si), self.interp_func_y(si)

    def get_xcoordinates(self, s):
        return self.get_coordinates(s)[0]

    def get_ycoordinates(self, s):
        return self.get_coordinates(s)[1]

    def get_arc_length(self):
        return max(self.node_arc_length)

    def write_coordinates(self):
        """"""
        # kp.writeaslist(
        #     os.path.join(
        #         config.it_dir, "coords_{0}.{1}".format(self.name, config.io.data_format)
        #     ),
        #     ["x [m]", [nd.x for nd in self.node]],
        #     ["y [m]", [nd.y for nd in self.node]],
        # )

    def load_coordinates(self):
        x, y = np.loadtxt(
            os.path.join(config.it_dir, "coords_{0}.{1}".format(self.name, config.io.data_format)),
            unpack=True,
        )
        for xx, yy, nd in zip(x, y, self.node):
            nd.set_coordinates(xx, yy)

    def update_fluid_forces(self):
        self.fluidS = []
        self.fluidP = []
        self.airS = []
        self.airP = []
        self.D = 0.0
        self.L = 0.0
        self.M = 0.0
        self.Da = 0.0
        self.La = 0.0
        self.Ma = 0.0
        if self.interpolator is not None:
            sMin, sMax = self.interpolator.get_min_max_s()

        for i, el in enumerate(self.el):
            # Get pressure at end points and all fluid points along element
            nodeS = [self.node_arc_length[i], self.node_arc_length[i + 1]]
            if self.interpolator is not None:
                s, pressure_fluid, tau = self.interpolator.get_loads_in_range(nodeS[0], nodeS[1])
                # Limit pressure to be below stagnation pressure
                if config.plotting.pressure_limiter:
                    pressure_fluid = np.min(
                        np.hstack(
                            (
                                pressure_fluid,
                                np.ones_like(pressure_fluid) * config.flow.stagnation_pressure,
                            )
                        ),
                        axis=0,
                    )

            else:
                s = np.array(nodeS)
                pressure_fluid = np.zeros_like(s)
                tau = np.zeros_like(s)

            ss = nodeS[1]
            Pc = 0.0
            if self.interpolator is not None:
                if ss > sMax:
                    Pc = self.interpolator.fluid.upstream_pressure
                elif ss < sMin:
                    Pc = self.interpolator.fluid.downstream_pressure
            elif self.cushion_pressure_type == "Total":
                Pc = config.body.Pc

            # Store fluid and air pressure components for element (for
            # plotting)
            if i == 0:
                self.fluidS += [s[0]]
                self.fluidP += [pressure_fluid[0]]
                self.airS += [nodeS[0]]
                self.airP += [Pc - self.seal_pressure]

            self.fluidS += [ss for ss in s[1:]]
            self.fluidP += [pp for pp in pressure_fluid[1:]]
            self.airS += [ss for ss in nodeS[1:]]
            if self.seal_pressure_method.lower() == "hydrostatic":
                self.airP += [
                    Pc
                    - self.seal_pressure
                    + config.flow.density
                    * config.flow.gravity
                    * (self.interp_func_y(si) - config.flow.waterline_height)
                    for si in nodeS[1:]
                ]
            else:
                self.airP += [Pc - self.seal_pressure for _ in nodeS[1:]]

            # Apply ramp to hydrodynamic pressure
            pressure_fluid *= config.ramp ** 2

            # Add external cushion pressure to external fluid pressure
            pressure_cushion = np.zeros_like(s)
            Pc = 0.0
            for ii, ss in enumerate(s):
                if self.interpolator is not None:
                    if ss > sMax:
                        Pc = self.interpolator.fluid.upstream_pressure
                    elif ss < sMin:
                        Pc = self.interpolator.fluid.downstream_pressure
                elif self.cushion_pressure_type == "Total":
                    Pc = config.body.Pc

                pressure_cushion[ii] = Pc

            # Calculate internal pressure
            if self.seal_pressure_method.lower() == "hydrostatic":
                pressure_internal = (
                    self.seal_pressure
                    - config.flow.density
                    * config.flow.gravity
                    * (
                        np.array([self.interp_func_y(si) for si in s])
                        - config.flow.waterline_height
                    )
                )
            else:
                pressure_internal = (
                    self.seal_pressure * np.ones_like(s) * self.seal_over_pressure_pct
                )

            pressure_external = pressure_fluid + pressure_cushion
            pressure_total = pressure_external - pressure_internal

            # Integrate pressure profile, calculate center of pressure and
            # distribute force to nodes
            integral = general.integrate(s, pressure_total)
            if integral == 0.0:
                qp = np.zeros(2)
            else:
                pct = (
                    general.integrate(s, s * pressure_total) / integral - s[0]
                ) / general.cumdiff(s)
                qp = integral * np.array([1 - pct, pct])

            integral = general.integrate(s, tau)
            if integral == 0.0:
                qs = np.zeros(2)
            else:
                pct = (general.integrate(s, s * tau) / integral - s[0]) / general.cumdiff(s)
                qs = -integral * np.array([1 - pct, pct])

            el.set_pressure_and_shear(qp, qs)

            # Calculate external force and moment for rigid body calculation
            if (
                config.body.cushion_force_method.lower() == "integrated"
                or config.body.cushion_force_method.lower() == "assumed"
            ):
                if config.body.cushion_force_method.lower() == "integrated":
                    integrand = pressure_external
                elif config.body.cushion_force_method.lower() == "assumed":
                    integrand = pressure_fluid

                n = list(map(self.get_normal_vector, s))
                t = [trig.rotate_vec_2d(ni, -90) for ni in n]

                f = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(integrand, tau, n, t)]

                r = [
                    np.array([pt[0] - self.parent.xCofR, pt[1] - self.parent.yCofR])
                    for pt in map(self.get_coordinates, s)
                ]

                m = [general.cross2(ri, fi) for ri, fi in zip(r, f)]

                self.D -= general.integrate(s, np.array(zip(*f)[0]))
                self.L += general.integrate(s, np.array(zip(*f)[1]))
                self.M += general.integrate(s, np.array(m))
            else:
                if self.interpolator is not None:
                    self.D = self.interpolator.fluid.drag_total
                    self.L = self.interpolator.fluid.lift_total
                    self.M = self.interpolator.fluid.moment_total

            integrand = pressure_cushion

            n = list(map(self.get_normal_vector, s))
            t = [trig.rotate_vec_2d(ni, -90) for ni in n]

            f = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(integrand, tau, n, t)]

            r = [
                np.array([pt[0] - self.parent.xCofR, pt[1] - self.parent.yCofR])
                for pt in map(self.get_coordinates, s)
            ]

            m = [general.cross2(ri, fi) for ri, fi in zip(r, f)]

            self.Da -= general.integrate(s, np.array(list(zip(*f))[0]))
            self.La += general.integrate(s, np.array(list(zip(*f))[1]))
            self.Ma += general.integrate(s, np.array(m))

    def get_normal_vector(self, s):
        dxds = general.getDerivative(lambda si: self.get_coordinates(si)[0], s)
        dyds = general.getDerivative(lambda si: self.get_coordinates(si)[1], s)

        return trig.rotate_vec_2d(trig.angd2vec2d(trig.atand2(dyds, dxds)), -90)

    def plot_pressure_profiles(self):
        if self.line_fluid_pressure is not None:
            self.line_fluid_pressure.set_data(
                self.get_pressure_plot_points(self.fluidS, self.fluidP)
            )
        if self.line_air_pressure is not None:
            self.line_air_pressure.set_data(self.get_pressure_plot_points(self.airS, self.airP))

    def get_pressure_plot_points(self, s0, p0):

        sp = [(s, p) for s, p in zip(s0, p0) if not np.abs(p) < 1e-4]

        if len(sp) > 0:
            s0, p0 = list(zip(*sp))
            nVec = list(map(self.get_normal_vector, s0))
            coords0 = [np.array(self.get_coordinates(s)) for s in s0]
            coords1 = [c + config.plotting.pScale * p * n for c, p, n in zip(coords0, p0, nVec)]

            return list(
                zip(
                    *[
                        xyi
                        for c0, c1 in zip(coords0, coords1)
                        for xyi in [c0, c1, np.ones(2) * np.nan]
                    ]
                )
            )
        else:
            return [], []

    def update_geometry(self):
        self.set_interp_function()

    def plot(self):
        for el in self.el:
            el.plot()
        #    for nd in [self.node[0],self.node[-1]]:
        #      nd.plot()
        self.plot_pressure_profiles()

    def set_attachments(self):
        return None

    def set_angle(self):
        return None

    def get_or_config(self, key, default):
        value = self.dict_.get(key, default)
        if isinstance(value, str):
            value = getattr(config, value)
        return value

    @abc.abstractmethod
    def set_fixed_dof(self) -> None:
        return NotImplemented


class FlexibleSubstructure(Substructure):

    obj = []
    res = 0.0

    @classmethod
    def update_all(cls):

        num_dof = fe.Node.count() * config.flow.num_dim
        Kg = np.zeros((num_dof, num_dof))
        Fg = np.zeros((num_dof, 1))
        Ug = np.zeros((num_dof, 1))

        # Assemble global matrices for all substructures together
        for ss in cls.obj:
            ss.update_fluid_forces()
            ss.assemble_global_stiffness_and_force()
            Kg += ss.K
            Fg += ss.F

        for nd in fe.Node.All():
            for i in range(2):
                Fg[nd.dof[i]] += nd.fixed_load[i]

        # Determine fixed degrees of freedom
        dof = [False for _ in Fg]

        for nd in fe.Node.All():
            for dofi, fdofi in zip(nd.dof, nd.is_dof_fixed):
                dof[dofi] = not fdofi

        # Solve FEM linear matrix equation
        if any(dof):
            Ug[np.ix_(dof)] = np.linalg.solve(Kg[np.ix_(dof, dof)], Fg[np.ix_(dof)])

        cls.res = np.max(np.abs(Ug))

        Ug *= config.solver.relax_FEM
        Ug *= np.min([config.solver.max_FEM_disp / np.max(Ug), 1.0])

        for nd in fe.Node.All():
            nd.move_coordinates(Ug[nd.dof[0], 0], Ug[nd.dof[1], 0])

        for ss in cls.obj:
            ss.update_geometry()

    def __init__(self, dict_):

        #    FlexibleSubstructure.obj.append(self)

        Substructure.__init__(self, dict_)
        self.element_type = fe.TrussElement
        self.pretension = self.dict_.get("pretension", -0.5)
        self.EA = self.dict_.get("EA", 5e7)

        self.K = None
        self.F = None
        #    self.U = None
        config.has_free_structure = True

    def get_residual(self):
        return np.max(np.abs(self.U))

    def initialize_matrices(self):
        num_dof = fe.Node.count() * config.flow.num_dim
        self.K = np.zeros((num_dof, num_dof))
        self.F = np.zeros((num_dof, 1))
        self.U = np.zeros((num_dof, 1))

    def assemble_global_stiffness_and_force(self):
        if self.K is None:
            self.initialize_matrices()
        else:
            self.K *= 0
            self.F *= 0
        for el in self.el:
            self.add_loads_from_element(el)

    def add_loads_from_element(self, el):
        K, F = el.get_stiffness_and_force()
        self.K[np.ix_(el.dof, el.dof)] += K
        self.F[np.ix_(el.dof)] += F

    #  def getPtDispFEM(self):
    # if self.K is None:
    # self.initializeMatrices()
    ##    self.U *= 0.0
    # self.update_fluid_forces()
    # self.assembleGlobalStiffnessAndForce()
    #
    #    dof = [False for dof in self.F]
    #    for nd in self.node:
    #      for dofi, fdofi in zip(nd.dof, nd.fixedDOF):
    #        dof[dofi] = not fdofi
    # if any(dof):
    ##      self.U[np.ix_(dof)] = np.linalg.solve(self.K[np.ix_(dof,dof)], self.F[np.ix_(dof)])
    #
    #    # Relax displacement and limit step if necessary
    #    self.U *= config.relaxFEM
    #    self.U *= np.min([config.maxFEMDisp / np.max(self.U), 1.0])
    #
    #    for nd in self.node:
    #      nd.moveCoordinates(self.U[nd.dof[0],0], self.U[nd.dof[1],0])
    #
    #    self.update_geometry()

    def set_element_properties(self):
        Substructure.set_element_properties(self)
        for el in self.el:
            el.set_properties(axialForce=-self.pretension, EA=self.EA)

    def set_fixed_dof(self) -> None:
        pass

    def update_geometry(self):
        for el in self.el:
            el.update_geometry()
        Substructure.set_interp_function(self)


class RigidSubstructure(Substructure):
    def __init__(self, dict_):
        Substructure.__init__(self, dict_)
        self.element_type = fe.RigidElement

    def set_attachments(self):
        return None

    def update_angle(self):
        return None

    def set_fixed_dof(self):
        for nd in self.node:
            for j in range(config.flow.num_dim):
                nd.is_dof_fixed[j] = True


class TorsionalSpringSubstructure(FlexibleSubstructure, RigidSubstructure):
    def __init__(self, dict_):
        FlexibleSubstructure.__init__(self, dict_)
        self.element_type = fe.RigidElement
        self.tip_load_pct = self.dict_.read("tipLoadPct", 0.0)
        self.base_pt_pct = self.dict_.read("basePtPct", 1.0)
        self.spring_constant = self.dict_.read("spring_constant", 1000.0)
        self.theta = 0.0
        self.Mt = 0.0  # TODO
        self.MOld = None
        self.relax = self.dict_.read("relaxAng", config.body.relax_rigid_body)
        self.attach_pct = self.dict_.read("attachPct", 0.0)
        self.attached_node = None
        self.attached_element = None
        self.minimum_angle = self.dict_.read("minimumAngle", -float("Inf"))
        self.max_angle_step = self.dict_.read("maxAngleStep", float("Inf"))
        config.has_free_structure = True

    def load_mesh(self):
        Substructure.load_mesh(self)
        self.set_fixed_dof()
        if self.base_pt_pct == 1.0:
            self.base_pt = self.node[-1].get_coordinates()
        elif self.base_pt_pct == 0.0:
            self.base_pt = self.node[0].get_coordinates()
        else:
            self.base_pt = self.get_coordinates(self.base_pt_pct * self.get_arc_length())

        self.set_element_properties()

        self.set_angle(self.dict_.read("initialAngle", 0.0))

    def set_attachments(self):
        attached_substructure_name = self.dict_.read("attachedSubstructure", None)
        if attached_substructure_name is not None:
            self.attached_substructure = Substructure.find_by_name(attached_substructure_name)
        else:
            self.attached_substructure = None

        if self.dict_.read("attachedSubstructureEnd", "End").lower() == "start":
            self.attached_ind = 0
        else:
            self.attached_ind = -1

        if self.attached_node is None and self.attached_substructure is not None:
            self.attached_node = self.attached_substructure.node[self.attached_ind]
            self.attached_element = self.attached_substructure.el[self.attached_ind]

    def update_fluid_forces(self):
        self.fluidS = []
        self.fluidP = []
        self.airS = []
        self.airP = []
        self.D = 0.0
        self.L = 0.0
        self.M = 0.0
        self.Dt = 0.0
        self.Lt = 0.0
        self.Mt = 0.0
        self.Da = 0.0
        self.La = 0.0
        self.Ma = 0.0
        if self.interpolator is not None:
            s_min, s_max = self.interpolator.get_min_max_s()

        for i, el in enumerate(self.el):
            # Get pressure at end points and all fluid points along element
            node_s = [self.node_arc_length[i], self.node_arc_length[i + 1]]
            if self.interpolator is not None:
                s, pressure_fluid, tau = self.interpolator.get_loads_in_range(node_s[0], node_s[1])

                # Limit pressure to be below stagnation pressure
                if config.plotting.pressure_limiter:
                    pressure_fluid = np.min(
                        np.hstack(
                            (
                                pressure_fluid,
                                np.ones_like(pressure_fluid) * config.flow.stagnation_pressure,
                            )
                        ),
                        axis=0,
                    )

            else:
                s = np.array(node_s)
                pressure_fluid = np.zeros_like(s)
                tau = np.zeros_like(s)

            ss = node_s[1]
            Pc = 0.0
            if self.interpolator is not None:
                if ss > s_max:
                    Pc = self.interpolator.fluid.get_upstream_pressure()
                elif ss < s_min:
                    Pc = self.interpolator.fluid.get_downstream_pressure()
            elif self.cushion_pressure_type == "Total":
                Pc = config.body.Pc

            # Store fluid and air pressure components for element (for
            # plotting)
            if i == 0:
                self.fluidS += [s[0]]
                self.fluidP += [pressure_fluid[0]]
                self.airS += [node_s[0]]
                self.airP += [Pc - self.seal_pressure]

            self.fluidS += [ss for ss in s[1:]]
            self.fluidP += [pp for pp in pressure_fluid[1:]]
            self.airS += [ss for ss in node_s[1:]]
            self.airP += [Pc - self.seal_pressure for _ in node_s[1:]]

            # Apply ramp to hydrodynamic pressure
            pressure_fluid *= config.ramp ** 2

            # Add external cushion pressure to external fluid pressure
            pressure_cushion = np.zeros_like(s)
            Pc = 0.0
            for ii, ss in enumerate(s):
                if self.interpolator is not None:
                    if ss > s_max:
                        Pc = self.interpolator.fluid.get_upstream_pressure()
                    elif ss < s_min:
                        Pc = self.interpolator.fluid.get_downstream_pressure()
                elif self.cushion_pressure_type == "Total":
                    Pc = config.body.Pc

                pressure_cushion[ii] = Pc

            pressure_internal = self.seal_pressure * np.ones_like(s)

            pressure_external = pressure_fluid + pressure_cushion
            pressure_total = pressure_external - pressure_internal

            # Calculate external force and moment for rigid body calculation
            if (
                config.body.cushion_force_method.lower() == "integrated"
                or config.body.cushion_force_method.lower() == "assumed"
            ):
                if config.body.cushion_force_method.lower() == "integrated":
                    integrand = pressure_external
                elif config.body.cushion_force_method.lower() == "assumed":
                    integrand = pressure_fluid

                n = list(map(self.get_normal_vector, s))
                t = [trig.rotate_vec_2d(ni, -90) for ni in n]

                fC = [
                    -pi * ni + taui * ti for pi, taui, ni, ti in zip(pressure_external, tau, n, t)
                ]
                fFl = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(pressure_fluid, tau, n, t)]
                f = fC + fFl
                print(("Cushion Lift-to-Weight: {0}".format(fC[1] / config.body.weight)))

                r = [
                    np.array([pt[0] - config.body.xCofR, pt[1] - config.body.yCofR])
                    for pt in map(self.get_coordinates, s)
                ]

                m = [general.cross2(ri, fi) for ri, fi in zip(r, f)]

                self.D -= general.integrate(s, np.array(zip(*f)[0]))
                self.L += general.integrate(s, np.array(zip(*f)[1]))
                self.M += general.integrate(s, np.array(m))
            else:
                if self.interpolator is not None:
                    self.D = self.interpolator.fluid.D
                    self.L = self.interpolator.fluid.L
                    self.M = self.interpolator.fluid.M

            # Apply pressure loading for moment calculation
            #      integrand = pFl
            integrand = pressure_total
            n = list(map(self.get_normal_vector, s))
            t = [trig.rotate_vec_2d(ni, -90) for ni in n]

            f = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(integrand, tau, n, t)]
            r = [
                np.array([pt[0] - self.base_pt[0], pt[1] - self.base_pt[1]])
                for pt in map(self.get_coordinates, s)
            ]

            m = [general.cross2(ri, fi) for ri, fi in zip(r, f)]
            fx, fy = list(zip(*f))

            self.Dt += general.integrate(s, np.array(fx))
            self.Lt += general.integrate(s, np.array(fy))
            self.Mt += general.integrate(s, np.array(m))

            integrand = pressure_cushion

            n = list(map(self.get_normal_vector, s))
            t = [trig.rotate_vec_2d(ni, -90) for ni in n]

            f = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(integrand, tau, n, t)]

            r = [
                np.array([pt[0] - self.parent.xCofR, pt[1] - self.parent.yCofR])
                for pt in map(self.get_coordinates, s)
            ]

            m = [general.cross2(ri, fi) for ri, fi in zip(r, f)]

            self.Da -= general.integrate(s, np.array(zip(*f)[0]))
            self.La += general.integrate(s, np.array(zip(*f)[1]))
            self.Ma += general.integrate(s, np.array(m))

        # Apply tip load
        tipC = self.get_coordinates(self.tip_load_pct * self.get_arc_length())
        tipR = np.array([tipC[i] - self.base_pt[i] for i in [0, 1]])
        tipF = np.array([0.0, self.tip_load]) * config.ramp
        tipM = general.cross2(tipR, tipF)
        self.Lt += tipF[1]
        self.Mt += tipM

        # Apply moment from attached substructure

    #    el = self.attachedEl
    #    attC = self.attachedNode.get_coordinates()
    #    attR = np.array([attC[i] - self.basePt[i] for i in [0,1]])
    #    attF = el.axialForce * kp.ang2vec(el.gamma + 180)
    #    attM = kp.cross2(attR, attF) * config.ramp
    ##    attM = np.min([np.abs(attM), np.abs(self.Mt)]) * kp.sign(attM)
    # if np.abs(attM) > 2 * np.abs(tipM):
    ##      attM = attM * np.abs(tipM) / np.abs(attM)
    #    self.Mt += attM

    def update_angle(self):

        if np.isnan(self.Mt):
            theta = 0.0
        else:
            theta = -self.Mt

        if not self.spring_constant == 0.0:
            theta /= self.spring_constant

        dTheta = (theta - self.theta) * self.relax
        dTheta = np.min([np.abs(dTheta), self.max_angle_step]) * np.sign(dTheta)

        self.set_angle(self.theta + dTheta)

    def set_angle(self, ang):
        dTheta = np.max([ang, self.minimum_angle]) - self.theta

        if self.attached_node is not None and not any(
            [nd == self.attached_node for nd in self.node]
        ):
            attNd = [self.attached_node]
        else:
            attNd = []

        #    basePt = np.array([c for c in self.basePt])
        basePt = np.array([c for c in self.node[-1].get_coordinates()])
        for nd in self.node + attNd:
            oldPt = np.array([c for c in nd.get_coordinates()])
            newPt = general.rotatePt(oldPt, basePt, -dTheta)
            nd.set_coordinates(newPt[0], newPt[1])

        self.theta += dTheta
        self.residual = dTheta
        self.update_geometry()
        print(("  Deformation for substructure {0}: {1}".format(self.name, self.theta)))

    def get_residual(self):
        return self.residual

    #    return self.theta + self.Mt / self.spring_constant

    def writeDeformation(self):
        general.writeasdict(
            os.path.join(
                config.it_dir, "deformation_{0}.{1}".format(self.name, config.io.data_format),
            ),
            ["angle", self.theta],
        )
