import os

import numpy as np
from scipy.interpolate import interp1d

# import planingfsi.io
from planingfsi import config

# from planingfsi import krampy as kp
# from planingfsi import io

from . import felib as fe


class FEStructure:
    """Parent object for solving the finite-element structure. Consists of
    several rigid bodies and substructures.
    """

    def __init__(self):
        self.rigid_body = []
        self.substructure = []
        self.node = []
        self.res = 1.0

    def add_rigid_body(self, dict_=None):
        if dict_ is None:
            dict_ = planingfsi.io.Dictionary()
        rigid_body = RigidBody(dict_)
        self.rigid_body.append(rigid_body)
        return rigid_body

    def add_substructure(self, dict_=None):
        if dict_ is None:
            dict_ = planingfsi.io.Dictionary()
        ss_type = dict_.read("substructureType", "rigid")
        if ss_type.lower() == "flexible" or ss_type.lower() == "truss":
            ss = FlexibleSubstructure(dict_)
            FlexibleSubstructure.obj.append(ss)
        elif ss_type.lower() == "torsionalspring":
            ss = TorsionalSpringSubstructure(dict_)
        else:
            ss = RigidSubstructure(dict_)
        self.substructure.append(ss)

        # Find parent body and add substructure to it
        body = [
            b for b in self.rigid_body if b.name == dict_.read("bodyName", "default")
        ]
        if len(body) > 0:
            body = body[0]
        else:
            body = self.rigid_body[0]
        body.add_substructure(ss)
        ss.add_parent(body)
        print(
            (
                "Adding Substructure {0} of type {1} to rigid body {2}".format(
                    ss.name, ss.type_, body.name
                )
            )
        )

        return ss

    def initialize_rigid_bodies(self):
        for bd in self.rigid_body:
            bd.initialize_position()

    def update_fluid_forces(self):
        for bd in self.rigid_body:
            bd.update_fluid_forces()

    def calculate_response(self):
        if config.io.results_from_file:
            self.load_response()
        else:
            for bd in self.rigid_body:
                bd.update_position()
                bd.update_substructure_positions()

    def get_residual(self):
        self.res = 0.0
        for bd in self.rigid_body:
            if bd.free_in_draft or bd.free_in_trim:
                self.res = np.max([np.abs(bd.res_l), self.res])
                self.res = np.max([np.abs(bd.res_m), self.res])
            self.res = np.max([FlexibleSubstructure.res, self.res])

    def load_response(self):
        self.update_fluid_forces()

        for bd in self.rigid_body:
            bd.load_motion()
            for ss in bd.substructure:
                ss.load_coordinates()
                ss.update_geometry()

    def write_results(self):
        for bd in self.rigid_body:
            bd.write_motion()
            for ss in bd.substructure:
                ss.write_coordinates()

    def plot(self):
        for body in self.rigid_body:
            for struct in body.substructure:
                struct.plot()

    def load_mesh(self):
        # Create all nodes
        x, y = np.loadtxt(os.path.join(config.path.mesh_dir, "nodes.txt"), unpack=True)
        xf, yf = np.loadtxt(
            os.path.join(config.path.mesh_dir, "fixedDOF.txt"), unpack=True
        )
        fx, fy = np.loadtxt(
            os.path.join(config.path.mesh_dir, "fixedLoad.txt"), unpack=True
        )

        for xx, yy, xxf, yyf, ffx, ffy in zip(x, y, xf, yf, fx, fy):
            nd = fe.Node()
            nd.set_coordinates(xx, yy)
            nd.is_dof_fixed = [bool(xxf), bool(yyf)]
            nd.fixed_load = np.array([ffx, ffy])
            self.node.append(nd)

        for struct in self.substructure:
            struct.load_mesh()
            if (
                struct.type_ == "rigid"
                or struct.type_ == "rotating"
                or struct.type_ == "torsionalSpring"
            ):
                struct.set_fixed_dof()

        for ss in self.substructure:
            ss.set_attachments()


class RigidBody(object):
    def __init__(self, dict_):
        self.dict_ = dict_
        self.num_dim = 2
        self.draft = 0.0
        self.trim = 0.0

        self.name = self.dict_.read("bodyName", "default")
        self.weight = self.dict_.read(
            "W", self.dict_.read("loadPct", 1.0) * config.body.weight
        )
        self.weight *= config.body.seal_load_pct
        self.m = self.dict_.read("m", self.weight / config.flow.gravity)
        self.Iz = self.dict_.read("Iz", self.m * config.body.reference_length ** 2 / 12)
        self.has_planing_surface = self.dict_.read("hasPlaningSurface", False)

        var = [
            "max_draft_step",
            "max_trim_step",
            "free_in_draft",
            "free_in_trim",
            "draft_damping",
            "trim_damping",
            "max_draft_acc",
            "max_trim_acc",
            "xCofG",
            "yCofG",
            "xCofR",
            "yCofR",
            "initial_draft",
            "initial_trim",
            "relax_draft",
            "relax_trim",
            "time_step",
            "num_damp",
        ]
        for v in var:
            if v in self.dict_:
                setattr(self, v, self.dict_.read(v))
            elif hasattr(config.body, v):
                setattr(self, v, getattr(config.body, v))
            elif hasattr(config.solver, v):
                setattr(self, v, getattr(config.solver, v))
            elif hasattr(config, v):
                setattr(self, v, getattr(config, v))
            else:
                raise ValueError("Cannot find symbol: {0}".format(v))
            # setattr(self, v, self.dict_.read(v, getattr(config.body, getattr(config, v))))

        #    self.xCofR = self.dict_.read('xCofR', self.xCofG)
        #    self.yCofR = self.dict_.read('yCofR', self.yCofG)

        self.xCofR0 = self.xCofR
        self.yCofR0 = self.yCofR

        self.max_disp = np.array([self.max_draft_step, self.max_trim_step])
        self.free_dof = np.array([self.free_in_draft, self.free_in_trim])
        self.c_damp = np.array([self.draft_damping, self.trim_damping])
        self.max_acc = np.array([self.max_draft_acc, self.max_trim_acc])
        self.relax = np.array([self.relax_draft, self.relax_trim])
        if self.free_in_draft or self.free_in_trim:
            config.has_free_structure = True

        self.v = np.zeros((self.num_dim))
        self.a = np.zeros((self.num_dim))
        self.v_old = np.zeros((self.num_dim))
        self.a_old = np.zeros((self.num_dim))

        self.beta = self.dict_.read("beta", 0.25)
        self.gamma = self.dict_.read("gamma", 0.5)

        self.D = 0.0
        self.L = 0.0
        self.M = 0.0
        self.Da = 0.0
        self.La = 0.0
        self.Ma = 0.0

        self.solver = None
        self.disp_old = 0.0
        self.res_old = None
        self.two_ago_disp = 0.0
        self.predictor = True
        self.f_old = 0.0
        self.two_ago_f = 0.0
        self.res_l = 1.0
        self.res_m = 1.0

        self.J = None
        self.J_tmp = None

        # Assign displacement function depending on specified method
        self.get_disp = lambda: (0.0, 0.0)
        if any(self.free_dof):
            if config.body.motion_method == "Secant":
                self.get_disp = self.get_disp_secant
            elif config.body.motion_method == "Broyden":
                self.get_disp = self.get_disp_broyden
            elif config.body.motion_method == "BroydenNew":
                self.get_disp = self.get_disp_broyden_new
            elif config.body.motion_method == "Physical":
                self.get_disp = self.get_disp_physical
            elif config.body.motion_method == "Newmark-Beta":
                self.get_disp = self.get_disp_newmark_beta
            elif config.body.motion_method == "PhysicalNoMass":
                self.get_disp = self.get_disp_physical_no_mass
            elif config.body.motion_method == "Sep":
                self.get_disp = self.getDispSep
                self.trim_solver = None
                self.draft_solver = None

        self.substructure = []
        self.node = None

        print(("Adding Rigid Body: {0}".format(self.name)))

    def add_substructure(self, ss):
        self.substructure.append(ss)

    def store_nodes(self):
        self.node = []
        for ss in self.substructure:
            for nd in ss.node:
                if not any([n.node_num == nd.node_num for n in self.node]):
                    self.node.append(nd)

    def initialize_position(self):
        self.set_position(self.initial_draft, self.initial_trim)

    def set_position(self, draft, trim):
        self.update_position(draft - self.draft, trim - self.trim)

    def update_position(self, dDraft=None, dTrim=None):
        if dDraft is None:
            dDraft, dTrim = self.get_disp()
            if np.isnan(dDraft):
                dDraft = 0.0
            if np.isnan(dTrim):
                dTrim = 0.0

        if self.node is None:
            self.store_nodes()

        for nd in self.node:
            xo, yo = nd.get_coordinates()
            newPos = kp.rotatePt([xo, yo], [self.xCofR, self.yCofR], dTrim)
            nd.move_coordinates(newPos[0] - xo, newPos[1] - yo - dDraft)

        for s in self.substructure:
            s.update_geometry()

        self.xCofG, self.yCofG = kp.rotatePt(
            [self.xCofG, self.yCofG], [self.xCofR, self.yCofR], dTrim
        )
        self.yCofG -= dDraft
        self.yCofR -= dDraft

        self.draft += dDraft
        self.trim += dTrim

        self.print_motion()

    def update_substructure_positions(self):
        FlexibleSubstructure.update_all()
        for ss in self.substructure:
            print(("Updating position for substructure: {0}".format(ss.name)))
            #             if ss.type.lower() == 'flexible':
            #                 ss.getPtDispFEM()

            if ss.type_.lower() == "torsionalspring" or ss.type_.lower() == "rigid":
                ss.updateAngle()

    def update_fluid_forces(self):
        self.reset_loads()
        for ss in self.substructure:
            ss.update_fluid_forces()
            self.D += ss.D
            self.L += ss.L
            self.M += ss.M
            self.Da += ss.Da
            self.La += ss.La
            self.Ma += ss.Ma

        self.res_l = self.get_res_lift()
        self.res_m = self.get_res_moment()

    def reset_loads(self):
        self.D *= 0.0
        self.L *= 0.0
        self.M *= 0.0
        if config.body.cushion_force_method.lower() == "assumed":
            self.L += (
                config.body.Pc * config.body.reference_length * kp.cosd(config.trim)
            )

    def get_disp_physical(self):
        disp = self.limit_disp(self.time_step * self.v)

        for i in range(self.num_dim):
            if np.abs(disp[i]) == np.abs(self.max_disp[i]):
                self.v[i] = disp[i] / self.time_step

        self.v += self.time_step * self.a

        self.a = np.array(
            [self.weight - self.L, self.M - self.weight * (self.xCofG - self.xCofR)]
        )
        #    self.a -= self.Cdamp * (self.v + self.v**3) * config.ramp
        self.a -= self.c_damp * self.v * config.ramp
        self.a /= np.array([self.m, self.Iz])
        self.a = np.min(
            np.vstack(
                (np.abs(self.a), np.array([self.max_draft_acc, self.max_trim_acc]))
            ),
            axis=0,
        ) * np.sign(self.a)

        #    accLimPct = np.min(np.vstack((np.abs(self.a), self.maxAcc)), axis=0) * np.sign(self.a)
        #    for i in range(len(self.a)):
        #      if self.a[i] == 0.0 or not self.freeDoF[i]:
        #        accLimPct[i] = 1.0
        #      else:
        #        accLimPct[i] /= self.a[i]
        #
        #    self.a *= np.min(accLimPct)
        disp *= config.ramp
        return disp

    def get_disp_newmark_beta(self):
        self.a = np.array(
            [self.weight - self.L, self.M - self.weight * (self.xCofG - self.xCofR)]
        )
        #    self.a -= self.Cdamp * self.v * config.ramp
        self.a /= np.array([self.m, self.Iz])
        self.a = np.min(
            np.vstack(
                (np.abs(self.a), np.array([self.max_draft_acc, self.max_trim_acc]))
            ),
            axis=0,
        ) * np.sign(self.a)

        dv = (1 - self.gamma) * self.a_old + self.gamma * self.a
        dv *= self.time_step
        dv *= 1 - self.numDamp
        self.v += dv

        disp = 0.5 * (1 - 2 * self.beta) * self.a_old + self.beta * self.a
        disp *= self.time_step
        disp += self.v_old
        disp *= self.time_step

        self.a_old = self.a
        self.v_old = self.v

        disp *= self.relax
        disp *= config.ramp
        disp = self.limit_disp(disp)
        #    disp *= config.ramp

        return disp

    def get_disp_physical_no_mass(self):
        F = np.array(
            [
                self.weight - self.L,
                self.M - self.weight * (config.body.xCofG - config.body.xCofR),
            ]
        )
        #    F -= self.Cdamp * self.v

        if self.predictor:
            disp = F / self.c_damp * self.time_step
            self.predictor = False
        else:
            disp = 0.5 * self.time_step / self.c_damp * (F - self.two_ago_f)
            self.predictor = True

        disp *= self.relax * config.ramp
        disp = self.limit_disp(disp)

        #    self.v = disp / self.timeStep

        self.two_ago_f = self.f_old
        self.f_old = disp * self.c_damp / self.time_step

        return disp

    def get_disp_secant(self):
        if self.solver is None:
            self.resFun = lambda x: np.array(
                [
                    self.L - self.weight,
                    self.M - self.weight * (config.body.xCofG - config.body.xCofR),
                ]
            )
            self.solver = kp.RootFinder(
                self.resFun,
                np.array([config.body.initial_draft, config.body.initial_trim]),
                "secant",
                dxMax=self.max_disp * self.free_dof,
            )

        if not self.disp_old is None:
            self.solver.takeStep(self.disp_old)

        # Solve for limited displacement
        disp = self.solver.limitStep(self.solver.getStep())

        self.two_ago_disp = self.disp_old
        self.disp_old = disp

        return disp

    def reset_jacobian(self):
        if self.J_tmp is None:
            self.Jit = 0
            self.J_tmp = np.zeros((self.num_dim, self.num_dim))
            self.step = 0
            self.Jfo = self.resFun(self.x)
            self.res_old = self.Jfo * 1.0
        else:
            f = self.resFun(self.x)

            self.J_tmp[:, self.Jit] = (f - self.Jfo) / self.disp_old[self.Jit]
            self.Jit += 1

        disp = np.zeros((self.num_dim))
        if self.Jit < self.num_dim:
            disp[self.Jit] = config.body.motion_jacobian_first_step

        if self.Jit > 0:
            disp[self.Jit - 1] = -config.body.motion_jacobian_first_step
        self.disp_old = disp
        if self.Jit >= self.num_dim:
            self.J = self.J_tmp * 1.0
            self.J_tmp = None
            self.disp_old = None

        return disp

    def get_disp_broyden_new(self):
        if self.solver is None:
            self.resFun = lambda x: np.array(
                [
                    f
                    for f, free_dof in zip(
                        [self.get_res_lift(), self.get_res_moment()], self.free_dof
                    )
                    if free_dof
                ]
            )
            maxDisp = [
                m for m, free_dof in zip(self.max_disp, self.free_dof) if free_dof
            ]

            self.x = np.array(
                [
                    f
                    for f, free_dof in zip([self.draft, self.trim], self.free_dof)
                    if free_dof
                ]
            )
            self.solver = kp.RootFinder(self.resFun, self.x, "broyden", dxMax=maxDisp)

        if not self.disp_old is None:
            self.solver.takeStep(self.disp_old)

        # Solve for limited displacement
        disp = self.solver.limitStep(self.solver.getStep())

        self.two_ago_disp = self.disp_old
        self.disp_old = disp
        return disp

    def get_disp_broyden(self):
        if self.solver is None:
            self.resFun = lambda x: np.array(
                [self.L - self.weight, self.M - self.weight * (self.xCofG - self.xCofR)]
            )
            #      self.resFun = lambda x: np.array([self.get_res_moment(), self.get_res_lift()])
            self.solver = 1.0
            self.x = np.array([self.draft, self.trim])

        if self.J is None:
            disp = self.reset_jacobian()
        else:
            self.f = self.resFun(self.x)
            if not self.disp_old is None:
                self.x += self.disp_old

                dx = np.reshape(self.disp_old, (self.num_dim, 1))
                df = np.reshape(self.f - self.res_old, (self.num_dim, 1))

                self.J += (
                    np.dot(df - np.dot(self.J, dx), dx.T) / np.linalg.norm(dx) ** 2
                )

            dof = self.free_dof
            dx = np.zeros_like(self.x)
            A = -self.J[np.ix_(dof, dof)]
            b = self.f.reshape(self.num_dim, 1)[np.ix_(dof)]

            dx[np.ix_(dof)] = np.linalg.solve(A, b)

            if self.res_old is not None:
                if any(np.abs(self.f) - np.abs(self.res_old) > 0.0):
                    self.step += 1

            if self.step >= 6:
                print("\nResetting Jacobian for Motion\n")
                disp = self.reset_jacobian()

            disp = dx.reshape(self.num_dim)

            #      disp = self.solver.getStep()

            disp *= self.relax
            disp = self.limit_disp(disp)

            self.disp_old = disp

            self.res_old = self.f * 1.0
            self.step += 1
        return disp

    def limit_disp(self, disp):
        dispLimPct = np.min(np.vstack((np.abs(disp), self.max_disp)), axis=0) * np.sign(
            disp
        )
        for i in range(len(disp)):
            if disp[i] == 0.0 or not self.free_dof[i]:
                dispLimPct[i] = 1.0
            else:
                dispLimPct[i] /= disp[i]

        return disp * np.min(dispLimPct) * self.free_dof

    def get_res_lift(self):
        if np.isnan(self.L):
            res = 1.0
        else:
            res = (self.L - self.weight) / (
                config.flow.stagnation_pressure * config.body.reference_length + 1e-6
            )
        return np.abs(res * self.free_in_draft)

    def get_res_moment(self):
        if np.isnan(self.M):
            res = 1.0
        else:
            if self.xCofG == self.xCofR and self.M == 0.0:
                res = 1.0
            else:
                res = (self.M - self.weight * (self.xCofG - self.xCofR)) / (
                    config.flow.stagnation_pressure * config.body.reference_length ** 2
                    + 1e-6
                )
        return np.abs(res * self.free_in_trim)

    def print_motion(self):
        print(("Rigid Body Motion: {0}".format(self.name)))
        print(("  CofR: ({0}, {1})".format(self.xCofR, self.yCofR)))
        print(("  CofG: ({0}, {1})".format(self.xCofG, self.yCofG)))
        print(("  Draft:      {0:5.4e}".format(self.draft)))
        print(("  Trim Angle: {0:5.4e}".format(self.trim)))
        print(("  Lift Force: {0:5.4e}".format(self.L)))
        print(("  Drag Force: {0:5.4e}".format(self.D)))
        print(("  Moment:     {0:5.4e}".format(self.M)))
        print(("  Lift Force Air: {0:5.4e}".format(self.La)))
        print(("  Drag Force Air: {0:5.4e}".format(self.Da)))
        print(("  Moment Air:     {0:5.4e}".format(self.Ma)))
        print(("  Lift Res:   {0:5.4e}".format(self.res_l)))
        print(("  Moment Res: {0:5.4e}".format(self.res_m)))

    def write_motion(self):
        kp.writeasdict(
            os.path.join(
                config.it_dir, "motion_{0}.{1}".format(self.name, config.io.data_format)
            ),
            ["xCofR", self.xCofR],
            ["yCofR", self.yCofR],
            ["xCofG", self.xCofG],
            ["yCofG", self.yCofG],
            ["draft", self.draft],
            ["trim", self.trim],
            ["liftRes", self.res_l],
            ["momentRes", self.res_m],
            ["Lift", self.L],
            ["Drag", self.D],
            ["Moment", self.M],
            ["LiftAir", self.La],
            ["DragAir", self.Da],
            ["MomentAir", self.Ma],
        )
        for ss in self.substructure:
            if ss.type_ == "torsionalSpring":
                ss.writeDeformation()

    def load_motion(self):
        K = kp.dict_ionary(
            os.path.join(
                config.it_dir, "motion_{0}.{1}".format(self.name, config.io.data_format)
            )
        )
        self.xCofR = K.read("xCofR", np.nan)
        self.yCofR = K.read("yCofR", np.nan)
        self.xCofG = K.read("xCofG", np.nan)
        self.yCofG = K.read("yCofG", np.nan)
        self.draft = K.read("draft", np.nan)
        self.trim = K.read("trim", np.nan)
        self.res_l = K.read("liftRes", np.nan)
        self.res_m = K.read("momentRes", np.nan)
        self.L = K.read("Lift", np.nan)
        self.D = K.read("Drag", np.nan)
        self.M = K.read("Moment", np.nan)
        self.La = K.read("LiftAir", np.nan)
        self.Da = K.read("DragAir", np.nan)
        self.Ma = K.read("MomentAir", np.nan)


class Substructure:
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
        self.name = self.dict_.read("substructureName", "")
        self.type_ = self.dict_.read("substructureType", "rigid")
        self.interpolator = None

        self.seal_pressure = self.dict_.read_load_or_default("Ps", 0.0)
        self.seal_pressure_method = self.dict_.read("PsMethod", "constant")

        self.seal_over_pressure_pct = self.dict_.read_load_or_default(
            "overPressurePct", 1.0
        )
        self.cushion_pressure_type = self.dict_.read("cushionPressureType", None)
        self.tip_load = self.dict_.read_load_or_default("tipLoad", 0.0)
        self.tip_constraint_height = self.dict_.read("tipConstraintHt", None)
        self.struct_interp_type = self.dict_.read("structInterpType", "linear")
        self.struct_extrap = self.dict_.read("structExtrap", True)
        self.line_fluid_pressure = None
        self.line_air_pressure = None
        self.fluidS = []
        self.fluidP = []
        self.airS = []
        self.airP = []
        self.U = 0.0

    def add_parent(self, parent):
        self.parent = parent

    def get_residual(self):
        return 0.0

    def set_interpolator(self, interpolator):
        self.interpolator = interpolator

    def set_element_properties(self):
        for el in self.el:
            el.set_properties(length=self.get_arc_length() / len(self.el))

    def load_mesh(self):
        ndSt, ndEnd = np.loadtxt(
            os.path.join(config.path.mesh_dir, "elements_{0}.txt".format(self.name)),
            unpack=True,
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
        for i, nd0, nd1 in zip(
            list(range(len(self.node) - 1)), self.node[:-1], self.node[1:]
        ):
            self.node_arc_length[i + 1] = (
                self.node_arc_length[i]
                + ((nd1.x - nd0.x) ** 2 + (nd1.y - nd0.y) ** 2) ** 0.5
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
                    return ys[-1] + (xi - xs[-1]) * (ys[-1] - ys[-2]) / (
                        xs[-1] - xs[-2]
                    )
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
        kp.writeaslist(
            os.path.join(
                config.it_dir, "coords_{0}.{1}".format(self.name, config.io.data_format)
            ),
            ["x [m]", [nd.x for nd in self.node]],
            ["y [m]", [nd.y for nd in self.node]],
        )

    def load_coordinates(self):
        x, y = np.loadtxt(
            os.path.join(
                config.it_dir, "coords_{0}.{1}".format(self.name, config.io.data_format)
            ),
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
                s, pressure_fluid, tau = self.interpolator.get_loads_in_range(
                    nodeS[0], nodeS[1]
                )
                # Limit pressure to be below stagnation pressure
                if config.plotting.pressure_limiter:
                    pressure_fluid = np.min(
                        np.hstack(
                            (
                                pressure_fluid,
                                np.ones_like(pressure_fluid)
                                * config.flow.stagnation_pressure,
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
            integral = kp.integrate(s, pressure_total)
            if integral == 0.0:
                qp = np.zeros(2)
            else:
                pct = (
                    kp.integrate(s, s * pressure_total) / integral - s[0]
                ) / kp.cumdiff(s)
                qp = integral * np.array([1 - pct, pct])

            integral = kp.integrate(s, tau)
            if integral == 0.0:
                qs = np.zeros(2)
            else:
                pct = (kp.integrate(s, s * tau) / integral - s[0]) / kp.cumdiff(s)
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
                t = [kp.rotateVec(ni, -90) for ni in n]

                f = [
                    -pi * ni + taui * ti
                    for pi, taui, ni, ti in zip(integrand, tau, n, t)
                ]

                r = [
                    np.array([pt[0] - self.parent.xCofR, pt[1] - self.parent.yCofR])
                    for pt in map(self.get_coordinates, s)
                ]

                m = [kp.cross2(ri, fi) for ri, fi in zip(r, f)]

                self.D -= kp.integrate(s, np.array(zip(*f)[0]))
                self.L += kp.integrate(s, np.array(zip(*f)[1]))
                self.M += kp.integrate(s, np.array(m))
            else:
                if self.interpolator is not None:
                    self.D = self.interpolator.fluid.D
                    self.L = self.interpolator.fluid.L
                    self.M = self.interpolator.fluid.M

            integrand = pressure_cushion

            n = list(map(self.get_normal_vector, s))
            t = [kp.rotateVec(ni, -90) for ni in n]

            f = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(integrand, tau, n, t)]

            r = [
                np.array([pt[0] - self.parent.xCofR, pt[1] - self.parent.yCofR])
                for pt in map(self.get_coordinates, s)
            ]

            m = [kp.cross2(ri, fi) for ri, fi in zip(r, f)]

            self.Da -= kp.integrate(s, np.array(list(zip(*f))[0]))
            self.La += kp.integrate(s, np.array(list(zip(*f))[1]))
            self.Ma += kp.integrate(s, np.array(m))

    def get_normal_vector(self, s):
        dxds = kp.getDerivative(lambda si: self.get_coordinates(si)[0], s)
        dyds = kp.getDerivative(lambda si: self.get_coordinates(si)[1], s)

        return kp.rotateVec(kp.ang2vecd(kp.atand2(dyds, dxds)), -90)

    def plot_pressure_profiles(self):
        if self.line_fluid_pressure is not None:
            self.line_fluid_pressure.set_data(
                self.get_pressure_plot_points(self.fluidS, self.fluidP)
            )
        if self.line_air_pressure is not None:
            self.line_air_pressure.set_data(
                self.get_pressure_plot_points(self.airS, self.airP)
            )

    def get_pressure_plot_points(self, s0, p0):

        sp = [(s, p) for s, p in zip(s0, p0) if not np.abs(p) < 1e-4]

        if len(sp) > 0:
            s0, p0 = list(zip(*sp))
            nVec = list(map(self.get_normal_vector, s0))
            coords0 = [np.array(self.get_coordinates(s)) for s in s0]
            coords1 = [
                c + config.plotting.pScale * p * n for c, p, n in zip(coords0, p0, nVec)
            ]

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

        Ug *= config.relax_FEM
        Ug *= np.min([config.max_FEM_disp / np.max(Ug), 1.0])

        for nd in fe.Node.All():
            nd.move_coordinates(Ug[nd.dof[0], 0], Ug[nd.dof[1], 0])

        for ss in cls.obj:
            ss.update_geometry()

    def __init__(self, dict_):

        #    FlexibleSubstructure.obj.append(self)

        Substructure.__init__(self, dict_)
        self.element_type = fe.TrussElement
        self.pretension = self.dict_.read("pretension", -0.5)
        self.EA = self.dict_.read("EA", 5e7)

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

    def updateAngle(self):
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
            self.base_pt = self.get_coordinates(
                self.base_pt_pct * self.get_arc_length()
            )

        self.set_element_properties()

        self.set_angle(self.dict_.read("initialAngle", 0.0))

    def set_attachments(self):
        attached_substructure_name = self.dict_.read("attachedSubstructure", None)
        if attached_substructure_name is not None:
            self.attached_substructure = Substructure.find_by_name(
                attached_substructure_name
            )
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
                s, pressure_fluid, tau = self.interpolator.get_loads_in_range(
                    node_s[0], node_s[1]
                )

                # Limit pressure to be below stagnation pressure
                if config.plotting.pressure_limiter:
                    pressure_fluid = np.min(
                        np.hstack(
                            (
                                pressure_fluid,
                                np.ones_like(pressure_fluid)
                                * config.flow.stagnation_pressure,
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
                t = [kp.rotateVec(ni, -90) for ni in n]

                fC = [
                    -pi * ni + taui * ti
                    for pi, taui, ni, ti in zip(pressure_external, tau, n, t)
                ]
                fFl = [
                    -pi * ni + taui * ti
                    for pi, taui, ni, ti in zip(pressure_fluid, tau, n, t)
                ]
                f = fC + fFl
                print(
                    ("Cushion Lift-to-Weight: {0}".format(fC[1] / config.body.weight))
                )

                r = [
                    np.array([pt[0] - config.body.xCofR, pt[1] - config.body.yCofR])
                    for pt in map(self.get_coordinates, s)
                ]

                m = [kp.cross2(ri, fi) for ri, fi in zip(r, f)]

                self.D -= kp.integrate(s, np.array(zip(*f)[0]))
                self.L += kp.integrate(s, np.array(zip(*f)[1]))
                self.M += kp.integrate(s, np.array(m))
            else:
                if self.interpolator is not None:
                    self.D = self.interpolator.fluid.D
                    self.L = self.interpolator.fluid.L
                    self.M = self.interpolator.fluid.M

            # Apply pressure loading for moment calculation
            #      integrand = pFl
            integrand = pressure_total
            n = list(map(self.get_normal_vector, s))
            t = [kp.rotateVec(ni, -90) for ni in n]

            f = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(integrand, tau, n, t)]
            r = [
                np.array([pt[0] - self.base_pt[0], pt[1] - self.base_pt[1]])
                for pt in map(self.get_coordinates, s)
            ]

            m = [kp.cross2(ri, fi) for ri, fi in zip(r, f)]
            fx, fy = list(zip(*f))

            self.Dt += kp.integrate(s, np.array(fx))
            self.Lt += kp.integrate(s, np.array(fy))
            self.Mt += kp.integrate(s, np.array(m))

            integrand = pressure_cushion

            n = list(map(self.get_normal_vector, s))
            t = [kp.rotateVec(ni, -90) for ni in n]

            f = [-pi * ni + taui * ti for pi, taui, ni, ti in zip(integrand, tau, n, t)]

            r = [
                np.array([pt[0] - self.parent.xCofR, pt[1] - self.parent.yCofR])
                for pt in map(self.get_coordinates, s)
            ]

            m = [kp.cross2(ri, fi) for ri, fi in zip(r, f)]

            self.Da -= kp.integrate(s, np.array(zip(*f)[0]))
            self.La += kp.integrate(s, np.array(zip(*f)[1]))
            self.Ma += kp.integrate(s, np.array(m))

        # Apply tip load
        tipC = self.get_coordinates(self.tip_load_pct * self.get_arc_length())
        tipR = np.array([tipC[i] - self.base_pt[i] for i in [0, 1]])
        tipF = np.array([0.0, self.tip_load]) * config.ramp
        tipM = kp.cross2(tipR, tipF)
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

    def updateAngle(self):

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
            newPt = kp.rotatePt(oldPt, basePt, -dTheta)
            nd.set_coordinates(newPt[0], newPt[1])

        self.theta += dTheta
        self.residual = dTheta
        self.update_geometry()
        print(("  Deformation for substructure {0}: {1}".format(self.name, self.theta)))

    def get_residual(self):
        return self.residual

    #    return self.theta + self.Mt / self.spring_constant

    def writeDeformation(self):
        kp.writeasdict(
            os.path.join(
                config.it_dir,
                "deformation_{0}.{1}".format(self.name, config.io.data_format),
            ),
            ["angle", self.theta],
        )
