import numpy as np
import pytest
from numpy.testing import assert_array_equal

from planingfsi.fe.rigid_body import RigidBody
from planingfsi.fe.rigid_body import RigidBodyMotionSolver


@pytest.fixture()
def rigid_body(monkeypatch):
    rigid_body = RigidBody(
        free_in_draft=True,
        free_in_trim=True,
        weight=100.0,
        x_cr=0.0,
        x_cg=1.0,
        max_draft_step=0.5,
        max_trim_step=0.5,
    )
    rigid_body.config.flow.flow_speed = 1.0

    def f(self):
        self.loads.L = 100.0 * self.draft
        self.loads.M = 100.0 * self.trim
        self._res_l = self.get_res_lift()
        self._res_m = self.get_res_moment()

    monkeypatch.setattr(RigidBody, "update_fluid_forces", f)

    return rigid_body


def test_rigid_body_init(rigid_body):
    assert rigid_body.name == "default"


@pytest.mark.parametrize(
    "attr_name, expected",
    [("draft", pytest.approx(1.0)), ("trim", pytest.approx(1.0, rel=1e-3))],
)
def test_solve(rigid_body, attr_name, expected):
    while rigid_body.residual > 1e-6:
        rigid_body.update_fluid_forces()
        rigid_body.update_position()

    assert getattr(rigid_body, attr_name) == expected


@pytest.fixture()
def solver(rigid_body) -> RigidBodyMotionSolver:
    rigid_body._max_disp = np.array([0.5, 0.5])
    return RigidBodyMotionSolver(rigid_body)


@pytest.mark.parametrize(
    ["free_dof", "disp", "expected"],
    [
        (np.array([True, True]), np.array([1.0, -1.0]), np.array([0.5, -0.5])),
        (np.array([True, True]), np.array([0.2, -1.0]), np.array([0.1, -0.5])),
        (np.array([True, True]), np.array([-0.2, -1.0]), np.array([-0.1, -0.5])),
        (np.array([False, True]), np.array([1.0, -1.0]), np.array([0.0, -0.5])),
    ],
)
def test_limit_disp(solver: RigidBodyMotionSolver, free_dof, disp, expected):
    """We can limit the step. If it is limited, the ratio remains the same."""
    solver.parent._free_dof = free_dof
    limited_disp = solver._limit_disp(disp)
    assert_array_equal(limited_disp, expected)
    if np.all(free_dof):
        assert limited_disp[1] / limited_disp[0] == pytest.approx(disp[1] / disp[0])
