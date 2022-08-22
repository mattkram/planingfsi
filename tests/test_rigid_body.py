import pytest

from planingfsi.fe.rigid_body import RigidBody


@pytest.fixture()
def rigid_body(monkeypatch):
    rigid_body = RigidBody(
        free_in_draft=True,
        free_in_trim=True,
        weight=100.0,
        x_cr=0.0,
        x_cg=1.0,
        motion_method="Broyden",
        max_draft_step=0.5,
        max_trim_step=0.5,
    )
    rigid_body.config.flow.flow_speed = 1.0

    def f(self):
        self.L = 100.0 * self.draft
        self.M = 100.0 * self.trim
        self.res_l = self.get_res_lift()
        self.res_m = self.get_res_moment()

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
