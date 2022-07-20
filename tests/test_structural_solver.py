from __future__ import annotations

from typing import Any

import numpy
import pytest

import planingfsi.fe.substructure as ss
from planingfsi.fe.rigid_body import RigidBody
from planingfsi.fe.structure import StructuralSolver
from planingfsi.fsi.simulation import Simulation  # noreorder


@pytest.fixture()
def solver() -> StructuralSolver:
    simulation = Simulation()
    solver = StructuralSolver(simulation=simulation)
    return solver


def test_empty_solver_has_no_free_structure(solver: StructuralSolver) -> None:
    """By default, there is no free structure."""
    assert not solver.has_free_structure


@pytest.mark.parametrize(
    "data, expected",
    [
        ({}, False),
        (dict(free_in_trim=True), True),
        (dict(free_in_draft=True), True),
        (dict(free_in_trim=True, free_in_draft=True), True),
    ],
)
def test_solver_with_rigid_body_has_free_structure(
    solver: StructuralSolver, data: dict[str, Any], expected: bool
) -> None:
    """The solver has a free structure if any rigid body is free in trim or draft."""
    solver.add_rigid_body(data)
    assert solver.has_free_structure is expected


@pytest.mark.parametrize(
    "class_, expected",
    [
        (ss.RigidSubstructure, False),
        (ss.FlexibleSubstructure, True),
        (ss.TorsionalSpringSubstructure, True),
    ],
)
def test_solver_with_substructure_has_free_structure(
    solver: StructuralSolver, class_: type[ss.Substructure], expected: bool
) -> None:
    """
    If the rigid body contains a substructure of a type that is free to move,
    then the solver has a free structure.
    """
    rigid_body = solver.add_rigid_body()
    # TODO: We shouldn't require a dict_ to be passed in
    rigid_body.add_substructure(class_(dict_={}, solver=solver))
    assert solver.has_free_structure is expected


@pytest.fixture()
def solver_with_body(solver: StructuralSolver) -> tuple[StructuralSolver, RigidBody]:
    """Select parameters such that lift residual is simply abs(L-W)."""
    # These make the stagnation pressure equal to 1.0
    solver.config.flow.flow_speed = 1.0
    solver.config.flow.density = 2.0
    solver.config.body.reference_length = 1.0
    solver.config.body.weight = 2.0

    body = solver.add_rigid_body()
    body.free_in_draft = True
    body.free_in_trim = True

    return solver, body


@pytest.mark.parametrize("lift, expected", [(1.0, 1.0), (numpy.nan, 1.0), (2.0, 0.0)])
def test_solver_lift_residual(
    solver_with_body: tuple[StructuralSolver, RigidBody], lift: float, expected: float
) -> None:
    solver, body = solver_with_body
    body.L = lift
    assert solver.res_l == pytest.approx(expected)


@pytest.mark.parametrize("moment, expected", [(1.0, 1.0), (numpy.nan, 1.0)])
def test_solver_moment_residual(
    solver_with_body: tuple[StructuralSolver, RigidBody], moment: float, expected: float
) -> None:
    # TODO: Need to cover change in CoG, CoR
    solver, body = solver_with_body
    body.M = moment
    assert solver.res_m == pytest.approx(expected)
