from typing import Any
from typing import Dict
from typing import Type

import pytest

from planingfsi.fsi.simulation import Simulation  # noreorder
from planingfsi.fe.structure import StructuralSolver
import planingfsi.fe.substructure as ss


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
    solver: StructuralSolver, data: Dict[str, Any], expected: bool
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
    solver: StructuralSolver, class_: Type[ss.Substructure], expected: bool
) -> None:
    """
    If the rigid body contains a substructure of a type that is free to move,
    then the solver has a free structure.
    """
    rigid_body = solver.add_rigid_body()
    # TODO: We shouldn't require a dict_ to be passed in
    rigid_body.add_substructure(class_(dict_={}))
    assert solver.has_free_structure is expected
