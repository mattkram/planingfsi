import pytest

from planingfsi.fsi.simulation import Simulation


@pytest.fixture()
def simulation() -> Simulation:
    return Simulation()


def test_has_references_to_solvers(simulation: Simulation) -> None:
    assert simulation.solid_solver is not None
    assert simulation.fluid_solver is not None
