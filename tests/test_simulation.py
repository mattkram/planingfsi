from pathlib import Path

import pytest

from planingfsi.simulation import Simulation


@pytest.fixture()
def simulation(tmp_path: Path) -> Simulation:
    simulation = Simulation()
    simulation.case_dir = tmp_path
    return simulation


def test_has_references_to_solvers(simulation: Simulation) -> None:
    assert simulation.structural_solver is not None
    assert simulation.fluid_solver is not None


def test_increment_normal(simulation: Simulation) -> None:
    """Normal iteration simply increments by one."""
    assert simulation.it == 0
    for i in range(1, 10):
        simulation.increment()
        assert simulation.it == i


def test_increment_from_files(simulation: Simulation) -> None:
    """Given a set of stored iteration directories, the iteration tracks those directories."""
    simulation.config.io.results_from_file = True
    simulation.config.solver.max_it = 100
    for it in [0, 10, 20]:
        (simulation.case_dir / str(it)).mkdir()

    assert simulation.it == 0
    simulation.increment()
    assert simulation.it == 10
    simulation.increment()
    assert simulation.it == 20
    simulation.increment()
    assert simulation.it == 101
