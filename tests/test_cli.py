import pytest

from planingfsi.cli import run_planingfsi, generate_mesh


def test_run_planingfsi(runner):
    results = runner.invoke(run_planingfsi)
    assert results.exit_code == 0


@pytest.mark.skip()
def test_run_generate_mesh(runner):
    results = runner.invoke(generate_mesh)
    assert results.exit_code == 0
