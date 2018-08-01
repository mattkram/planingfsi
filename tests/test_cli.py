import pytest
from click.testing import CliRunner

from planingfsi.cli import planingfsi, generate_mesh


@pytest.fixture
def runner():
    return CliRunner()


def test_run_main_cli(runner):
    results = runner.invoke(planingfsi)

    assert results.exit_code == 0


def test_run_mesh_cli(runner):
    results = runner.invoke(generate_mesh)
    assert results.exit_code == 0
