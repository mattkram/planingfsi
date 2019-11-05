import pytest

from planingfsi.cli import cli


@pytest.mark.skip
def test_run_planingfsi(runner):    
    results = runner.invoke(cli, ["run"])
    assert results.exit_code == 0


@pytest.mark.skip
@pytest.mark.skip()
def test_run_generate_mesh(runner):
    results = runner.invoke(cli, ["mesh"])
    assert results.exit_code == 0
