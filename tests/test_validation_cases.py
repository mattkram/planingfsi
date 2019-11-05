import os
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner
from planingfsi.cli import cli


@pytest.fixture()
def cli_runner(tmpdir, validation_base_dir):
    """Return a factory function which can return a case runner in a temporary directory with input files and results
    copied into it.

    Returns:
        function: A function accepting a string which identifies the folder within "validation_cases" to run.

    """

    def f(case_name):
        runner = CliRunner()
        for item in os.listdir(validation_base_dir / case_name):
            source = validation_base_dir / case_name / item
            destination = tmpdir / item
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy(source, destination)
        os.chdir(tmpdir)
        return runner

    return f


@pytest.mark.skip
@pytest.mark.parametrize("case_name", ("flat_plate",))
def test_flat_plate(cli_runner, case_name):
    runner = cli_runner(case_name)
    runner.invoke(cli, ["run"])

    validation_extension = ".validated"
    validation_items = [f for f in os.listdir(".") if f.endswith(validation_extension)]

    # For each validation item, recursively check all files exist and contents are identical
    for item in validation_items:
        new_item = item.replace(validation_extension, "")
        assert Path(new_item).exists()
        # TODO: Expand tests once results become available
