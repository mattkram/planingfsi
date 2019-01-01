import os
import shutil

import pytest
from click.testing import CliRunner

from planingfsi.cli import main


@pytest.fixture()
def cli_runner(validation_base_dir):
    """Return a factory function which can return a case runner in a temporary directory with input files and results
    copied into it.

    Returns:
        function: A function accepting a string which identifies the folder within "validation_cases" to run.

    """

    def f(case_name):
        runner = CliRunner()
        with runner.isolated_filesystem():
            for item in os.listdir(validation_base_dir / case_name):
                source = validation_base_dir / case_name / item
                if source.is_dir():
                    shutil.copytree(source, item)
                else:
                    shutil.copy(source, item)

            return runner

    return f


@pytest.mark.parametrize("case_name", ("flat_plate",))
def test_flat_plate(cli_runner, case_name):
    runner = cli_runner(case_name)
    runner.invoke(main, ["run"])
