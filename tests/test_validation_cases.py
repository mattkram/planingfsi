import os
import shutil

import pytest
from click.testing import CliRunner

from planingfsi.cli import main


@pytest.fixture()
def validation_case_runner(tmpdir, validation_base_dir):
    """Return a factory function which can return a case runner in a temporary directory with input files and results
    copied into it."""

    def f(case_name):
        for item in os.listdir(validation_base_dir / case_name):
            source = validation_base_dir / case_name / item
            destination = tmpdir / item
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy(source, destination)

        os.chdir(tmpdir)

        runner = CliRunner()
        return runner

    return f


def test_flat_plate(validation_case_runner):
    runner = validation_case_runner("flat_plate")
    runner.invoke(main, ["run"])
