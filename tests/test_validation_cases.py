import os
import shutil
from pathlib import Path
from typing import Callable

import pytest
from click.testing import CliRunner
from planingfsi.cli import cli


CliRunnerMaker = Callable[[str], CliRunner]


@pytest.fixture()
def cli_runner(tmpdir: Path, validation_base_dir: Path) -> CliRunnerMaker:
    """Return a factory function which can return a case runner in a temporary directory with input
    files and results copied into it.

    Returns:
        function: A function accepting a string which identifies the folder within
        "validation_cases" to run.

    """

    def f(case_name: str) -> CliRunner:
        runner = CliRunner()
        for source in (validation_base_dir / case_name).glob("*"):
            destination = tmpdir / source.name
            if source.is_dir():
                shutil.copytree(str(source), str(destination))
            else:
                shutil.copy(str(source), str(destination))
        os.chdir(str(tmpdir))
        return runner

    return f


@pytest.mark.skip
@pytest.mark.parametrize("case_name", ("flat_plate",))
def test_flat_plate(cli_runner: CliRunnerMaker, case_name: str) -> None:
    runner = cli_runner(case_name)
    runner.invoke(cli, ["run"])

    validation_extension = ".validated"
    validation_items = [f for f in os.listdir(".") if f.endswith(validation_extension)]

    # For each validation item, recursively check all files exist and contents are identical
    for item in validation_items:
        new_item = item.replace(validation_extension, "")
        assert Path(new_item).exists()
        # TODO: Expand tests once results become available
