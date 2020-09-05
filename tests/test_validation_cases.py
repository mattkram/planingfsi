import os
import shutil
from pathlib import Path
from typing import Callable

import pytest
from click.testing import CliRunner
from planingfsi.cli import cli


VALIDATED_EXTENSION = ".validated"

RunCaseFunction = Callable[[str], Path]


@pytest.fixture()
def run_case(tmpdir: Path, validation_base_dir: Path) -> RunCaseFunction:
    """A function which is used to run a specific validation case.

    The case runner executes in in a temporary directory with input files and results copied into it from the base
    directory.

    """

    def f(case_name: str) -> Path:
        """Create a case runner for a specific case in a temporary directory and run planingfsi.

        Args:
            case_name: The name of the case directory within the validation base directory.

        Returns:
            The path to the temporary case directory.

        """

        cli_runner = CliRunner()
        case_base_dir = validation_base_dir / case_name
        for source in case_base_dir.glob("*"):
            destination = tmpdir / source.name
            try:
                shutil.copytree(source, destination)
            except NotADirectoryError:
                shutil.copyfile(source, destination)
        os.chdir(tmpdir)
        cli_runner.invoke(cli, ["run"])
        return Path(tmpdir)

    return f


@pytest.mark.parametrize("case_name", ("flat_plate",))
def test_run_validation_case(run_case: RunCaseFunction, case_name: str) -> None:

    validation_items = run_case(case_name).glob(f"*{VALIDATED_EXTENSION}")
    # For each validation item, recursively check all files exist and contents are identical
    for item in validation_items:
        new_item = item.with_suffix("")
        assert new_item.exists()
        for old_file in item.glob("*"):
            new_file = new_item / old_file.name
            assert new_file.exists()
            with old_file.open() as fp, new_file.open() as gp:
                assert fp.read() == gp.read()
