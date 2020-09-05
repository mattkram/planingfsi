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
    """A function which can construct a case runner for a specific validation case.

    The case runner executes in in a temporary directory with input files and results copied into it from the base
    directory.

    """

    def f(case_name: str) -> Path:
        """Create a case runner for a specific case.

        Copies all input files from the base case into a temporary directory in order to compare the contents.

        Args:
            case_name: The name of the case directory within the validation base directory.

        Returns:
            The CLI runner object for planingfsi, which can be called via the invoke method.

        """

        runner = CliRunner()
        case_dir = validation_base_dir / case_name
        for source in case_dir.glob("*"):
            destination = tmpdir / source.name
            try:
                shutil.copytree(source, destination)
            except NotADirectoryError:
                shutil.copyfile(source, destination)
        os.chdir(tmpdir)
        runner.invoke(cli, ["run"])
        return Path(tmpdir)

    return f


@pytest.mark.parametrize("case_name", ("flat_plate",))
def test_run_validation_case(run_case: RunCaseFunction, case_name: str) -> None:

    validation_items = run_case(case_name).glob(f"*{VALIDATED_EXTENSION}")
    # For each validation item, recursively check all files exist and contents are identical
    for item in validation_items:
        new_item = item.with_suffix("")
        assert new_item.exists()
        # TODO: Expand tests once results become available
