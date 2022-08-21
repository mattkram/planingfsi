from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable
from typing import Tuple

import pytest
from click.testing import CliRunner

from planingfsi.cli import cli

VALIDATED_EXTENSION = ".validated"

RunCaseFunction = Callable[[str], Tuple[Path, Path]]


@pytest.fixture()
def run_case(tmpdir: Path, validation_base_dir: Path) -> RunCaseFunction:
    """A function which is used to run a specific validation case.

    The case runner executes in in a temporary directory with input files and results copied into it
    from the base directory.

    Args:
        tmpdir: A temporary case directory in which to run.
        validation_base_dir: The base path holding the validation cases.

    Returns:
        function: A function accepting a string which identifies the folder within
        "validation_cases" to run.

    """

    def f(case_name: str) -> tuple[Path, Path]:
        """Copy all input files from the base directory into the case directory and run the `mesh`
        and `run` CLI subcommands.

        Args:
            case_name: The name of the case directory within the validation base directory.

        Returns:
            The paths to the original and new case directories.

        """
        orig_case_dir = validation_base_dir / case_name
        new_case_dir = Path(tmpdir)
        for source in orig_case_dir.glob("*"):
            if source.suffix == VALIDATED_EXTENSION:
                continue  # Don't copy validated files to the temporary directory
            destination = new_case_dir / source.name
            try:
                shutil.copytree(source, destination)
            except NotADirectoryError:
                shutil.copyfile(source, destination)
        os.chdir(new_case_dir)

        cli_runner = CliRunner()
        mesh_result = cli_runner.invoke(cli, ["mesh"], catch_exceptions=False)
        assert mesh_result.exit_code == 0

        run_result = cli_runner.invoke(cli, ["run"], catch_exceptions=False)
        assert run_result.exit_code == 0

        return orig_case_dir, new_case_dir

    return f


@pytest.mark.parametrize(
    "case_name",
    (
        "flat_plate",
        "stepped_planing_plate",
        pytest.param("flexible_membrane", marks=pytest.mark.slow),
        pytest.param("sprung_plate", marks=pytest.mark.slow),
    ),
)
def test_run_validation_case(run_case: RunCaseFunction, case_name: str) -> None:
    """For each results directory marked with a '.validated' suffix, check that all newly calculated
    files exist and contents are identical."""

    orig_case_dir, new_case_dir = run_case(case_name)
    for orig_results_dir in orig_case_dir.glob(f"*{VALIDATED_EXTENSION}"):
        new_results_dir = (new_case_dir / orig_results_dir.name).with_suffix("")
        for orig_results_file in orig_results_dir.iterdir():
            new_results_file = new_results_dir / orig_results_file.name
            assert_files_almost_equal(orig_results_file, new_results_file)


def assert_files_almost_equal(orig_file: Path, new_file: Path) -> None:
    """Read two files line-by-line, convert any float-like number to a float in each line, and then
    compare each list of floats using pytest.approx."""
    with orig_file.open() as fp, new_file.open() as gp:
        for f_line, g_line in zip(fp.readlines(), gp.readlines()):
            f_values, g_values = [], []
            for f_str, g_str in zip(f_line.split(), g_line.split()):
                try:
                    f_val = float(f_str)
                    g_val = float(g_str)
                except ValueError:
                    continue
                f_values.append(f_val)
                g_values.append(g_val)
            assert f_values == pytest.approx(g_values, rel=1e-3, abs=1e-6)
