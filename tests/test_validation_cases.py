import os
import shutil
from pathlib import Path
from typing import Callable, Tuple

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

    """

    def f(case_name: str) -> Tuple[Path, Path]:
        """Create a case runner for a specific case in a temporary directory and run planingfsi.

        Args:
            case_name: The name of the case directory within the validation base directory.

        Returns:
            The path to the temporary case directory.

        """
        case_base_dir = validation_base_dir / case_name
        for source in case_base_dir.glob("*"):
            if source.suffix == VALIDATED_EXTENSION:
                continue  # Don't copy validated files to the temporary directory
            destination = tmpdir / source.name
            try:
                shutil.copytree(source, destination)
            except NotADirectoryError:
                shutil.copyfile(source, destination)
        os.chdir(tmpdir)

        cli_runner = CliRunner()
        assert cli_runner.invoke(cli, ["mesh"]).exit_code == 0
        assert cli_runner.invoke(cli, ["run"]).exit_code == 0

        # This is the leftover module-centric way to run the CLI invoked above
        # config.load_from_file("configDict")
        #
        # mesh = Mesh()
        # exec(Path(config.path.mesh_dict_dir).open("r").read())
        #
        # mesh.display()
        # mesh.plot()
        # mesh.write()
        #
        # simulation = Simulation()
        # simulation.load_input_files()
        # simulation.run()

        return case_base_dir, Path(tmpdir)

    return f


@pytest.mark.parametrize(
    "case_name",
    (
        "flat_plate",
        "stepped_planing_plate",
        "flexible_membrane",
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
            assert f_values == pytest.approx(g_values)
