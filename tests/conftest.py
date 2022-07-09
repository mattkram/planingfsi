import os
from pathlib import Path
from typing import Any
from typing import Dict

import pytest
from _pytest.config.argparsing import Parser
from _pytest.python import Function
from click.testing import CliRunner

from planingfsi.dictionary import load_dict_from_file

PROJECT_DIR = Path(__file__).parents[1]


@pytest.fixture()
def project_dir() -> Path:
    return PROJECT_DIR


@pytest.fixture()
def test_dir(project_dir: Path) -> Path:
    return project_dir / "tests"


@pytest.fixture()
def input_dir(test_dir: Path) -> Path:
    return test_dir / "input_files"


@pytest.fixture()
def validation_base_dir(test_dir: Path) -> Path:
    return test_dir / "validation_cases"


@pytest.fixture()
def test_dict(input_dir: Path) -> Dict[str, Any]:
    os.environ["HOME"] = "Dummy"
    return load_dict_from_file(input_dir / "testDict")


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def pytest_addoption(parser: Parser) -> None:
    parser.addoption("--slow", action="store_true", help="run the tests marked 'slow'")


def pytest_runtest_setup(item: Function) -> None:
    if "slow" in item.keywords and not item.config.getoption("--slow"):
        pytest.skip("need --slow option to run this test")


@pytest.fixture(autouse=True)
def _cleanup_globals() -> None:
    """Clear out class-global lists, which causes trouble when running the program multiple times
    during the same testing session.

    Todo:
        * This should be removed after the globals are factored out.

    """
    from planingfsi.fe.felib import Element
    from planingfsi.fe.felib import Node
    from planingfsi.fe.substructure import FlexibleSubstructure
    from planingfsi.fe.substructure import Substructure

    Node._Node__all = []  # type: ignore
    Element._Element__all = []  # type: ignore
    Substructure._Substructure__all = []  # type: ignore
    FlexibleSubstructure._FlexibleSubstructure__all = []  # type: ignore
