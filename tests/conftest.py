from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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
def test_dict(input_dir: Path) -> dict[str, Any]:
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
