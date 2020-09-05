import os
from pathlib import Path
from typing import Any, Dict

import pytest
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
