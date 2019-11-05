import os
from pathlib import Path

import pytest
from click.testing import CliRunner
from planingfsi.dictionary import load_dict_from_file

PROJECT_DIR = Path(__file__).parents[1]


@pytest.fixture()
def project_dir():
    return PROJECT_DIR


@pytest.fixture()
def test_dir(project_dir):
    return project_dir / "tests"


@pytest.fixture()
def input_dir(test_dir):
    return test_dir / "input_files"


@pytest.fixture()
def validation_base_dir(test_dir):
    return test_dir / "validation_cases"


@pytest.fixture()
def test_dict(input_dir):
    os.environ["HOME"] = "Dummy"
    dict_ = Dictionary(from_file=str(input_dir / "testDict"))


@pytest.fixture()
def runner():
    return CliRunner()
