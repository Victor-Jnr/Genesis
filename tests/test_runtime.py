import os
import sys
from pathlib import Path

import pytest

from genesis.core.runtime import assert_project_venv, project_venv_python, repo_root


def test_repo_root_is_the_project() -> None:
    assert (repo_root() / "README.md").is_file()
    assert (repo_root() / "config" / "genesis.yaml").is_file()
    assert not (repo_root() / "writing-block.md").exists()


def test_project_venv_python_lives_in_repo() -> None:
    expected = project_venv_python()
    assert expected.parent.parent == repo_root() / ".venv"
    assert expected.name in {"python.exe", "python"}


def test_running_interpreter_is_the_project_venv() -> None:
    assert_project_venv()
    assert os.path.normcase(sys.executable) == os.path.normcase(str(project_venv_python()))


def test_other_interpreters_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "executable", str(Path("C:/Python/python.exe")))
    with pytest.raises(SystemExit, match="project virtual environment"):
        assert_project_venv()
