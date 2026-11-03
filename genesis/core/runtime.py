"""Process and path checks. This module must stay free of third-party imports."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    override = os.environ.get("GENESIS_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


def project_venv_python() -> Path:
    folder = "Scripts" if os.name == "nt" else "bin"
    name = "python.exe" if os.name == "nt" else "python"
    return repo_root() / ".venv" / folder / name


def assert_project_venv() -> None:
    """Exit unless this process is the repository virtual environment."""
    expected = project_venv_python()
    actual = Path(sys.executable)
    if _same_path(actual, expected):
        return
    raise SystemExit(
        "Genesis only runs inside the project virtual environment. "
        f"Use {expected}. Current interpreter: {actual}"
    )


def _same_path(actual: Path, expected: Path) -> bool:
    if os.path.normcase(os.path.normpath(str(actual))) == os.path.normcase(os.path.normpath(str(expected))):
        return True
    try:
        return os.path.normcase(str(actual.resolve())) == os.path.normcase(str(expected.resolve()))
    except OSError:
        return False
