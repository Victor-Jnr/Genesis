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
    if os.path.normcase(str(actual)) != os.path.normcase(str(expected)):
        raise SystemExit(
            "Genesis only runs inside the project virtual environment. "
            f"Use {expected}. Current interpreter: {actual}"
        )
