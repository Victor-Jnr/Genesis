import pytest

from genesis.core.runtime import assert_project_venv


def pytest_configure(config: pytest.Config) -> None:
    try:
        assert_project_venv()
    except SystemExit as exc:
        raise pytest.UsageError(str(exc)) from exc
