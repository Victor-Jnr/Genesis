from pathlib import Path

import pytest

from genesis.core.configuration.settings import load_settings
from genesis.core.model_interface.discovery import list_gguf, model_file, model_key, resolve_model_choice


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GENESIS_MODEL_DIR", raising=False)
    monkeypatch.delenv("GENESIS_MODEL_FILENAME", raising=False)
    monkeypatch.delenv("GENESIS_CONFIG", raising=False)


def test_shipped_config_loads_phase0_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    settings = load_settings()
    assert settings.model.directory == "models"
    assert settings.model.filename is None
    assert settings.model.n_gpu_layers == 0
    assert settings.model.context_length == 4096
    assert settings.server.host == "127.0.0.1"
    assert settings.database.path == "storage/genesis.db"
    assert settings.research.enabled is True
    assert settings.research.allow_private_hosts is False
    assert settings.research.max_actions == 4


def test_yaml_and_env_override_without_code_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    config = tmp_path / "genesis.yaml"
    config.write_text(
        "model:\n  directory: custom-models\n  filename: chosen.gguf\n  n_gpu_layers: -1\n",
        encoding="utf-8",
    )
    settings = load_settings(config)
    assert settings.model.filename == "chosen.gguf"
    assert settings.model.n_gpu_layers == -1
    monkeypatch.setenv("GENESIS_MODEL_DIR", "from-env")
    monkeypatch.setenv("GENESIS_MODEL_FILENAME", "from-env.gguf")
    overridden = load_settings(config)
    assert overridden.model.directory == "from-env"
    assert overridden.model.filename == "from-env.gguf"


def test_list_gguf_includes_nested_files(tmp_path: Path) -> None:
    (tmp_path / "one.gguf").write_bytes(b"a")
    (tmp_path / "notes.txt").write_text("nope", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "hidden.gguf").write_bytes(b"b")
    found = list_gguf(tmp_path)
    assert [model_key(tmp_path, path) for path in found] == ["nested/hidden.gguf", "one.gguf"]


def test_model_file_accepts_a_nested_path_and_rejects_parents(tmp_path: Path) -> None:
    nested = tmp_path / "vendor" / "repo"
    nested.mkdir(parents=True)
    target = nested / "demo.gguf"
    target.write_bytes(b"a")
    assert model_file(tmp_path, "vendor/repo/demo.gguf") == target.resolve()
    with pytest.raises(ValueError):
        model_file(tmp_path, "../demo.gguf")
    with pytest.raises(ValueError):
        model_file(tmp_path, "vendor/../../demo.gguf")


def test_resolve_model_choice_rules(tmp_path: Path) -> None:
    one = tmp_path / "only.gguf"
    files = [one]
    auto = resolve_model_choice(files, directory=tmp_path, configured_filename=None, remembered_filename=None)
    assert auto.path == one
    assert auto.remember_as == "only.gguf"

    other = tmp_path / "other.gguf"
    many = resolve_model_choice(
        [one, other],
        directory=tmp_path,
        configured_filename=None,
        remembered_filename=None,
    )
    assert many.path is None
    assert many.detail is not None
    assert "Multiple GGUF" in many.detail

    remembered = resolve_model_choice(
        [one, other],
        directory=tmp_path,
        configured_filename=None,
        remembered_filename="other.gguf",
    )
    assert remembered.path == other

    missing = resolve_model_choice(
        files,
        directory=tmp_path,
        configured_filename="missing.gguf",
        remembered_filename="only.gguf",
    )
    assert missing.path is None
    assert missing.detail is not None
    assert "missing.gguf" in missing.detail

    empty = resolve_model_choice([], directory=tmp_path, configured_filename=None, remembered_filename=None)
    assert empty.path is None
    assert empty.detail is not None
    assert "No .gguf" in empty.detail

    nested = tmp_path / "vendor" / "only.gguf"
    by_basename = resolve_model_choice(
        [nested],
        directory=tmp_path,
        configured_filename=None,
        remembered_filename="only.gguf",
    )
    assert by_basename.path == nested
    ambiguous = resolve_model_choice(
        [tmp_path / "a" / "only.gguf", tmp_path / "b" / "only.gguf"],
        directory=tmp_path,
        configured_filename="only.gguf",
        remembered_filename=None,
    )
    assert ambiguous.path is None
    assert ambiguous.detail is not None
    assert "Multiple GGUF" in ambiguous.detail
