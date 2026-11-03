import json
import os
import subprocess
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from genesis.api.app import build_context, create_app
from genesis.cognition.learning import covered_gap_reply
from genesis.core.configuration.settings import (
    DatabaseConfig,
    DevelopmentConfig,
    LoggingConfig,
    ModelConfig,
    Settings,
)
from genesis.core.model_interface.llama_cpp_backend import GenerationResult
from genesis.core.runtime import repo_root
from genesis.development.service import link_project_venv

PLAN = json.dumps({"steps": [{"title": "Edit", "instruction": "Change the sample value."}]})


class ScriptedBackend:
    def __init__(self, replies: list[str]) -> None:
        self.path: Path | None = None
        self.replies = list(replies)

    def load(self, path: Path, model: ModelConfig) -> None:
        self.path = path

    def unload(self) -> None:
        self.path = None

    def generate(self, messages, model, *, temperature=None, max_tokens=None) -> GenerationResult:
        content = self.replies.pop(0) if self.replies else "done"
        return GenerationResult(content, 1, 1, 1.0, None if self.path is None else self.path.name)

    def stream(self, messages, model, *, temperature=None, max_tokens=None):
        yield "done"


def _settings(tmp_path: Path) -> Settings:
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "demo.gguf").write_bytes(b"gguf")
    return Settings(
        model=ModelConfig(directory=str(model_dir), n_gpu_layers=0),
        database=DatabaseConfig(path=str(tmp_path / "genesis.db")),
        logging=LoggingConfig(directory=str(tmp_path / "logs"), level="WARNING"),
        development=DevelopmentConfig(
            candidates_directory=str(tmp_path / "candidates"),
            test_timeout_seconds=30,
        ),
    )


def _sample(tmp_path: Path, source: str = "value = 1\n") -> None:
    (tmp_path / "genesis").mkdir(exist_ok=True)
    (tmp_path / "genesis" / "sample.py").write_text(source, encoding="utf-8")


def _runner(directory: Path, repo: Path, timeout: float) -> tuple[int, str]:
    text = (directory / "genesis" / "sample.py").read_text(encoding="utf-8")
    if "value = 2" not in text:
        return 1, "candidate did not contain the replacement"
    return 0, "1 passed"


def test_inspect_reads_a_file_and_rejects_escape(tmp_path: Path) -> None:
    _sample(tmp_path)
    context = build_context(tmp_path, _settings(tmp_path), ScriptedBackend([]), candidate_runner=_runner)
    assert context.development.inspect("genesis/sample.py") == "value = 1\n"
    listed = context.development.inspect("genesis")
    assert "genesis/sample.py" in listed
    escaped = context.development.execute(None, _action("inspect", path="../genesis/sample.py"))
    assert escaped.startswith("Modification failed:")
    blocked = context.development.execute(None, _action("inspect", path="private/writing-block.md"))
    assert blocked.startswith("Modification failed:")


def test_proposal_keeps_the_live_file_and_writes_a_candidate(tmp_path: Path) -> None:
    _sample(tmp_path)
    context = build_context(tmp_path, _settings(tmp_path), ScriptedBackend([]), candidate_runner=_runner)
    proposal, candidate = context.development.propose(
        None,
        "genesis/sample.py",
        "Count from two.",
        "value = 1",
        "value = 2",
    )
    assert (tmp_path / "genesis" / "sample.py").read_text(encoding="utf-8") == "value = 1\n"
    copied = Path(candidate.directory) / "genesis" / "sample.py"
    assert copied.read_text(encoding="utf-8") == "value = 2\n"
    assert proposal.base_sha256
    broken = context.development.execute(
        None,
        _action("propose", path="genesis/sample.py", old="value = 1", text="value = (", rationale="Break it."),
    )
    assert "valid Python" in broken
    tested = context.development.test(candidate.id)
    assert tested.status == "passed"
    comparison = context.development.compare(candidate.id)
    assert "Live file matches the previous version: True" in comparison
    assert "Candidate contains the replacement: True" in comparison
    retained = context.development.retain(candidate.id)
    assert retained.status == "retained"
    assert "retained" in context.development.archive()


def test_a_failed_candidate_stays_recoverable(tmp_path: Path) -> None:
    _sample(tmp_path)

    def fail(directory: Path, repo: Path, timeout: float) -> tuple[int, str]:
        return 1, "1 failed"

    context = build_context(tmp_path, _settings(tmp_path), ScriptedBackend([]), candidate_runner=fail)
    _proposal, candidate = context.development.propose(None, "genesis/sample.py", "Try two.", "value = 1", "value = 2")
    tested = context.development.test(candidate.id)
    assert tested.status == "failed"
    with pytest.raises(Exception, match="Only a passing candidate"):
        context.development.retain(candidate.id)
    message = context.development.execute(None, _action("retain", package=candidate.id))
    assert message.startswith("Modification failed:")
    assert (tmp_path / "genesis" / "sample.py").read_text(encoding="utf-8") == "value = 1\n"


def test_objective_proposes_without_replacing_the_live_file(tmp_path: Path) -> None:
    _sample(tmp_path)
    inspect = json.dumps({"action": "inspect", "path": "genesis/sample.py"})
    propose = json.dumps(
        {
            "action": "propose",
            "path": "genesis/sample.py",
            "old": "value = 1",
            "text": "value = 2",
            "rationale": "Count from two.",
        }
    )
    context = build_context(
        tmp_path,
        _settings(tmp_path),
        ScriptedBackend([covered_gap_reply(), PLAN, inspect, propose, "Candidate ready.", "Candidate ready."]),
        candidate_runner=_runner,
    )
    context.models.startup()
    objective = context.orchestrator.submit("Change the sample value and keep the current file.")
    context.orchestrator.run_available()
    stored, events = context.orchestrator.get(objective.id)
    assert stored.status == "completed"
    assert [event.kind for event in events if event.kind in {"inspect", "propose"}] == ["inspect", "propose"]
    assert (tmp_path / "genesis" / "sample.py").read_text(encoding="utf-8") == "value = 1\n"
    client = TestClient(create_app(context))
    with client:
        listed = client.get("/candidates")
        assert listed.status_code == 200
        body = listed.json()[0]
        assert body["status"] == "created"
        assert body["path"] == "genesis/sample.py"
        missing = client.get("/candidates/missing")
        assert missing.status_code == 404
        source = client.get("/source", params={"path": "genesis/sample.py"})
        assert source.status_code == 200
        assert source.json()["text"] == "value = 1\n"


def test_candidate_python_is_the_project_virtual_environment(tmp_path: Path) -> None:
    package = tmp_path / "genesis" / "core"
    package.mkdir(parents=True)
    (tmp_path / "genesis" / "__init__.py").write_text("", encoding="utf-8")
    (package.parent / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    source = repo_root() / "genesis" / "core" / "runtime.py"
    (package / "runtime.py").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    link_project_venv(repo_root(), tmp_path)
    folder = "Scripts" if os.name == "nt" else "bin"
    name = "python.exe" if os.name == "nt" else "python"
    completed = subprocess.run(
        [str(tmp_path / ".venv" / folder / name), "-c", "from genesis.core.runtime import assert_project_venv; assert_project_venv()"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _action(name: str, **kwargs):
    from genesis.tools.research.actions import ResearchAction

    return ResearchAction(name, **kwargs)
