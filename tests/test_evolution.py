import json
import re
from pathlib import Path

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
from genesis.evolution.service import ArchiveError

PLAN = json.dumps({"steps": [{"title": "Edit", "instruction": "Change the sample value."}]})


class ScriptedBackend:
    def __init__(self, replies: list[str]) -> None:
        self.path: Path | None = None
        self.replies = list(replies)

    def load(self, path: Path, model: ModelConfig) -> None:
        self.path = path

    def unload(self) -> None:
        self.path = None

    def generate(self, messages, model, *, temperature=None, max_tokens=None, response_format=None) -> GenerationResult:
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
        development=DevelopmentConfig(candidates_directory=str(tmp_path / "candidates"), test_timeout_seconds=30),
    )


def _sample(tmp_path: Path) -> None:
    (tmp_path / "genesis").mkdir(exist_ok=True)
    (tmp_path / "genesis" / "sample.py").write_text("value = 1\n", encoding="utf-8")


def _runner(directory: Path, repo: Path, timeout: float):
    text = (directory / "genesis" / "sample.py").read_text(encoding="utf-8")
    if "value = 3" in text:
        return 0, "1 passed", 5.0
    if "value = 2" in text:
        return 0, "1 passed", 40.0
    return 1, "failed", 1.0


def _context(tmp_path: Path, replies: list[str] | None = None, runner=_runner):
    _sample(tmp_path)
    return build_context(tmp_path, _settings(tmp_path), ScriptedBackend(replies or []), candidate_runner=runner)


def test_selection_keeps_every_generation_and_prefers_a_faster_success(tmp_path: Path) -> None:
    context = _context(tmp_path)
    archive = context.archive
    _proposal, slow_candidate, slow = archive.propose(None, "genesis/sample.py", "Count from two.", "value = 1", "value = 2")
    archive.evaluate(slow.id)
    archive.retain(slow.id)
    _proposal, child, fast = archive.propose(
        None,
        "genesis/sample.py",
        "Count from three.",
        "value = 2",
        "value = 3",
        slow.id,
    )
    archive.evaluate(fast.id)
    archive.retain(fast.id)
    _proposal, failed, rejected = archive.propose(None, "genesis/sample.py", "Try nine.", "value = 1", "value = 9")
    failed_report = archive.evaluate(rejected.id)
    assert "success 0.0" in failed_report
    assert context.db.get_generation(rejected.id).status == "rejected"

    report = archive.select("tests")
    assert report.startswith(f"Benchmark: tests\nSelected {fast.id}")
    assert slow.id in report
    assert rejected.id in report
    assert "Other generations stay in the archive" in report
    assert context.db.latest_selection("tests").generation_id == fast.id

    listed = archive.lineage()
    assert "retained" in listed
    assert "rejected" in listed
    assert f"Selected parent: {fast.id}" in listed
    assert archive.revisit(slow.id, "genesis/sample.py") == "value = 2\n"
    assert archive.revisit(fast.id, "genesis/sample.py") == "value = 3\n"
    assert (tmp_path / "genesis" / "sample.py").read_text(encoding="utf-8") == "value = 1\n"
    assert (Path(slow_candidate.directory) / "genesis" / "sample.py").read_text(encoding="utf-8") == "value = 2\n"
    assert (Path(child.directory) / "genesis" / "sample.py").read_text(encoding="utf-8") == "value = 3\n"
    assert (Path(failed.directory) / "genesis" / "sample.py").read_text(encoding="utf-8") == "value = 9\n"


def test_select_with_a_candidate_id_scores_the_live_baseline(tmp_path: Path) -> None:
    from genesis.tools.research.actions import ResearchAction

    context = _context(tmp_path)
    generation = context.archive.propose(None, "genesis/sample.py", "Count from two.", "value = 1", "value = 2")[2]
    report = context.archive.execute(None, ResearchAction("select", package=generation.candidate_id))
    assert f"Selected {generation.id}" in report
    assert "success 1.0" in report
    assert "success 0.0" in report
    assert (tmp_path / "genesis" / "sample.py").read_text(encoding="utf-8") == "value = 1\n"


def test_an_earlier_tie_remains_the_parent(tmp_path: Path) -> None:
    def tied(directory: Path, repo: Path, timeout: float):
        return 0, "1 passed", 10.0

    context = _context(tmp_path, runner=tied)
    first = context.archive.propose(None, "genesis/sample.py", "First.", "value = 1", "value = 2")[2]
    second = context.archive.propose(None, "genesis/sample.py", "Second.", "value = 1", "value = 2")[2]
    context.archive.evaluate(first.id)
    context.archive.evaluate(second.id)
    context.archive.select()
    assert context.db.latest_selection("tests").generation_id == first.id


def test_a_different_benchmark_does_not_choose_the_parent(tmp_path: Path) -> None:
    context = _context(tmp_path)
    generation = context.archive.propose(None, "genesis/sample.py", "Count from two.", "value = 1", "value = 2")[2]
    context.archive.evaluate(generation.id, "other")
    try:
        context.archive.select("tests")
        raised = False
    except ArchiveError:
        raised = True
    assert raised
    context.archive.evaluate(generation.id, "tests")
    assert context.archive.select("tests").startswith(f"Benchmark: tests\nSelected {generation.id}")


def test_archive_routes_are_readable_after_a_proposal(tmp_path: Path) -> None:
    context = _context(tmp_path)
    client = TestClient(create_app(context))
    with client:
        created = client.post(
            "/proposals",
            json={"path": "genesis/sample.py", "rationale": "Count from two.", "old": "value = 1", "text": "value = 2"},
        )
        assert created.status_code == 200
        generation_id = created.json()["generation_id"]
        evaluated = client.post(f"/candidates/{created.json()['id']}/evaluate")
        assert evaluated.status_code == 200
        assert evaluated.json()["success"] == 1.0
        assert evaluated.json()["elapsed_ms"] == 40.0
        selected = client.post("/archive/select")
        assert selected.status_code == 200
        assert selected.json()["id"] == generation_id
        listed = client.get("/archive")
        assert listed.status_code == 200
        assert {item["status"] for item in listed.json()} >= {"baseline", "open"}
        source = client.get(f"/archive/{generation_id}/source", params={"path": "genesis/sample.py"})
        assert source.status_code == 200
        assert source.json()["text"] == "value = 2\n"
        missing = client.get("/archive/missing")
        assert missing.status_code == 404
        empty = client.post("/archive/select", params={"benchmark": "missing"})
        assert empty.status_code == 400


def test_objective_proposal_opens_a_generation(tmp_path: Path) -> None:
    propose = json.dumps(
        {
            "action": "propose",
            "path": "genesis/sample.py",
            "old": "value = 1",
            "text": "value = 2",
            "rationale": "Count from two.",
        }
    )
    context = _context(
        tmp_path,
        [covered_gap_reply(), PLAN, propose, "Candidate ready.", "Candidate ready."],
    )
    context.models.startup()
    objective = context.orchestrator.submit("Change the sample and keep the current file.")
    context.orchestrator.run_available()
    stored, events = context.orchestrator.get(objective.id)
    assert stored.status == "completed"
    assert [event.kind for event in events if event.kind == "propose"] == ["propose"]
    generations = context.db.list_generations()
    assert [item.status for item in generations] == ["baseline", "open"]
    assert (tmp_path / "genesis" / "sample.py").read_text(encoding="utf-8") == "value = 1\n"


def test_a_passing_selection_retains_the_child_and_stops(tmp_path: Path) -> None:
    propose = json.dumps(
        {
            "action": "propose",
            "path": "genesis/sample.py",
            "old": "value = 1",
            "text": "value = 2",
            "rationale": "Count from two.",
        }
    )
    plan = json.dumps(
        {
            "steps": [
                {"title": "Edit", "instruction": "Propose a candidate and select it."},
                {"title": "Again", "instruction": "Inspect the file again."},
            ]
        }
    )

    class SelectingBackend(ScriptedBackend):
        def generate(self, messages, model, *, temperature=None, max_tokens=None, response_format=None) -> GenerationResult:
            content = self.replies.pop(0) if self.replies else "inspected again"
            if content == "SELECT":
                blob = "\n".join(str(item.get("content", "")) for item in messages)
                found = re.search(r"Generation ([0-9a-f-]{36})", blob)
                assert found is not None
                content = json.dumps({"action": "select", "name": found.group(1)})
            return GenerationResult(content, 1, 1, 1.0, None if self.path is None else self.path.name)

    context = _context(tmp_path)
    context.models.backend = SelectingBackend([covered_gap_reply(), plan, propose, "SELECT", "inspected again"])
    context.models.startup()
    objective = context.orchestrator.submit("Change the sample through a candidate.")
    context.orchestrator.run_available()
    stored, events = context.orchestrator.get(objective.id)
    assert stored.status == "completed"
    assert stored.error is None
    assert "Selected " in (stored.result or "")
    assert "success 1.0" in (stored.result or "")
    assert "Retained generation" in (stored.result or "")
    assert stored.steps[1].result is None
    assert "inspected again" not in (stored.result or "")
    generations = context.db.list_generations()
    assert [item.status for item in generations] == ["baseline", "retained"]
    assert (tmp_path / "genesis" / "sample.py").read_text(encoding="utf-8") == "value = 1\n"
    assert any(event.kind == "retained" for event in events)
    assert not any(event.kind == "failed" for event in events)
