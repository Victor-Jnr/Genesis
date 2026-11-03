import json
import time
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from genesis.api.app import build_context, create_app
from genesis.cognition.learning import covered_gap_reply
from genesis.cognition.planner import parse_plan
from genesis.core.configuration.settings import DatabaseConfig, LoggingConfig, ModelConfig, Settings
from genesis.core.model_interface.llama_cpp_backend import GenerationResult
from genesis.core.orchestrator.service import ObjectiveStateError
from genesis.storage.database import StepRecord


PLAN = json.dumps(
    {
        "steps": [
            {"title": "Inspect", "instruction": "Restate the request."},
            {"title": "Answer", "instruction": "Give the answer."},
        ]
    }
)


class ScriptedBackend:
    def __init__(self, replies: list[str]) -> None:
        self.path: Path | None = None
        self.replies = list(replies)
        self.messages: list[dict[str, str]] | None = None

    def load(self, path: Path, model: ModelConfig) -> None:
        self.path = path

    def unload(self) -> None:
        self.path = None

    def generate(self, messages, model, *, temperature=None, max_tokens=None, response_format=None) -> GenerationResult:
        self.messages = messages
        content = self.replies.pop(0) if self.replies else "done"
        return GenerationResult(content, 1, 1, 1.0, None if self.path is None else self.path.name)

    def stream(self, messages, model, *, temperature=None, max_tokens=None):
        yield "done"


def _context(tmp_path: Path, names: list[str], replies: list[str]):
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    for name in names:
        (model_dir / name).write_bytes(b"gguf")
    settings = Settings(
        model=ModelConfig(directory=str(model_dir), n_gpu_layers=0),
        database=DatabaseConfig(path=str(tmp_path / "genesis.db")),
        logging=LoggingConfig(directory=str(tmp_path / "logs"), level="WARNING"),
    )
    return build_context(tmp_path, settings, ScriptedBackend(replies))


def _success_replies() -> list[str]:
    return [covered_gap_reply(), PLAN, "The request is to add.", "The answer is 4.", "4"]


def test_a_repeated_inspect_finishes_the_step(tmp_path: Path) -> None:
    source = tmp_path / "genesis"
    source.mkdir()
    (source / "note.py").write_text("answer = 1\n", encoding="utf-8")
    plan = json.dumps({"steps": [{"title": "Read", "instruction": "Read the note."}]})
    inspect = json.dumps({"action": "inspect", "path": "genesis/note.py"})
    context = _context(tmp_path, ["demo.gguf"], [covered_gap_reply(), plan, inspect, inspect, "done"])
    context.models.startup()
    objective = context.orchestrator.submit("Read genesis/note.py.")
    context.orchestrator.run_available()
    stored, events = context.orchestrator.get(objective.id)
    assert stored.status == "completed"
    assert stored.error is None
    assert stored.steps[0].result == "answer = 1\n"
    assert stored.result == "done"
    assert not any(event.kind == "failed" for event in events)


def test_parse_plan_reads_json_and_strips_thinking() -> None:
    steps = parse_plan('<think>draft</think>\n```json\n' + PLAN + "\n```")
    assert [step.title for step in steps] == ["Inspect", "Answer"]
    assert steps[0].result is None


def test_a_reply_that_asks_the_user_is_not_the_step_result(tmp_path: Path) -> None:
    replies = [
        covered_gap_reply(),
        PLAN,
        "I am ready to help. Please provide the necessary details.",
        "The request is to add.",
        "The answer is 4.",
        "4",
    ]
    context = _context(tmp_path, ["demo.gguf"], replies)
    context.models.startup()
    objective = context.orchestrator.submit("Add 2 and 2")
    context.orchestrator.run_available()
    stored, _events = context.orchestrator.get(objective.id)
    assert stored.status == "completed"
    assert [step.result for step in stored.steps] == ["The request is to add.", "The answer is 4."]


def test_a_broken_json_action_is_asked_again(tmp_path: Path) -> None:
    replies = [covered_gap_reply(), PLAN, "{", "The request is to add.", "The answer is 4.", "4"]
    context = _context(tmp_path, ["demo.gguf"], replies)
    context.models.startup()
    objective = context.orchestrator.submit("Add 2 and 2")
    context.orchestrator.run_available()
    stored, _events = context.orchestrator.get(objective.id)
    assert stored.status == "completed"
    assert stored.result == "4"
    assert [step.result for step in stored.steps] == ["The request is to add.", "The answer is 4."]


def test_objective_runs_to_completion_and_records_events(tmp_path: Path) -> None:
    context = _context(tmp_path, ["demo.gguf"], _success_replies())
    context.models.startup()
    objective = context.orchestrator.submit("Add 2 and 2")
    context.orchestrator.run_available()
    stored, events = context.orchestrator.get(objective.id)
    assert stored.status == "completed"
    assert stored.result == "4"
    assert [step.result for step in stored.steps] == ["The request is to add.", "The answer is 4."]
    assert [event.kind for event in events] == [
        "created",
        "planning",
        "gap",
        "planned",
        "step",
        "step",
        "completed",
    ]


def test_planner_asks_again_when_the_first_reply_is_not_json(tmp_path: Path) -> None:
    context = _context(
        tmp_path,
        ["demo.gguf"],
        [covered_gap_reply(), "pong", PLAN, "The request is to add.", "The answer is 4.", "4"],
    )
    context.models.startup()
    objective = context.orchestrator.submit("Reply with the single word pong.")
    context.orchestrator.run_available()
    stored, _events = context.orchestrator.get(objective.id)
    assert stored.status == "completed"
    assert stored.result == "4"


def test_invalid_plan_fails_the_objective(tmp_path: Path) -> None:
    context = _context(tmp_path, ["demo.gguf"], [covered_gap_reply(), "not a plan"])
    context.models.startup()
    objective = context.orchestrator.submit("Do something")
    context.orchestrator.run_available()
    stored, events = context.orchestrator.get(objective.id)
    assert stored.status == "failed"
    assert stored.error is not None
    assert "JSON" in stored.error
    assert events[-1].kind == "failed"


def test_pause_holds_work_until_resume(tmp_path: Path) -> None:
    context = _context(tmp_path, ["demo.gguf"], _success_replies())
    context.models.startup()
    objective = context.orchestrator.submit("Add 2 and 2")
    paused = context.orchestrator.pause(objective.id)
    context.orchestrator.run_available()
    assert context.orchestrator.get(objective.id)[0].status == "paused"
    assert paused.status == "paused"
    context.orchestrator.resume(objective.id)
    context.orchestrator.run_available()
    assert context.orchestrator.get(objective.id)[0].status == "completed"
    try:
        context.orchestrator.pause(objective.id)
    except ObjectiveStateError as exc:
        assert "completed" in str(exc)
    else:
        raise AssertionError("pausing a completed objective should fail")


def test_restart_continues_an_executing_objective(tmp_path: Path) -> None:
    context = _context(tmp_path, ["demo.gguf"], ["The sum is 4.", "4"])
    context.models.startup()
    created = context.db.create_objective("Add 2 and 2")
    context.db.save_objective(
        replace(
            created,
            status="executing",
            current_step=1,
            steps=(
                StepRecord("Add", "Add the numbers.", "4"),
                StepRecord("State", "State the sum.", None),
            ),
        )
    )
    restarted = _context(tmp_path, ["demo.gguf"], ["The sum is 4.", "4"])
    restarted.models.startup()
    restarted.orchestrator.run_available()
    stored, _events = restarted.orchestrator.get(created.id)
    assert stored.status == "completed"
    assert stored.steps[0].result == "4"
    assert stored.steps[1].result == "The sum is 4."
    assert stored.result == "4"


def test_missing_model_waits_until_one_is_selected(tmp_path: Path) -> None:
    context = _context(tmp_path, ["a.gguf", "b.gguf"], _success_replies())
    context.models.startup()
    objective = context.orchestrator.submit("Add 2 and 2")
    context.orchestrator.run_available()
    waiting, _events = context.orchestrator.get(objective.id)
    assert waiting.status == "waiting"
    context.models.select("b.gguf")
    context.orchestrator.notify_model()
    context.orchestrator.run_available()
    assert context.orchestrator.get(objective.id)[0].status == "completed"


def test_objective_http_completes_in_the_background(tmp_path: Path) -> None:
    context = _context(tmp_path, ["demo.gguf"], _success_replies())
    client = TestClient(create_app(context))
    with client:
        created = client.post("/objectives", json={"request": "Add 2 and 2"})
        assert created.status_code == 200
        objective_id = created.json()["id"]
        body: dict[str, object] = {}
        for _ in range(50):
            body = client.get(f"/objectives/{objective_id}").json()
            if body["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)
        assert body["status"] == "completed"
        assert body["result"] == "4"
        assert client.get("/objectives/missing").status_code == 404
        listed = client.get("/objectives")
        assert listed.status_code == 200
        assert listed.json()[0]["id"] == objective_id
        assert listed.json()[0]["events"] is None
