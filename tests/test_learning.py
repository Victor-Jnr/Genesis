import json
from pathlib import Path

from fastapi.testclient import TestClient

from genesis.api.app import build_context, create_app
from genesis.cognition.learning import LearningError, covered_gap_reply, parse_gap
from genesis.core.configuration.settings import DatabaseConfig, LearningConfig, LoggingConfig, ModelConfig, Settings
from genesis.core.model_interface.llama_cpp_backend import GenerationResult
from genesis.tools.research.http_client import HttpResponse, ResearchError

PLAN = json.dumps({"steps": [{"title": "Repeat", "instruction": "Repeat ping with echo."}]})
GAP = json.dumps(
    {
        "missing": ["echo"],
        "known": [],
        "experiments": [
            {
                "name": "echo",
                "hypothesis": "The echo module repeats the task.",
                "probe": "Install echo and invoke it with ping.",
                "expect": "echo ping",
                "uses": ["install", "invoke"],
            }
        ],
    }
)


class MapClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.routes: list[tuple[str, HttpResponse]] = []

    def add(self, prefix: str, body: bytes, *, status: int = 200, headers: dict[str, str] | None = None) -> None:
        self.routes.append(
            (prefix, HttpResponse(prefix, status, headers or {"content-type": "text/html"}, body, False))
        )

    def get(self, url: str, *, headers: dict[str, str] | None = None) -> HttpResponse:
        self.calls.append(url)
        for prefix, response in self.routes:
            if url.startswith(prefix):
                return HttpResponse(url, response.status, response.headers, response.body, response.truncated)
        raise ResearchError(f"No fixture for {url}")


class ScriptedBackend:
    def __init__(self, replies: list[str]) -> None:
        self.path: Path | None = None
        self.replies = list(replies)
        self.prompts: list[str] = []

    def load(self, path: Path, model: ModelConfig) -> None:
        self.path = path

    def unload(self) -> None:
        self.path = None

    def generate(self, messages, model, *, temperature=None, max_tokens=None) -> GenerationResult:
        self.prompts.append(str(messages[-1]["content"]))
        content = self.replies.pop(0) if self.replies else "done"
        return GenerationResult(content, 1, 1, 1.0, None if self.path is None else self.path.name)

    def stream(self, messages, model, *, temperature=None, max_tokens=None):
        yield "done"


def _echo(root: Path) -> None:
    directory = root / "incoming" / "echo" / "src"
    directory.mkdir(parents=True)
    (directory.parent / "manifest.json").write_text(
        json.dumps(
            {
                "id": "echo",
                "name": "Echo",
                "version": "0.1.0",
                "description": "Repeats a task.",
                "source": "test",
                "dependencies": [],
                "entry_point": "echo:Echo",
            }
        ),
        encoding="utf-8",
    )
    (directory / "echo.py").write_text(
        "class Echo:\n"
        "    def initialize(self, context):\n"
        "        return None\n"
        "    def execute(self, task, context):\n"
        "        return 'echo ' + str(task)\n"
        "    def verify(self, result):\n"
        "        return isinstance(result, str) and result.startswith('echo ')\n"
        "    def shutdown(self):\n"
        "        return None\n",
        encoding="utf-8",
    )


def _context(tmp_path: Path, replies: list[str], http: MapClient | None = None, *, max_experiments: int = 3):
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "demo.gguf").write_bytes(b"gguf")
    settings = Settings(
        model=ModelConfig(directory=str(model_dir), n_gpu_layers=0),
        database=DatabaseConfig(path=str(tmp_path / "genesis.db")),
        logging=LoggingConfig(directory=str(tmp_path / "logs"), level="WARNING"),
        learning=LearningConfig(max_experiments=max_experiments),
    )
    backend = ScriptedBackend(replies)
    context = build_context(tmp_path, settings, backend, http)
    return context, backend


def test_parse_gap_strips_thinking_and_keeps_the_experiment_limit() -> None:
    raw = json.loads(GAP)
    raw["experiments"] = raw["experiments"] * 3
    text = "<think>draft</think>\n```json\n" + json.dumps(raw) + "\n```"
    assessment = parse_gap(text, max_experiments=1)
    assert assessment.missing == ("echo",)
    assert len(assessment.experiments) == 1
    assert assessment.experiments[0].expect == "echo ping"
    try:
        parse_gap('{"missing":[],"experiments":[{"name":"bad name","hypothesis":"x","probe":"y","expect":"z"}]}', max_experiments=1)
    except LearningError as exc:
        assert "skill name" in str(exc)
    else:
        raise AssertionError("an invalid experiment name should fail")


def test_unfamiliar_task_is_tried_and_the_procedure_is_kept(tmp_path: Path) -> None:
    _echo(tmp_path)
    http = MapClient()
    http.add(
        "https://lite.duckduckgo.com/lite/",
        b"<a href='//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fecho&amp;rut=abc' class='result-link'>Echo</a>"
        b"<td class='result-snippet'>Repeats a task.</td>",
    )
    search = json.dumps({"action": "search", "query": "echo module"})
    install = json.dumps({"action": "install", "path": "incoming/echo"})
    invoke = json.dumps({"action": "invoke", "name": "echo", "task": "ping"})
    context, _backend = _context(
        tmp_path,
        [GAP, search, install, invoke, "echo ping", PLAN, "echo ping", "echo ping"],
        http,
    )
    context.models.startup()
    objective = context.orchestrator.submit("Learn echo and repeat ping")
    context.orchestrator.run_available()
    stored, events = context.orchestrator.get(objective.id)
    assert stored.status == "completed"
    assert stored.result == "echo ping"
    experiments = context.db.list_experiments(objective.id)
    assert len(experiments) == 1
    assert experiments[0].success is True
    assert experiments[0].name == "echo"
    skills = context.db.list_skills()
    assert [skill.name for skill in skills] == ["echo"]
    assert "Result: echo ping" in skills[0].body
    assert (tmp_path / "storage" / "skills" / "echo.md").is_file()
    assert context.db.list_resources(objective.id)[0].kind == "search"
    assert context.db.get_module("echo") is not None
    kinds = [event.kind for event in events]
    assert "gap" in kinds
    assert "experiment" in kinds
    assert "retained" in kinds
    assert events[kinds.index("gap")].message == "Missing: echo."

    client = TestClient(create_app(context))
    with client:
        listed = client.get(f"/objectives/{objective.id}/experiments")
        assert listed.status_code == 200
        assert listed.json()[0]["success"] is True
        assert listed.json()[0]["expected"] == "echo ping"
        assert client.get("/objectives/missing/experiments").status_code == 404


def test_a_saved_skill_is_reused_instead_of_repeating_the_experiment(tmp_path: Path) -> None:
    _echo(tmp_path)
    http = MapClient()
    http.add(
        "https://lite.duckduckgo.com/lite/",
        b"<a class='result-link' href='//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fecho&amp;rut=abc'>Echo</a>"
        b"<td class='result-snippet'>Repeats a task.</td>",
    )
    search = json.dumps({"action": "search", "query": "echo module"})
    install = json.dumps({"action": "install", "path": "incoming/echo"})
    invoke = json.dumps({"action": "invoke", "name": "echo", "task": "ping"})
    context, backend = _context(
        tmp_path,
        [
            GAP,
            search,
            install,
            invoke,
            "echo ping",
            PLAN,
            "echo ping",
            "echo ping",
            PLAN,
            "echo ping",
            "echo ping",
        ],
        http,
    )
    context.models.startup()
    first = context.orchestrator.submit("Learn echo and repeat ping")
    context.orchestrator.run_available()
    assert context.orchestrator.get(first.id)[0].status == "completed"
    before = len(backend.prompts)
    second = context.orchestrator.submit("Use echo to repeat ping")
    context.orchestrator.run_available()
    stored, events = context.orchestrator.get(second.id)
    assert stored.status == "completed"
    assert stored.result == "echo ping"
    assert context.db.list_experiments(second.id) == []
    assert [event.kind for event in events if event.kind == "experiment"] == []
    assert any(event.kind == "gap" and "Covered by echo" in event.message for event in events)
    learned = "\n".join(backend.prompts[before:])
    assert "Skill echo:" in learned
    assert "Result: echo ping" in learned


def test_a_failed_experiment_is_kept_without_a_skill(tmp_path: Path) -> None:
    context, _backend = _context(tmp_path, [GAP, "nope", PLAN, "done anyway", "done anyway"])
    context.models.startup()
    objective = context.orchestrator.submit("Learn echo and repeat ping")
    context.orchestrator.run_available()
    stored, _events = context.orchestrator.get(objective.id)
    assert stored.status == "completed"
    assert stored.result == "done anyway"
    experiment = context.db.list_experiments(objective.id)[0]
    assert experiment.success is False
    assert experiment.outcome == "nope"
    assert context.db.list_skills() == []


def test_assessment_asks_again_when_the_first_reply_is_not_json(tmp_path: Path) -> None:
    context, _backend = _context(tmp_path, ["nope", covered_gap_reply(), PLAN, "ok", "ok"])
    context.models.startup()
    objective = context.orchestrator.submit("Learn echo and repeat ping")
    context.orchestrator.run_available()
    stored, events = context.orchestrator.get(objective.id)
    assert stored.status == "completed"
    assert stored.result == "ok"
    assert any(event.kind == "gap" and event.message == "No missing capability." for event in events)
    assert context.db.list_experiments(objective.id) == []


def test_a_bad_assessment_fails_the_objective_after_one_retry(tmp_path: Path) -> None:
    context, _backend = _context(tmp_path, ["nope", "still nope"])
    context.models.startup()
    objective = context.orchestrator.submit("Learn echo and repeat ping")
    context.orchestrator.run_available()
    stored, events = context.orchestrator.get(objective.id)
    assert stored.status == "failed"
    assert stored.error is not None
    assert "JSON" in stored.error
    assert events[-1].kind == "failed"
    assert context.db.list_gaps(objective.id) == []
