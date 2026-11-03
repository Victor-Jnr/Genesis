import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from genesis.api.app import build_context, create_app
from genesis.cognition.learning import GapAssessment, ExperimentSpec, ground_gap
from genesis.core.configuration.settings import (
    DatabaseConfig,
    ExecutionConfig,
    LabConfig,
    LoggingConfig,
    ModelConfig,
    Settings,
)
from genesis.core.model_interface.llama_cpp_backend import GenerationResult
from genesis.tools.execution.service import ExecutionService
from genesis.tools.research.actions import ResearchAction


class QuietBackend:
    def __init__(self) -> None:
        self.path: Path | None = None

    def load(self, path: Path, model: ModelConfig) -> None:
        self.path = path

    def unload(self) -> None:
        self.path = None

    def generate(self, messages, model, *, temperature=None, max_tokens=None, response_format=None) -> GenerationResult:
        return GenerationResult("done", 1, 1, 1.0, "demo.gguf")

    def stream(self, messages, model, *, temperature=None, max_tokens=None):
        yield "done"


def _settings(tmp_path: Path, *, enabled: bool = False, lab: bool = False) -> Settings:
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "demo.gguf").write_bytes(b"gguf")
    return Settings(
        model=ModelConfig(directory=str(model_dir), n_gpu_layers=0),
        database=DatabaseConfig(path=str(tmp_path / "genesis.db")),
        logging=LoggingConfig(directory=str(tmp_path / "logs"), level="WARNING"),
        execution=ExecutionConfig(enabled=enabled, timeout_seconds=10),
        lab=LabConfig(enabled=lab),
    )


def test_commands_stay_off_until_enabled(tmp_path: Path) -> None:
    service = ExecutionService(_settings(tmp_path), tmp_path)
    message = service.execute(None, ResearchAction("run", command="python -c print(1)"))
    assert message.startswith("Execution failed:")
    assert "disabled" in message


def test_a_local_program_runs_without_a_shell(tmp_path: Path) -> None:
    context = build_context(tmp_path, _settings(tmp_path, enabled=True), QuietBackend())
    command = subprocess_command([sys.executable, "-c", "print(7)"])
    output = context.execution.run(command)
    assert output.startswith("Exit 0")
    assert "7" in output
    client = TestClient(create_app(context))
    with client:
        denied = client.post("/commands", json={"command": "python -c \"print(1)\" | more"})
        assert denied.status_code == 400
        private = client.post("/commands", json={"command": "ping 10.10.1.1"})
        assert private.status_code == 400
        assert "lab.enabled" in private.json()["detail"]
        public = client.post("/commands", json={"command": "ping 8.8.8.8"})
        assert public.status_code == 400
        assert "outside the lab" in public.json()["detail"]


def test_lab_mode_allows_a_private_address(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    def runner(argv: list[str], cwd: Path, timeout: float) -> tuple[int, str]:
        seen.append(argv)
        return 0, "ok"

    context = build_context(
        tmp_path,
        _settings(tmp_path, enabled=True, lab=True),
        QuietBackend(),
        command_runner=runner,
    )
    output = context.execution.run("ping 10.10.1.1")
    assert output.startswith("Exit 0")
    assert seen == [["ping", "10.10.1.1"]]
    blocked = context.execution.execute(None, ResearchAction("run", command="ping 1.1.1.1"))
    assert "outside the lab" in blocked


def test_gap_keeps_only_experiments_that_name_real_actions() -> None:
    assessment = GapAssessment(
        ("parse", "fetch"),
        (),
        (
            ExperimentSpec("parse-page", "Parse it.", "Parse the page.", "module data", ("parse",)),
            ExperimentSpec("read-page", "Read it.", "Read the page.", "title", ("fetch", "read")),
            ExperimentSpec("unnamed", "Try it.", "Try something.", "done", ()),
        ),
    )
    grounded = ground_gap(assessment, {"fetch", "read", "install"})
    assert grounded.missing == ("parse",)
    assert [item.name for item in grounded.experiments] == ["read-page"]


def test_objective_can_run_a_program(tmp_path: Path) -> None:
    command = subprocess_command([sys.executable, "-c", "print(9)"])
    replies = [
        json.dumps({"missing": [], "known": [], "experiments": []}),
        json.dumps({"steps": [{"title": "Run", "instruction": "Print nine."}]}),
        json.dumps({"action": "run", "command": command}),
        "Printed.",
        "Printed.",
    ]

    class Scripted(QuietBackend):
        def __init__(self) -> None:
            super().__init__()
            self.replies = list(replies)

        def generate(self, messages, model, *, temperature=None, max_tokens=None, response_format=None) -> GenerationResult:
            content = self.replies.pop(0) if self.replies else "done"
            return GenerationResult(content, 1, 1, 1.0, "demo.gguf")

    context = build_context(tmp_path, _settings(tmp_path, enabled=True), Scripted())
    context.models.startup()
    objective = context.orchestrator.submit("Print nine with the local interpreter.")
    context.orchestrator.run_available()
    stored, events = context.orchestrator.get(objective.id)
    assert stored.status == "completed"
    assert any(event.kind == "run" and "Exit 0" in event.message and "9" in event.message for event in events)


def subprocess_command(argv: list[str]) -> str:
    return " ".join(f'"{part}"' if any(char in part for char in ' \t\\"') else part for part in argv)
