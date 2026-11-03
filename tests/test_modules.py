import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from genesis.api.app import build_context, create_app
from genesis.cognition.learning import covered_gap_reply
from genesis.capabilities.service import requirements_command
from genesis.core.configuration.settings import DatabaseConfig, LoggingConfig, ModelConfig, Settings
from genesis.core.model_interface.llama_cpp_backend import GenerationResult
from genesis.core.runtime import project_venv_python
from genesis.tools.research.actions import ResearchAction


PLAN = json.dumps({"steps": [{"title": "Use echo", "instruction": "Install echo and invoke it."}]})


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


def _echo(root: Path, *, requirements: str | None = None) -> Path:
    directory = root / "incoming" / "echo"
    (directory / "src").mkdir(parents=True)
    (directory / "manifest.json").write_text(
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
    (directory / "src" / "echo.py").write_text(
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
    if requirements is not None:
        (directory / "requirements.txt").write_text(requirements, encoding="utf-8")
    return directory


def _context(tmp_path: Path, replies: list[str] | None = None, runner=None):
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "demo.gguf").write_bytes(b"gguf")
    settings = Settings(
        model=ModelConfig(directory=str(model_dir), n_gpu_layers=0),
        database=DatabaseConfig(path=str(tmp_path / "genesis.db")),
        logging=LoggingConfig(directory=str(tmp_path / "logs"), level="WARNING"),
    )
    return build_context(tmp_path, settings, ScriptedBackend(replies or []), module_runner=runner)


def test_requirements_command_uses_the_project_venv() -> None:
    command = requirements_command(Path("requirements.txt"))
    assert Path(command[0]).resolve() == project_venv_python().resolve()
    assert ".venv" in command[0].replace("\\", "/")
    assert command[1:] == ["-m", "pip", "install", "-r", "requirements.txt"]


def test_install_invoke_and_reload_without_changing_the_core(tmp_path: Path) -> None:
    source = _echo(tmp_path)
    context = _context(tmp_path)
    record = context.modules.install(source)
    assert record.status == "active"
    assert record.id == "echo"
    assert context.modules.invoke(None, "echo", "ping") == "echo ping"
    assert (tmp_path / "genesis_modules" / "echo" / "manifest.json").is_file()

    reloaded = _context(tmp_path)
    reloaded.modules.startup()
    assert [item.id for item in reloaded.db.list_modules()] == ["echo"]
    assert reloaded.modules.invoke(None, "echo", "again") == "echo again"
    inventory = reloaded.modules.inventory()
    assert "echo 0.1.0 (module, active)" in inventory
    assert "search (built-in, active)" in inventory
    assert "install (built-in, active)" in inventory
    assert "invoke (built-in, active)" in inventory


def test_install_rejects_a_path_outside_the_project(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"outside-{tmp_path.name}"
    shutil.rmtree(outside, ignore_errors=True)
    try:
        _echo(outside)
        context = _context(tmp_path)
        failure = context.modules.execute(
            None,
            ResearchAction("install", path=str(outside / "incoming" / "echo")),
        )
        assert failure.startswith("Module failed:")
        assert "outside the project" in failure
    finally:
        shutil.rmtree(outside, ignore_errors=True)


def test_requirements_use_the_injected_installer(tmp_path: Path) -> None:
    source = _echo(tmp_path, requirements="# comment\nexample-package\n")
    seen: list[Path] = []
    context = _context(tmp_path, runner=lambda path: seen.append(path))
    context.modules.install(source)
    assert len(seen) == 1
    assert seen[0].name == "requirements.txt"


def test_skill_is_saved_and_found_after_reopen(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.modules.remember("greet", "Say hello before the task.")
    reloaded = _context(tmp_path)
    found = reloaded.modules.search_skills("hello")
    assert [skill.name for skill in found] == ["greet"]
    assert (tmp_path / "storage" / "skills" / "greet.md").read_text(encoding="utf-8").startswith("Say hello")


def test_objective_installs_and_invokes_a_module(tmp_path: Path) -> None:
    _echo(tmp_path)
    install = json.dumps({"action": "install", "path": "incoming/echo"})
    invoke = json.dumps({"action": "invoke", "name": "echo", "task": "ping"})
    context = _context(
        tmp_path,
        [covered_gap_reply(), PLAN, install, invoke, "The module echoed ping.", "The module echoed ping."],
    )
    context.models.startup()
    objective = context.orchestrator.submit("Use the echo module")
    context.orchestrator.run_available()
    stored, events = context.orchestrator.get(objective.id)
    assert stored.status == "completed"
    assert stored.result == "The module echoed ping."
    assert [event.kind for event in events if event.kind in {"install", "invoke"}] == ["install", "invoke"]
    client = TestClient(create_app(context))
    with client:
        listed = client.get("/modules")
        assert listed.status_code == 200
        assert listed.json()[0]["id"] == "echo"
        called = client.post("/modules/echo/invoke", json={"task": "live"})
        assert called.status_code == 200
        assert called.json()["result"] == "echo live"
        assert any(item["name"] == "echo" for item in client.get("/capabilities").json())
