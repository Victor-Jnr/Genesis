from pathlib import Path

from fastapi.testclient import TestClient

from genesis.api.app import build_context, create_app
from genesis.core.configuration.settings import DatabaseConfig, LoggingConfig, ModelConfig, Settings
from genesis.core.model_interface.llama_cpp_backend import GenerationResult


class FakeBackend:
    def __init__(self) -> None:
        self.path: Path | None = None

    def load(self, path: Path, model: ModelConfig) -> None:
        self.path = path

    def unload(self) -> None:
        self.path = None

    def generate(self, messages, model, *, temperature=None, max_tokens=None) -> GenerationResult:
        return GenerationResult("pong", 1, 1, 1.0, "demo.gguf")

    def stream(self, messages, model, *, temperature=None, max_tokens=None):
        yield "pong"


def _client(tmp_path: Path) -> TestClient:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "demo.gguf").write_bytes(b"gguf")
    settings = Settings(
        model=ModelConfig(directory=str(model_dir), n_gpu_layers=0),
        database=DatabaseConfig(path=str(tmp_path / "genesis.db")),
        logging=LoggingConfig(directory=str(tmp_path / "logs"), level="WARNING"),
    )
    return TestClient(create_app(build_context(tmp_path, settings, FakeBackend())))


def test_workstation_pages_and_assets(tmp_path: Path) -> None:
    client = _client(tmp_path)
    with client:
        dashboard = client.get("/workstation")
        assert dashboard.status_code == 200
        assert "text/html" in dashboard.headers["content-type"]
        page = dashboard.text
        for label in ("Dashboard", "Monitor", "Objectives", "Development", "Capabilities"):
            assert label in page
        assert 'data-page="dashboard"' in page
        monitor = client.get("/workstation/monitor")
        objectives = client.get("/workstation/objectives")
        development = client.get("/workstation/development")
        capabilities = client.get("/workstation/capabilities")
        detail = client.get("/workstation/objectives/abc")
        assert monitor.status_code == objectives.status_code == development.status_code == 200
        assert capabilities.status_code == detail.status_code == 200
        assert 'data-page="monitor"' in monitor.text
        assert 'data-page="objective"' in detail.text
        assert 'data-id="abc"' in detail.text
        assert client.get("/workstation/missing").status_code == 404
        script = client.get("/workstation/assets/workstation.js")
        assert script.status_code == 200
        assert "Message sent. Waiting for a reply." in script.text
        assert 'aria-live="polite"' in page
        assert client.get("/workstation/assets/workstation.css").status_code == 200


def test_config_snapshot_omits_secrets(tmp_path: Path) -> None:
    client = _client(tmp_path)
    with client:
        config = client.get("/config")
        assert config.status_code == 200
        body = config.json()
        assert body["n_gpu_layers"] == 0
        assert body["evolution_benchmark"] == "tests"
        assert body["model_directory"].endswith("models")
        assert "github_token" not in body
        assert "system_prompt" not in body
