import json
from pathlib import Path

from fastapi.testclient import TestClient

from genesis.api.app import build_context, create_app
from genesis.core.configuration.settings import DatabaseConfig, LoggingConfig, ModelConfig, Settings
from genesis.core.model_interface.llama_cpp_backend import GenerationResult, ModelLoadError
from genesis.storage.database import Database


class FakeBackend:
    def __init__(self, reply: str = "pong", fail_load: bool = False) -> None:
        self.path: Path | None = None
        self.reply = reply
        self.fail_load = fail_load
        self.messages: list[dict[str, str]] | None = None

    def load(self, path: Path, model: ModelConfig) -> None:
        if self.fail_load:
            raise ModelLoadError("boom")
        self.path = path

    def unload(self) -> None:
        self.path = None

    def generate(self, messages, model, *, temperature=None, max_tokens=None) -> GenerationResult:
        self.messages = messages
        return GenerationResult(
            content=self.reply,
            prompt_tokens=4,
            completion_tokens=1,
            duration_ms=1.5,
            model_filename=None if self.path is None else self.path.name,
        )

    def stream(self, messages, model, *, temperature=None, max_tokens=None):
        self.messages = messages
        yield "pon"
        yield "g"


def _open(tmp_path: Path, names: list[str], filename: str | None = None, backend: FakeBackend | None = None):
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = model_dir / name
        if not target.exists():
            target.write_bytes(b"gguf")
    settings = Settings(
        model=ModelConfig(directory=str(model_dir), filename=filename, n_gpu_layers=0),
        database=DatabaseConfig(path=str(tmp_path / "genesis.db")),
        logging=LoggingConfig(directory=str(tmp_path / "logs"), level="WARNING"),
    )
    fake = backend or FakeBackend()
    context = build_context(tmp_path, settings, fake)
    return TestClient(create_app(context)), fake, context


def test_health_and_single_dropped_model_loads(tmp_path: Path) -> None:
    client, backend, _context = _open(tmp_path, ["demo.gguf"])
    with client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["phase"] == "0"
        assert ".venv" in health.json()["venv"]
        status = client.get("/model").json()
        assert status["loaded"] is True
        assert status["filename"] == "demo.gguf"
        assert status["discovered"] == ["demo.gguf"]
        assert status["n_gpu_layers"] == 0
        assert backend.path is not None
        assert backend.path.name == "demo.gguf"


def test_chat_persists_across_a_new_process(tmp_path: Path) -> None:
    client, backend, _context = _open(tmp_path, ["demo.gguf"])
    with client:
        first = client.post("/chat", json={"message": "Remember this"})
        assert first.status_code == 200
        body = first.json()
        assert body["message"] == "pong"
        assert body["model"] == "demo.gguf"
        conversation_id = body["conversation_id"]
        assert backend.messages is not None
        assert backend.messages[0]["role"] == "system"
        assert backend.messages[-1] == {"role": "user", "content": "Remember this"}
        second = client.post("/chat", json={"message": "And this", "conversation_id": conversation_id})
        assert second.status_code == 200
        assert [item["content"] for item in backend.messages if item["role"] != "system"] == [
            "Remember this",
            "pong",
            "And this",
        ]

    reopened, _backend, _context = _open(tmp_path, ["demo.gguf"])
    with reopened:
        stored = reopened.get(f"/conversations/{conversation_id}")
        assert stored.status_code == 200
        contents = [item["content"] for item in stored.json()["messages"]]
        assert contents == ["Remember this", "pong", "And this", "pong"]
        listed = reopened.get("/conversations")
        assert listed.status_code == 200
        assert listed.json()[0]["id"] == conversation_id


def test_database_file_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "genesis.db"
    first = Database(path)
    first.migrate()
    first.set_setting("model.filename", "demo.gguf")
    second = Database(path)
    second.migrate()
    assert second.get_setting("model.filename") == "demo.gguf"


def test_multiple_models_wait_for_a_selection(tmp_path: Path) -> None:
    client, _backend, _context = _open(tmp_path, ["a.gguf", "b.gguf"])
    with client:
        status = client.get("/model").json()
        assert status["loaded"] is False
        assert status["discovered"] == ["a.gguf", "b.gguf"]
        blocked = client.post("/chat", json={"message": "hello"})
        assert blocked.status_code == 409
        chosen = client.post("/model/select", json={"filename": "b.gguf"})
        assert chosen.status_code == 200
        assert chosen.json()["filename"] == "b.gguf"
        assert chosen.json()["loaded"] is True
        sent = client.post("/chat", json={"message": "hello"})
        assert sent.status_code == 200
        missing = client.post("/model/select", json={"filename": "../secret.gguf"})
        assert missing.status_code == 400


def test_pinned_filename_overrides_selection(tmp_path: Path) -> None:
    client, _backend, _context = _open(tmp_path, ["a.gguf", "b.gguf"], filename="a.gguf")
    with client:
        assert client.get("/model").json()["filename"] == "a.gguf"
        rejected = client.post("/model/select", json={"filename": "b.gguf"})
        assert rejected.status_code == 409
        assert client.get("/model").json()["filename"] == "a.gguf"


def test_stream_and_failed_load(tmp_path: Path) -> None:
    client, _backend, _context = _open(tmp_path, ["demo.gguf"])
    with client:
        response = client.post("/chat/stream", json={"message": "hi"})
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        events = [json.loads(line.removeprefix("data: ")) for line in response.text.splitlines() if line.startswith("data: ")]
        assert [event["delta"] for event in events if event["type"] == "delta"] == ["pon", "g"]
        assert events[-1]["type"] == "done"
        assert events[-1]["message"] == "pong"

    broken, _backend, _context = _open(tmp_path / "broken", ["demo.gguf"], backend=FakeBackend(fail_load=True))
    with broken:
        status = broken.get("/model")
        assert status.status_code == 200
        assert status.json()["loaded"] is False
        assert "boom" in status.json()["detail"]
        assert broken.get("/health").status_code == 200
        assert broken.post("/chat", json={"message": "hi"}).status_code == 409


def test_unknown_conversation_is_404(tmp_path: Path) -> None:
    client, _backend, _context = _open(tmp_path, ["demo.gguf"])
    with client:
        missing = client.post("/chat", json={"message": "hi", "conversation_id": "missing"})
        assert missing.status_code == 404
        assert client.get("/conversations/missing").status_code == 404
