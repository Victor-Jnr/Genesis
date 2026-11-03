import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from genesis.api.app import build_context, create_app
from genesis.cognition.learning import covered_gap_reply
from genesis.core.configuration.settings import DatabaseConfig, LoggingConfig, ModelConfig, ResearchConfig, Settings
from genesis.core.model_interface.llama_cpp_backend import GenerationResult
from genesis.tools.research.actions import parse_action
from genesis.tools.research.documents import html_to_text, read_documentation
from genesis.tools.research.http_client import HttpResponse, ResearchError, UrllibClient, validate_public_url

DOC = """
# httpx

$ pip install httpx

```requirements.txt
certifi>=2024.1.0
idna==3.7
```

dependencies = [
  "httpcore>=1.0",
  "sniffio"
]
"""

PLAN = json.dumps(
    {"steps": [{"title": "Look up", "instruction": "Find the httpx package and its dependencies."}]}
)


class MapClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.routes: list[tuple[str, HttpResponse]] = []

    def add(self, prefix: str, body: bytes, *, status: int = 200, headers: dict[str, str] | None = None) -> None:
        self.routes.append(
            (
                prefix,
                HttpResponse(
                    url=prefix,
                    status=status,
                    headers=headers or {"content-type": "application/json"},
                    body=body,
                    truncated=False,
                ),
            )
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

    def load(self, path: Path, model: ModelConfig) -> None:
        self.path = path

    def unload(self) -> None:
        self.path = None

    def generate(self, messages, model, *, temperature=None, max_tokens=None, response_format=None) -> GenerationResult:
        content = self.replies.pop(0) if self.replies else "done"
        return GenerationResult(content, 1, 1, 1.0, None if self.path is None else self.path.name)

    def stream(self, messages, model, *, temperature=None, max_tokens=None):
        yield "done"


def _settings(tmp_path: Path, *, enabled: bool = True) -> Settings:
    return Settings(
        model=ModelConfig(directory=str(tmp_path / "models"), n_gpu_layers=0),
        database=DatabaseConfig(path=str(tmp_path / "genesis.db")),
        logging=LoggingConfig(directory=str(tmp_path / "logs"), level="WARNING"),
        research=ResearchConfig(
            enabled=enabled,
            download_directory=str(tmp_path / "downloads"),
            allow_private_hosts=False,
        ),
    )


def _open(tmp_path: Path, replies: list[str], http: MapClient | None = None, *, enabled: bool = True):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "demo.gguf").write_bytes(b"gguf")
    return build_context(tmp_path, _settings(tmp_path, enabled=enabled), ScriptedBackend(replies), http)


def test_documentation_reader_finds_dependencies_and_install_lines() -> None:
    notes = read_documentation(DOC)
    assert notes.dependencies == ("httpx", "certifi", "idna", "httpcore", "sniffio")
    assert notes.install_requirements == ("pip install httpx",)
    assert "pip install httpx" in notes.excerpt


def test_html_to_text_drops_scripts() -> None:
    assert html_to_text("<p>Hello <b>there</b></p><script>secret()</script>") == "Hello there"


def test_parse_action_ignores_plain_text() -> None:
    assert parse_action("The answer is 4.") is None
    action = parse_action('{"action":"package","name":"httpx"}')
    assert action is not None
    assert action.name == "package"
    assert action.package == "httpx"


def test_private_and_non_http_urls_are_refused() -> None:
    with pytest.raises(ResearchError, match="public host"):
        validate_public_url("http://127.0.0.1/latest", allow_private_hosts=False)
    with pytest.raises(ResearchError, match="public host"):
        validate_public_url("http://169.254.169.254/", allow_private_hosts=False)
    with pytest.raises(ResearchError, match="http"):
        validate_public_url("file:///C:/secret", allow_private_hosts=False)
    validate_public_url("http://127.0.0.1/", allow_private_hosts=True)
    client = UrllibClient(timeout=1, max_bytes=100, user_agent="Genesis", allow_private_hosts=False)
    with pytest.raises(ResearchError, match="public host"):
        client.get("http://127.0.0.1/")


def test_search_package_read_and_download_record_metadata(tmp_path: Path) -> None:
    http = MapClient()
    http.add(
        "https://lite.duckduckgo.com/lite/",
        (
            "<a href=\"//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.python-httpx.org%2F&amp;rut=abc\" "
            "class='result-link'>HTTPX</a>"
            "<td class='result-snippet'>An HTTP client.</td>"
        ).encode(),
        headers={"content-type": "text/html"},
    )
    http.add(
        "https://pypi.org/pypi/httpx/json",
        json.dumps(
            {
                "info": {
                    "name": "httpx",
                    "version": "0.27.0",
                    "summary": "HTTP client",
                    "home_page": "https://www.python-httpx.org/",
                    "requires_dist": ["httpcore>=1.0.0", "certifi"],
                }
            }
        ).encode(),
    )
    http.add(
        "https://example.com/readme",
        b"<html><body><h1>httpx</h1><p>pip install httpx</p></body></html>",
        headers={"content-type": "text/html"},
    )
    http.add("https://example.com/file.txt", b"hello")
    context = _open(tmp_path, [], http)
    context.models.startup()
    objective = context.db.create_objective("Learn httpx")
    research = context.research
    found = research.execute(objective.id, parse_action('{"action":"search","query":"httpx"}'))  # type: ignore[arg-type]
    assert "www.python-httpx.org" in found
    package = research.execute(objective.id, parse_action('{"action":"package","name":"httpx"}'))  # type: ignore[arg-type]
    assert "httpcore>=1.0.0" in package
    assert "certifi" in package
    notes = research.execute(objective.id, parse_action('{"action":"read","url":"https://example.com/readme"}'))  # type: ignore[arg-type]
    assert "Dependencies: httpx" in notes
    assert "pip install httpx" in notes
    saved = research.execute(objective.id, parse_action('{"action":"download","url":"https://example.com/file.txt"}'))  # type: ignore[arg-type]
    digest = hashlib.sha256(b"hello").hexdigest()
    assert digest in saved
    path = next((tmp_path / "downloads").rglob("file.txt"))
    assert path.read_bytes() == b"hello"
    kinds = [item.kind for item in context.db.list_resources(objective.id)]
    assert kinds == ["search", "package", "documentation", "download"]
    blocked = research.execute(objective.id, parse_action('{"action":"fetch","url":"file:///C:/secret"}'))  # type: ignore[arg-type]
    assert blocked.startswith("Research failed:")


def test_disabled_research_does_not_call_the_network(tmp_path: Path) -> None:
    http = MapClient()
    context = _open(tmp_path, [], http, enabled=False)
    objective = context.db.create_objective("Look around")
    text = context.research.execute(objective.id, parse_action('{"action":"search","query":"httpx"}'))  # type: ignore[arg-type]
    assert "disabled" in text
    assert http.calls == []


def test_objective_searches_before_finishing_a_step(tmp_path: Path) -> None:
    http = MapClient()
    http.add(
        "https://lite.duckduckgo.com/lite/",
        (
            "<a href=\"//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.python-httpx.org%2F&amp;rut=abc\" "
            "class='result-link'>HTTPX</a>"
            "<td class='result-snippet'>An HTTP client.</td>"
        ).encode(),
        headers={"content-type": "text/html"},
    )
    action = json.dumps({"action": "search", "query": "httpx"})
    context = _open(
        tmp_path,
        [covered_gap_reply(), PLAN, action, "httpx is an HTTP client.", "httpx is an HTTP client."],
        http,
    )
    context.models.startup()
    objective = context.orchestrator.submit("Find what httpx is")
    context.orchestrator.run_available()
    stored, events = context.orchestrator.get(objective.id)
    assert stored.status == "completed"
    assert stored.result == "httpx is an HTTP client."
    assert "search" in [event.kind for event in events]
    resources = context.db.list_resources(objective.id)
    assert resources[0].kind == "search"
    assert resources[0].metadata_json
    client = TestClient(create_app(context))
    with client:
        listed = client.get(f"/objectives/{objective.id}/resources")
        assert listed.status_code == 200
        assert listed.json()[0]["kind"] == "search"
        assert client.get("/objectives/missing/resources").status_code == 404
        assert client.get("/resources").json()[0]["objective_id"] == objective.id
