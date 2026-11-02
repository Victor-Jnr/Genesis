import json
import logging

from genesis.monitoring.logging import JsonFormatter


def test_json_formatter_includes_fields() -> None:
    record = logging.LogRecord(
        name="genesis",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request",
        args=(),
        exc_info=None,
    )
    record.fields = {"path": "/health", "status": 200}
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "request"
    assert payload["level"] == "INFO"
    assert payload["path"] == "/health"
    assert payload["status"] == 200
    assert "timestamp" in payload
