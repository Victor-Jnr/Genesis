"""JSON logs for the genesis logger. Configured from the CLI, not at import."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from genesis.core.configuration.settings import Settings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(settings: Settings, root: Path) -> None:
    log_dir = settings.resolve_path(settings.logging.directory, root)
    log_dir.mkdir(parents=True, exist_ok=True)
    level_name = settings.logging.level.upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = JsonFormatter()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir / "genesis.jsonl", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger = logging.getLogger("genesis")
    logger.handlers.clear()
    logger.setLevel(level)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    logger.propagate = False


def log_event(level: int, message: str, **fields: object) -> None:
    logging.getLogger("genesis").log(level, message, extra={"fields": fields})
