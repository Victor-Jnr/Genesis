"""CLI entry. The virtual-environment check runs before the server starts."""

from __future__ import annotations

import uvicorn

from genesis.api.app import build_runtime_context, create_app
from genesis.core.runtime import assert_project_venv
from genesis.monitoring.logging import configure_logging


def main() -> None:
    assert_project_venv()
    context = build_runtime_context()
    configure_logging(context.settings, context.root)
    app = create_app(context)
    uvicorn.run(
        app,
        host=context.settings.server.host,
        port=context.settings.server.port,
        log_level=context.settings.logging.level.lower(),
    )
