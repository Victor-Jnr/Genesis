"""FastAPI surface for Phase 0. This is the conversation UI until the workstation exists."""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from genesis.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    HealthResponse,
    MessageResponse,
    ModelStatusResponse,
    SelectModelRequest,
)
from genesis.core.conversation import ConversationService, UnknownConversation
from genesis.core.model_interface.llama_cpp_backend import ModelLoadError
from genesis.core.model_interface.service import ConfigPinned, ModelNotReady, ModelService, ModelSnapshot
from genesis.core.runtime import repo_root
from genesis.monitoring.logging import log_event
from genesis.storage.database import Database

log = logging.getLogger("genesis")


def create_app(settings_root: PathContext) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.context.models.startup()
        yield
        app.state.context.models.shutdown()

    app = FastAPI(title="Genesis", version="0.1.0", lifespan=lifespan)
    app.state.context = settings_root

    @app.middleware("http")
    async def log_request(request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        log_event(
            logging.INFO,
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return response

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", venv=sys.prefix, phase="0")

    @app.get("/model", response_model=ModelStatusResponse)
    def model_status() -> ModelStatusResponse:
        return _status(app.state.context.models.snapshot())

    @app.post("/model/reload", response_model=ModelStatusResponse)
    def model_reload() -> ModelStatusResponse:
        return _status(app.state.context.models.reload())

    @app.post("/model/select", response_model=ModelStatusResponse)
    def model_select(body: SelectModelRequest) -> ModelStatusResponse:
        try:
            return _status(app.state.context.models.select(body.filename))
        except ConfigPinned as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ModelLoadError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest) -> ChatResponse:
        try:
            turn = app.state.context.conversations.send(
                body.message,
                body.conversation_id,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
        except UnknownConversation as exc:
            raise HTTPException(status_code=404, detail="Conversation not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ModelNotReady as exc:
            raise HTTPException(
                status_code=409,
                detail={"message": exc.detail, "discovered": exc.discovered},
            ) from exc
        except ModelLoadError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return ChatResponse(
            conversation_id=turn.conversation_id,
            message=turn.result.content,
            model=turn.result.model_filename,
            prompt_tokens=turn.result.prompt_tokens,
            completion_tokens=turn.result.completion_tokens,
            duration_ms=turn.result.duration_ms,
        )

    @app.post("/chat/stream")
    def chat_stream(body: ChatRequest) -> StreamingResponse:
        try:
            events = app.state.context.conversations.stream(
                body.message,
                body.conversation_id,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
            first = next(events)
        except UnknownConversation as exc:
            raise HTTPException(status_code=404, detail="Conversation not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ModelNotReady as exc:
            raise HTTPException(
                status_code=409,
                detail={"message": exc.detail, "discovered": exc.discovered},
            ) from exc
        except ModelLoadError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except StopIteration:
            raise HTTPException(status_code=502, detail="Model returned no stream.") from None

        def encoded() -> Iterator[str]:
            yield _sse(first)
            for event in events:
                yield _sse(event)

        return StreamingResponse(encoded(), media_type="text/event-stream")

    @app.get("/conversations", response_model=list[ConversationResponse])
    def list_conversations() -> list[ConversationResponse]:
        return [
            ConversationResponse(
                id=item.id,
                title=item.title,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            for item in app.state.context.conversations.list_conversations()
        ]

    @app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
    def get_conversation(conversation_id: str) -> ConversationResponse:
        try:
            conversation, messages = app.state.context.conversations.get_conversation(conversation_id)
        except UnknownConversation as exc:
            raise HTTPException(status_code=404, detail="Conversation not found.") from exc
        return ConversationResponse(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            messages=[
                MessageResponse(id=item.id, role=item.role, content=item.content, created_at=item.created_at)
                for item in messages
            ],
        )

    return app


class PathContext:
    def __init__(self, root, settings, db: Database, models: ModelService) -> None:
        self.root = root
        self.settings = settings
        self.db = db
        self.models = models
        self.conversations = ConversationService(settings, db, models)


def build_context(root, settings, backend) -> PathContext:
    db_path = settings.resolve_path(settings.database.path, root)
    db = Database(db_path)
    db.migrate()
    models = ModelService(settings, db, backend, root)
    return PathContext(root, settings, db, models)


def build_runtime_context(backend=None) -> PathContext:
    from genesis.core.configuration.settings import load_settings
    from genesis.core.model_interface.service import default_backend

    root = repo_root()
    settings = load_settings(root=root)
    return build_context(root, settings, backend or default_backend())


def _status(snapshot: ModelSnapshot) -> ModelStatusResponse:
    return ModelStatusResponse(
        loaded=snapshot.loaded,
        directory=snapshot.directory,
        filename=snapshot.filename,
        discovered=snapshot.discovered,
        context_length=snapshot.context_length,
        n_gpu_layers=snapshot.n_gpu_layers,
        detail=snapshot.detail,
    )


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"
