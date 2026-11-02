"""Load and talk to the selected GGUF without owning conversation history."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

from genesis.core.configuration.settings import ModelConfig, Settings
from genesis.core.model_interface.discovery import (
    ModelChoice,
    list_gguf,
    model_file,
    resolve_model_choice,
)
from genesis.core.model_interface.llama_cpp_backend import GenerationResult, LlamaCppBackend, ModelLoadError
from genesis.storage.database import Database

log = logging.getLogger("genesis")
SELECTED_MODEL_KEY = "model.filename"


class Backend(Protocol):
    path: Path | None

    def load(self, path: Path, model: ModelConfig) -> None: ...

    def unload(self) -> None: ...

    def generate(
        self,
        messages: list[dict[str, str]],
        model: ModelConfig,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult: ...

    def stream(
        self,
        messages: list[dict[str, str]],
        model: ModelConfig,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]: ...


@dataclass
class ModelSnapshot:
    loaded: bool
    directory: str
    filename: str | None
    discovered: list[str]
    context_length: int
    n_gpu_layers: int
    detail: str | None


class ModelNotReady(RuntimeError):
    def __init__(self, detail: str, discovered: list[str]) -> None:
        super().__init__(detail)
        self.detail = detail
        self.discovered = discovered


class ConfigPinned(RuntimeError):
    pass


class ModelService:
    def __init__(self, settings: Settings, db: Database, backend: Backend, root: Path) -> None:
        self.settings = settings
        self.db = db
        self.backend = backend
        self.root = root
        self.directory = settings.resolve_path(settings.model.directory, root)
        self._detail: str | None = None
        self._lock = threading.Lock()

    def startup(self) -> ModelSnapshot:
        try:
            return self.reload()
        except ModelLoadError as exc:
            log.exception("model load failed")
            self.backend.unload()
            self._detail = str(exc)
            return self.snapshot()

    def shutdown(self) -> None:
        self.backend.unload()

    def reload(self) -> ModelSnapshot:
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            try:
                self._apply(self._choice())
            except ModelLoadError as exc:
                self.backend.unload()
                self._detail = str(exc)
                log.exception("model load failed")
            return self.snapshot()

    def select(self, filename: str) -> ModelSnapshot:
        with self._lock:
            if self.settings.model.filename:
                raise ConfigPinned(
                    "model.filename is set in config. Clear it to select a model through the API, "
                    "or change that value in the yaml."
                )
            path = model_file(self.directory, filename)
            self.db.set_setting(SELECTED_MODEL_KEY, path.name)
            self._detail = None
            try:
                self._apply(self._choice())
            except ModelLoadError as exc:
                self.backend.unload()
                self._detail = str(exc)
                log.exception("model load failed")
            return self.snapshot()

    def snapshot(self) -> ModelSnapshot:
        files = list_gguf(self.directory) if self.directory.exists() else []
        loaded_path = self.backend.path
        return ModelSnapshot(
            loaded=loaded_path is not None,
            directory=str(self.directory),
            filename=None if loaded_path is None else loaded_path.name,
            discovered=[path.name for path in files],
            context_length=self.settings.model.context_length,
            n_gpu_layers=self.settings.model.n_gpu_layers,
            detail=self._detail,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        with self._lock:
            self._require_loaded()
            return self.backend.generate(
                messages,
                self.settings.model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        with self._lock:
            self._require_loaded()
            yield from self.backend.stream(
                messages,
                self.settings.model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    def _choice(self) -> ModelChoice:
        return resolve_model_choice(
            list_gguf(self.directory),
            directory=self.directory,
            configured_filename=self.settings.model.filename,
            remembered_filename=self.db.get_setting(SELECTED_MODEL_KEY),
        )

    def _apply(self, choice: ModelChoice) -> None:
        if choice.remember_as:
            self.db.set_setting(SELECTED_MODEL_KEY, choice.remember_as)
        if choice.path is None:
            self.backend.unload()
            self._detail = choice.detail
            return
        current = self.backend.path
        if current is not None and _same_path(current, choice.path):
            self._detail = None
            return
        started = time.perf_counter()
        self.backend.load(choice.path, self.settings.model)
        self._detail = None
        log.info(
            "model loaded",
            extra={"fields": {"model": choice.path.name, "duration_ms": round((time.perf_counter() - started) * 1000, 1)}},
        )

    def _require_loaded(self) -> None:
        if self.backend.path is None:
            snapshot = self.snapshot()
            detail = snapshot.detail or "No GGUF model is loaded."
            raise ModelNotReady(detail, snapshot.discovered)


def default_backend() -> LlamaCppBackend:
    return LlamaCppBackend()


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve()))
