"""In-process GGUF backend. llama-cpp-python is imported only when a model loads."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from genesis.core.configuration.settings import ModelConfig


class ModelLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    content: str
    prompt_tokens: int | None
    completion_tokens: int | None
    duration_ms: float
    model_filename: str | None


class LlamaCppBackend:
    def __init__(self) -> None:
        self.path: Path | None = None
        self._llm: object | None = None

    def load(self, path: Path, model: ModelConfig) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ModelLoadError(
                "llama-cpp-python is not installed in .venv. From the repo root run scripts\\install.ps1"
            ) from exc
        self.unload()
        try:
            self._llm = Llama(
                model_path=str(path),
                n_ctx=model.context_length,
                n_gpu_layers=model.n_gpu_layers,
                verbose=False,
            )
        except Exception as exc:
            self._llm = None
            self.path = None
            raise ModelLoadError(f"Failed to load {path.name}: {exc}") from exc
        self.path = path

    def unload(self) -> None:
        self._llm = None
        self.path = None

    def generate(
        self,
        messages: list[dict[str, str]],
        model: ModelConfig,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, object] | None = None,
    ) -> GenerationResult:
        llm = self._require()
        started = time.perf_counter()
        response = llm.create_chat_completion(
            messages=messages,
            temperature=model.temperature if temperature is None else temperature,
            top_p=model.top_p,
            max_tokens=model.max_tokens if max_tokens is None else max_tokens,
            response_format=response_format,
        )
        duration_ms = (time.perf_counter() - started) * 1000
        return GenerationResult(
            content=_message_content(response),
            prompt_tokens=_usage(response, "prompt_tokens"),
            completion_tokens=_usage(response, "completion_tokens"),
            duration_ms=duration_ms,
            model_filename=None if self.path is None else self.path.name,
        )

    def stream(
        self,
        messages: list[dict[str, str]],
        model: ModelConfig,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        llm = self._require()
        chunks = llm.create_chat_completion(
            messages=messages,
            temperature=model.temperature if temperature is None else temperature,
            top_p=model.top_p,
            max_tokens=model.max_tokens if max_tokens is None else max_tokens,
            stream=True,
        )
        for chunk in chunks:
            delta = _delta_content(chunk)
            if delta:
                yield delta

    def _require(self) -> object:
        if self._llm is None:
            raise ModelLoadError("No GGUF model is loaded.")
        return self._llm


def _message_content(response: object) -> str:
    if not isinstance(response, dict):
        return str(response)
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    return "" if content is None else str(content)


def _delta_content(chunk: object) -> str:
    if not isinstance(chunk, dict):
        return ""
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return "" if content is None else str(content)


def _usage(response: object, key: str) -> int | None:
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if not isinstance(usage, dict) or key not in usage:
        return None
    value = usage[key]
    return value if isinstance(value, int) else None
