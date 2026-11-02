"""Persist a conversation and send its history to the loaded model."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

from genesis.core.configuration.settings import Settings
from genesis.core.model_interface.llama_cpp_backend import GenerationResult
from genesis.core.model_interface.service import ModelService
from genesis.storage.database import Conversation, Database, Message


class UnknownConversation(KeyError):
    pass


@dataclass(frozen=True)
class ChatTurn:
    conversation_id: str
    result: GenerationResult


class ConversationService:
    def __init__(self, settings: Settings, db: Database, models: ModelService) -> None:
        self.settings = settings
        self.db = db
        self.models = models

    def list_conversations(self) -> list[Conversation]:
        return self.db.list_conversations()

    def get_conversation(self, conversation_id: str) -> tuple[Conversation, list[Message]]:
        conversation = self.db.get_conversation(conversation_id)
        if conversation is None:
            raise UnknownConversation(conversation_id)
        return conversation, self.db.list_messages(conversation_id)

    def send(
        self,
        message: str,
        conversation_id: str | None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatTurn:
        conversation_id = self._prepare(message, conversation_id)
        result = self.models.generate(
            self._prompt(conversation_id),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.db.add_message(conversation_id, "assistant", result.content)
        filename = None if self.models.backend.path is None else self.models.backend.path.name
        return ChatTurn(
            conversation_id,
            GenerationResult(
                content=result.content,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                duration_ms=result.duration_ms,
                model_filename=filename,
            ),
        )

    def stream(
        self,
        message: str,
        conversation_id: str | None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[dict[str, object]]:
        conversation_id = self._prepare(message, conversation_id)
        started = time.perf_counter()
        parts: list[str] = []
        for delta in self.models.stream(
            self._prompt(conversation_id),
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            parts.append(delta)
            yield {"type": "delta", "delta": delta, "conversation_id": conversation_id}
        content = "".join(parts)
        self.db.add_message(conversation_id, "assistant", content)
        filename = None if self.models.backend.path is None else self.models.backend.path.name
        yield {
            "type": "done",
            "conversation_id": conversation_id,
            "message": content,
            "model": filename,
            "duration_ms": (time.perf_counter() - started) * 1000,
        }

    def _prepare(self, message: str, conversation_id: str | None) -> str:
        text = message.strip()
        if not text:
            raise ValueError("Message is empty.")
        if conversation_id is None:
            conversation = self.db.create_conversation(text)
            conversation_id = conversation.id
        elif self.db.get_conversation(conversation_id) is None:
            raise UnknownConversation(conversation_id)
        self.db.add_message(conversation_id, "user", text)
        return conversation_id

    def _prompt(self, conversation_id: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        prompt = self.settings.identity.system_prompt.strip()
        if prompt:
            messages.append({"role": "system", "content": prompt})
        for item in self.db.list_messages(conversation_id):
            messages.append({"role": item.role, "content": item.content})
        return messages
