"""SQLite application state. Conversations survive restarts and model changes."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Conversation:
    id: str
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Message:
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: str


@dataclass(frozen=True)
class StepRecord:
    title: str
    instruction: str
    result: str | None = None


@dataclass(frozen=True)
class Objective:
    id: str
    request: str
    status: str
    steps: tuple[StepRecord, ...]
    current_step: int
    result: str | None
    error: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ObjectiveEvent:
    id: str
    objective_id: str
    kind: str
    message: str
    created_at: str


@dataclass(frozen=True)
class Resource:
    id: str
    objective_id: str | None
    kind: str
    title: str
    url: str | None
    path: str | None
    sha256: str | None
    metadata_json: str
    retrieved_at: str


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def migrate(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id),
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages (conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS objectives (
                    id TEXT PRIMARY KEY,
                    request TEXT NOT NULL,
                    status TEXT NOT NULL,
                    steps_json TEXT NOT NULL,
                    current_step INTEGER NOT NULL,
                    result TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS objective_events (
                    id TEXT PRIMARY KEY,
                    objective_id TEXT NOT NULL REFERENCES objectives(id),
                    kind TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_objective_events
                    ON objective_events (objective_id, created_at);
                CREATE TABLE IF NOT EXISTS resources (
                    id TEXT PRIMARY KEY,
                    objective_id TEXT REFERENCES objectives(id),
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT,
                    path TEXT,
                    sha256 TEXT,
                    metadata_json TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_resources_objective
                    ON resources (objective_id, retrieved_at);
                """
            )

    def get_setting(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_setting(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, value, utc_now()),
            )

    def clear_setting(self, key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM settings WHERE key = ?", (key,))

    def create_conversation(self, title: str) -> Conversation:
        now = utc_now()
        conversation = Conversation(str(uuid4()), _title(title), now, now)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation.id, conversation.title, conversation.created_at, conversation.updated_at),
            )
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return None if row is None else _conversation(row)

    def list_conversations(self) -> list[Conversation]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [_conversation(row) for row in rows]

    def add_message(self, conversation_id: str, role: str, content: str) -> Message:
        if self.get_conversation(conversation_id) is None:
            raise KeyError(conversation_id)
        message = Message(str(uuid4()), conversation_id, role, content, utc_now())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (message.id, message.conversation_id, message.role, message.content, message.created_at),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (message.created_at, conversation_id),
            )
        return message

    def create_objective(self, request: str) -> Objective:
        now = utc_now()
        objective = Objective(str(uuid4()), request, "pending", (), 0, None, None, now, now)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO objectives (
                    id, request, status, steps_json, current_step, result, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    objective.id,
                    objective.request,
                    objective.status,
                    "[]",
                    0,
                    None,
                    None,
                    objective.created_at,
                    objective.updated_at,
                ),
            )
        return objective

    def get_objective(self, objective_id: str) -> Objective | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM objectives WHERE id = ?", (objective_id,)).fetchone()
        return None if row is None else _objective(row)

    def list_objectives(self) -> list[Objective]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM objectives ORDER BY updated_at DESC, rowid DESC").fetchall()
        return [_objective(row) for row in rows]

    def next_runnable_objective(self) -> Objective | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM objectives
                WHERE status IN ('pending', 'planning', 'executing')
                ORDER BY created_at ASC, rowid ASC
                LIMIT 1
                """
            ).fetchone()
        return None if row is None else _objective(row)

    def save_objective(self, objective: Objective) -> Objective:
        now = utc_now()
        steps_json = json.dumps(
            [
                {"title": step.title, "instruction": step.instruction, "result": step.result}
                for step in objective.steps
            ]
        )
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE objectives
                SET status = ?, steps_json = ?, current_step = ?, result = ?, error = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    objective.status,
                    steps_json,
                    objective.current_step,
                    objective.result,
                    objective.error,
                    now,
                    objective.id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(objective.id)
        return Objective(
            objective.id,
            objective.request,
            objective.status,
            objective.steps,
            objective.current_step,
            objective.result,
            objective.error,
            objective.created_at,
            now,
        )

    def requeue_waiting_objectives(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE objectives SET status = 'pending', updated_at = ? WHERE status = 'waiting'",
                (utc_now(),),
            )

    def add_objective_event(self, objective_id: str, kind: str, message: str) -> ObjectiveEvent:
        if self.get_objective(objective_id) is None:
            raise KeyError(objective_id)
        event = ObjectiveEvent(str(uuid4()), objective_id, kind, message, utc_now())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO objective_events (id, objective_id, kind, message, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event.id, event.objective_id, event.kind, event.message, event.created_at),
            )
        return event

    def list_objective_events(self, objective_id: str) -> list[ObjectiveEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, objective_id, kind, message, created_at
                FROM objective_events
                WHERE objective_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (objective_id,),
            ).fetchall()
        return [
            ObjectiveEvent(
                str(row["id"]),
                str(row["objective_id"]),
                str(row["kind"]),
                str(row["message"]),
                str(row["created_at"]),
            )
            for row in rows
        ]

    def add_resource(
        self,
        objective_id: str | None,
        kind: str,
        title: str,
        url: str | None,
        path: str | None,
        sha256: str | None,
        metadata_json: str,
    ) -> Resource:
        if objective_id is not None and self.get_objective(objective_id) is None:
            raise KeyError(objective_id)
        resource = Resource(
            str(uuid4()),
            objective_id,
            kind,
            title,
            url,
            path,
            sha256,
            metadata_json,
            utc_now(),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO resources (
                    id, objective_id, kind, title, url, path, sha256, metadata_json, retrieved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resource.id,
                    resource.objective_id,
                    resource.kind,
                    resource.title,
                    resource.url,
                    resource.path,
                    resource.sha256,
                    resource.metadata_json,
                    resource.retrieved_at,
                ),
            )
        return resource

    def list_resources(self, objective_id: str | None = None) -> list[Resource]:
        with self._connect() as conn:
            if objective_id is None:
                rows = conn.execute(
                    "SELECT * FROM resources ORDER BY retrieved_at DESC, rowid DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM resources
                    WHERE objective_id = ?
                    ORDER BY retrieved_at ASC, rowid ASC
                    """,
                    (objective_id,),
                ).fetchall()
        return [_resource(row) for row in rows]

    def list_messages(self, conversation_id: str) -> list[Message]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, role, content, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (conversation_id,),
            ).fetchall()
        return [
            Message(
                str(row["id"]),
                str(row["conversation_id"]),
                str(row["role"]),
                str(row["content"]),
                str(row["created_at"]),
            )
            for row in rows
        ]

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _conversation(row: sqlite3.Row) -> Conversation:
    return Conversation(str(row["id"]), str(row["title"]), str(row["created_at"]), str(row["updated_at"]))


def _resource(row: sqlite3.Row) -> Resource:
    objective_id = row["objective_id"]
    url = row["url"]
    path = row["path"]
    digest = row["sha256"]
    return Resource(
        str(row["id"]),
        None if objective_id is None else str(objective_id),
        str(row["kind"]),
        str(row["title"]),
        None if url is None else str(url),
        None if path is None else str(path),
        None if digest is None else str(digest),
        str(row["metadata_json"]),
        str(row["retrieved_at"]),
    )


def _objective(row: sqlite3.Row) -> Objective:
    raw_steps = json.loads(str(row["steps_json"]))
    steps: list[StepRecord] = []
    if isinstance(raw_steps, list):
        for item in raw_steps:
            if not isinstance(item, dict):
                continue
            result = item.get("result")
            steps.append(
                StepRecord(
                    str(item.get("title") or ""),
                    str(item.get("instruction") or ""),
                    None if result is None else str(result),
                )
            )
    result = row["result"]
    error = row["error"]
    return Objective(
        str(row["id"]),
        str(row["request"]),
        str(row["status"]),
        tuple(steps),
        int(row["current_step"]),
        None if result is None else str(result),
        None if error is None else str(error),
        str(row["created_at"]),
        str(row["updated_at"]),
    )


def _title(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) <= 80:
        return compact or "Conversation"
    return compact[:77] + "..."
