"""Read a module manifest. The core does not gain a new import for each module."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_ID = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
_ENTRY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*:[A-Za-z_][A-Za-z0-9_]*$")


class ModuleError(RuntimeError):
    pass


@dataclass(frozen=True)
class Manifest:
    id: str
    name: str
    version: str
    description: str
    source: str
    dependencies: tuple[str, ...]
    entry_point: str
    input_schema: dict[str, object]
    output_schema: dict[str, object]


def load_manifest(path: Path) -> Manifest:
    if not path.is_file():
        raise ModuleError(f"{path} is not a manifest.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModuleError(f"{path.name} is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ModuleError(f"{path.name} must be a JSON object.")
    module_id = str(payload.get("id") or "").strip()
    name = str(payload.get("name") or "").strip()
    version = str(payload.get("version") or "").strip()
    description = str(payload.get("description") or "").strip()
    entry_point = str(payload.get("entry_point") or "").strip()
    if _ID.fullmatch(module_id) is None:
        raise ModuleError("Module id must be a short lowercase name.")
    if not name or not version or not description:
        raise ModuleError("Manifest requires name, version, and description.")
    if _ENTRY.fullmatch(entry_point) is None:
        raise ModuleError("entry_point must look like module:Class.")
    dependencies = payload.get("dependencies") or []
    if not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies):
        raise ModuleError("dependencies must be a list of strings.")
    return Manifest(
        id=module_id,
        name=name,
        version=version,
        description=description,
        source=str(payload.get("source") or path.parent),
        dependencies=tuple(dependencies),
        entry_point=entry_point,
        input_schema=_schema(payload.get("input_schema")),
        output_schema=_schema(payload.get("output_schema")),
    )


def _schema(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}
