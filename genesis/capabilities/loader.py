"""Load a module entry point from its own folder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from genesis.capabilities.manifest import ModuleError


def load_capability(directory: Path, entry_point: str) -> object:
    module_name, _, attribute = entry_point.partition(":")
    source = _entry_file(directory, module_name)
    spec = importlib.util.spec_from_file_location(f"genesis_module_{directory.name}_{module_name}", source)
    if spec is None or spec.loader is None:
        raise ModuleError(f"Could not load {source}.")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ModuleError(f"Importing {source.name} failed: {exc}") from exc
    try:
        factory = getattr(module, attribute)
    except AttributeError as exc:
        raise ModuleError(f"{source.name} has no {attribute}.") from exc
    try:
        instance = factory()
    except Exception as exc:
        raise ModuleError(f"Creating {attribute} failed: {exc}") from exc
    if not callable(getattr(instance, "execute", None)):
        raise ModuleError(f"{attribute} must define execute(task, context).")
    return instance


def call_capability(instance: object, task: str, context: dict[str, object]) -> tuple[str, bool]:
    _call_optional(instance, "initialize", context)
    try:
        result = instance.execute(task, context)  # type: ignore[attr-defined]
    except Exception as exc:
        raise ModuleError(f"Module execution failed: {exc}") from exc
    verified = True
    verify = getattr(instance, "verify", None)
    if callable(verify):
        try:
            verified = bool(verify(result))
        except Exception as exc:
            raise ModuleError(f"Module verification failed: {exc}") from exc
    _call_optional(instance, "shutdown")
    return _present(result), verified


def _entry_file(directory: Path, module_name: str) -> Path:
    for candidate in (directory / "src" / f"{module_name}.py", directory / f"{module_name}.py"):
        if candidate.is_file():
            return candidate
    raise ModuleError(f"{directory} has no {module_name}.py entry file.")


def _call_optional(instance: object, name: str, *args: object) -> None:
    method = getattr(instance, name, None)
    if not callable(method):
        return
    try:
        method(*args)
    except Exception as exc:
        raise ModuleError(f"Module {name} failed: {exc}") from exc


def _present(result: object) -> str:
    if isinstance(result, str):
        text = result.strip()
    else:
        text = json.dumps(result, default=str)
    if not text:
        raise ModuleError("Module returned an empty result.")
    return text

