"""Find GGUF files and decide which one the process should load."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelChoice:
    path: Path | None
    remember_as: str | None
    detail: str | None


def list_gguf(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".gguf"
    ]
    return sorted(files, key=lambda path: path.name.lower())


def safe_model_filename(filename: str) -> str:
    if not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("Model filename must be a file name inside the models directory.")
    return filename


def model_file(directory: Path, filename: str) -> Path:
    name = safe_model_filename(filename)
    directory_resolved = directory.resolve()
    candidate = (directory_resolved / name).resolve()
    inside = os.path.normcase(str(candidate.parent)) == os.path.normcase(str(directory_resolved))
    if not inside or not candidate.is_file() or candidate.suffix.lower() != ".gguf":
        raise FileNotFoundError(f"{name} is not a .gguf file in {directory_resolved}")
    return candidate


def resolve_model_choice(
    files: list[Path],
    *,
    directory: Path,
    configured_filename: str | None,
    remembered_filename: str | None,
) -> ModelChoice:
    by_name = {path.name: path for path in files}
    if configured_filename:
        match = by_name.get(configured_filename)
        if match is not None:
            return ModelChoice(match, None, None)
        return ModelChoice(
            None,
            None,
            f"Configured model '{configured_filename}' is not in {directory}.",
        )
    if remembered_filename and remembered_filename in by_name:
        return ModelChoice(by_name[remembered_filename], None, None)
    if len(files) == 1:
        return ModelChoice(files[0], files[0].name, None)
    if not files:
        return ModelChoice(
            None,
            None,
            f"No .gguf files in {directory}. Drop a model there and reload.",
        )
    names = ", ".join(path.name for path in files)
    return ModelChoice(
        None,
        None,
        f"Multiple GGUF files found ({names}). Select one with POST /model/select.",
    )
