"""Find GGUF files and decide which one the process should load."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelChoice:
    path: Path | None
    remember_as: str | None
    detail: str | None


def model_key(directory: Path, path: Path) -> str:
    return path.resolve().relative_to(directory.resolve()).as_posix()


def list_gguf(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() == ".gguf"
    ]
    return sorted(files, key=lambda path: model_key(directory, path).lower())


def safe_model_filename(filename: str) -> str:
    raw = filename.replace("\\", "/").strip()
    parts = [part for part in raw.split("/") if part]
    drive = parts[0] if parts else ""
    if not parts or ":" in drive or any(part in {".", ".."} for part in parts):
        raise ValueError("Model filename must be a path inside the models directory.")
    return "/".join(parts)


def model_file(directory: Path, filename: str) -> Path:
    key = safe_model_filename(filename)
    directory_resolved = directory.resolve()
    candidate = (directory_resolved.joinpath(*key.split("/"))).resolve()
    try:
        candidate.relative_to(directory_resolved)
    except ValueError as exc:
        raise ValueError("Model filename must be a path inside the models directory.") from exc
    if not candidate.is_file() or candidate.suffix.lower() != ".gguf":
        raise FileNotFoundError(f"{key} is not a .gguf file in {directory_resolved}")
    return candidate


def resolve_model_choice(
    files: list[Path],
    *,
    directory: Path,
    configured_filename: str | None,
    remembered_filename: str | None,
) -> ModelChoice:
    keyed = {model_key(directory, path): path for path in files}

    def pick(name: str) -> tuple[Path | None, str | None]:
        found = keyed.get(name)
        if found is not None:
            return found, None
        bare = [path for path in files if path.name == name]
        if len(bare) == 1:
            return bare[0], None
        if len(bare) > 1:
            return None, (
                f"Multiple GGUF files are named '{name}'. "
                f"Select one by its path under {directory}."
            )
        return None, None

    if configured_filename:
        match, problem = pick(configured_filename)
        if match is not None:
            return ModelChoice(match, None, None)
        return ModelChoice(
            None,
            None,
            problem or f"Configured model '{configured_filename}' is not in {directory}.",
        )
    if remembered_filename:
        match, _problem = pick(remembered_filename)
        if match is not None:
            return ModelChoice(match, None, None)
    if len(files) == 1:
        only = files[0]
        return ModelChoice(only, model_key(directory, only), None)
    if not files:
        return ModelChoice(
            None,
            None,
            f"No .gguf files in {directory}. Drop a model there and reload.",
        )
    names = ", ".join(model_key(directory, path) for path in files)
    return ModelChoice(
        None,
        None,
        f"Multiple GGUF files found ({names}). Select one with POST /model/select.",
    )
