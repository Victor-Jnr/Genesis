"""Load config/genesis.yaml. Paths and sampling stay data, not code."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from genesis.core.runtime import repo_root


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    host: str = "127.0.0.1"
    port: int = Field(default=8765, gt=0, lt=65536)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    directory: str = "models"
    filename: str | None = None
    context_length: int = Field(default=4096, gt=0)
    n_gpu_layers: int = 0
    temperature: float = Field(default=0.7, ge=0)
    top_p: float = Field(default=0.95, ge=0, le=1)
    max_tokens: int = Field(default=1024, gt=0)


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str = "storage/genesis.db"


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    level: str = "INFO"
    directory: str = "storage/logs"


class IdentityConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = "Genesis"
    system_prompt: str = "You are Genesis, a local developmental AI running on this machine."


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)

    def resolve_path(self, value: str, root: Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        return path.resolve()


def default_config_path(root: Path | None = None) -> Path:
    base = root or repo_root()
    override = os.environ.get("GENESIS_CONFIG")
    if override:
        path = Path(override)
        return path if path.is_absolute() else (base / path).resolve()
    return base / "config" / "genesis.yaml"


def load_settings(path: Path | None = None, root: Path | None = None) -> Settings:
    config_path = path or default_config_path(root)
    data: object = {}
    if config_path.is_file():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data = loaded or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    settings = Settings.model_validate(data)
    model_dir = os.environ.get("GENESIS_MODEL_DIR")
    if model_dir:
        settings.model.directory = model_dir
    model_filename = os.environ.get("GENESIS_MODEL_FILENAME")
    if model_filename:
        settings.model.filename = model_filename
    return settings
