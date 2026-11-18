"""Load and validate a YAML experiment config."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from rag.config.settings import AppConfig
from rag.errors import ConfigError


def load_config(path: Path) -> AppConfig:
    """Read ``path`` and return a validated :class:`AppConfig`.

    Raises:
        ConfigError: the file is missing, is not valid YAML, or fails validation.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid config {path}:\n{exc}") from exc
