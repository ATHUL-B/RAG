"""Structured logging: one JSON object per line, easy to grep and to load later."""

from __future__ import annotations

import json
import logging
from typing import Any, Final

_STANDARD_ATTRS: Final = frozenset(vars(logging.makeLogRecord({})).keys()) | {"message"}


class JsonFormatter(logging.Formatter):
    """Formats a record as JSON, including any ``extra={...}`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure the root logger once, for entry-point scripts (not libraries)."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
