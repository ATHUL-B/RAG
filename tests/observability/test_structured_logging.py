import json
import logging

from rag.observability import JsonFormatter, configure_logging


def test_formats_message_and_extra_fields_as_json() -> None:
    record = logging.makeLogRecord(
        {"name": "rag.x", "levelname": "INFO", "msg": "hello %s", "args": ("world",), "k": 3}
    )
    payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "hello world"
    assert payload["logger"] == "rag.x"
    assert payload["k"] == 3
    assert "args" not in payload


def test_configure_logging_installs_single_json_handler() -> None:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        configure_logging(logging.DEBUG)
        configure_logging(logging.DEBUG)  # idempotent: no duplicate handlers
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
        assert root.level == logging.DEBUG
    finally:
        root.handlers[:], root.level = saved_handlers, saved_level
