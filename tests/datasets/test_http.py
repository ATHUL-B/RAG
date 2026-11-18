import io
import urllib.error
import urllib.request
from email.message import Message
from typing import Any

import pytest

from rag.datasets.http import UrllibFetcher
from rag.errors import DatasetError

URL = "https://example.test/file"


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(URL, code, "err", Message(), io.BytesIO())


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, outcomes: list[Exception | bytes]) -> None:
    """Make urlopen return/raise each scripted outcome in turn."""

    def fake_urlopen(request: Any, timeout: float) -> io.BytesIO:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return io.BytesIO(outcome)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_returns_body(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_urlopen(monkeypatch, [b"ok"])
    assert UrllibFetcher().fetch(URL) == b"ok"


def test_retries_transient_errors_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    outcomes: list[Exception | bytes] = [_http_error(503), urllib.error.URLError("reset"), b"ok"]
    _patch_urlopen(monkeypatch, outcomes)
    fetcher = UrllibFetcher(backoff_base_s=0.5, sleep=sleeps.append)
    assert fetcher.fetch(URL) == b"ok"
    assert sleeps == [0.5, 1.0]


def test_does_not_retry_client_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    _patch_urlopen(monkeypatch, [_http_error(404)])
    with pytest.raises(DatasetError, match="HTTP 404"):
        UrllibFetcher(sleep=sleeps.append).fetch(URL)
    assert sleeps == []


def test_gives_up_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[Exception | bytes] = [_http_error(500), _http_error(500)]
    _patch_urlopen(monkeypatch, errors)
    with pytest.raises(DatasetError, match="HTTP 500"):
        UrllibFetcher(max_attempts=2, sleep=lambda _: None).fetch(URL)


def test_refuses_plain_http() -> None:
    with pytest.raises(DatasetError, match="non-HTTPS"):
        UrllibFetcher().fetch("http://example.test/file")


def test_rejects_invalid_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        UrllibFetcher(max_attempts=0)
