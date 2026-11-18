"""Minimal HTTP fetching behind an interface, with timeouts and retry/backoff.

Kept behind :class:`Fetcher` so dataset code is unit-tested with an in-memory
fake and never touches the network in tests.
"""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable

from rag.errors import DatasetError

logger = logging.getLogger(__name__)

_USER_AGENT = "rag-lab-dataset-downloader"


class Fetcher(ABC):
    @abstractmethod
    def fetch(self, url: str) -> bytes:
        """Return the body at ``url`` or raise :class:`DatasetError`."""


class UrllibFetcher(Fetcher):
    """Standard-library fetcher with exponential backoff on transient failures."""

    _RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        timeout_s: float = 60.0,
        max_attempts: int = 4,
        backoff_base_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._backoff_base_s = backoff_base_s
        self._sleep = sleep

    def fetch(self, url: str) -> bytes:
        if not url.startswith("https://"):
            raise DatasetError(f"refusing non-HTTPS URL: {url}")
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        for attempt in range(1, self._max_attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                    body: bytes = response.read()
                    return body
            except urllib.error.HTTPError as exc:
                if exc.code not in self._RETRYABLE_STATUS or attempt == self._max_attempts:
                    raise DatasetError(f"HTTP {exc.code} fetching {url}") from exc
                reason = f"HTTP {exc.code}"
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == self._max_attempts:
                    raise DatasetError(f"network error fetching {url}: {exc}") from exc
                reason = str(exc)
            delay = self._backoff_base_s * 2 ** (attempt - 1)
            logger.warning(
                "retrying fetch",
                extra={"url": url, "attempt": attempt, "reason": reason, "delay_s": delay},
            )
            self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover
