"""Small, safe file-storage helpers shared by datasets and pipeline artifacts."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ValidationError

from rag.errors import StorageError

_SAFE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_key(key: str) -> str:
    """Keys become file names: never let them contain path separators or dots."""
    if not _SAFE_KEY.fullmatch(key):
        raise StorageError(f"unsafe artifact key: {key!r}")
    return key


def atomic_write_bytes(target: Path, body: bytes) -> None:
    """Write to a temp file then rename, so a crash never leaves a half-written file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.write_bytes(body)
    tmp.replace(target)


class JsonArtifactStore[M: BaseModel]:
    """One pretty-printed JSON file per key, holding one validated model.

    Pretty-printing costs disk space but makes artifacts readable by hand, which is
    the point: you must be able to open what the next stage will see.
    """

    def __init__(self, root: Path, model: type[M]) -> None:
        self._root = root
        self._model = model

    @property
    def root(self) -> Path:
        return self._root

    def path(self, key: str) -> Path:
        return self._root / f"{validate_key(key)}.json"

    def exists(self, key: str) -> bool:
        return self.path(key).exists()

    def save(self, key: str, item: M) -> Path:
        target = self.path(key)
        atomic_write_bytes(target, item.model_dump_json(indent=2).encode("utf-8"))
        return target

    def load(self, key: str) -> M:
        path = self.path(key)
        try:
            return self._model.model_validate_json(path.read_bytes())
        except FileNotFoundError as exc:
            raise StorageError(f"no artifact for {key!r} at {path}") from exc
        except ValidationError as exc:
            raise StorageError(f"corrupt artifact {path}: {exc}") from exc

    def ids(self) -> list[str]:
        """Keys of all stored artifacts, sorted."""
        if not self._root.exists():
            return []
        return sorted(p.stem for p in self._root.glob("*.json"))
