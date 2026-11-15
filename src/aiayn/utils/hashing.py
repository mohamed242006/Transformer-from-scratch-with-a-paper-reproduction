"""Content hashing for data provenance (checksums, tokenizer fingerprints)."""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["sha256_bytes", "sha256_file"]

_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of an in-memory byte string."""
    return hashlib.sha256(data).hexdigest()
