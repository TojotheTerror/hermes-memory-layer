"""Immutable document records and deterministic identity helpers."""
from __future__ import annotations

import copy
import hashlib
import posixpath
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal


SourceKind = Literal["obsidian", "git"]
ContentKind = Literal["markdown", "code", "text"]


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, Set):
        return frozenset(_deep_freeze(item) for item in value)
    return copy.deepcopy(value)


@dataclass(frozen=True)
class SourceDocument:
    source_id: str
    corpus_id: str
    source_kind: SourceKind
    content_kind: ContentKind
    root: Path
    path: Path
    relative_path: str
    source_uri: str
    revision: str
    content_hash: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _deep_freeze(self.metadata))


@dataclass(frozen=True)
class AtomicUnit:
    text: str
    heading_path: tuple[str, ...]
    symbol: str | None
    start_line: int
    end_line: int
    token_estimate: int


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source_id: str
    corpus_id: str
    ordinal: int
    text: str
    contextual_text: str
    heading_path: tuple[str, ...]
    symbol: str | None
    start_line: int
    end_line: int
    content_hash: str
    citation: str
    embedding: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _deep_freeze(self.metadata))


def normalize_text(text: str) -> str:
    """Normalize platform line endings before content hashing."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def sha256_text(text: str) -> str:
    """Return the full SHA-256 digest of normalized text."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _sha256_parts(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _normalize_path(value: str) -> str:
    normalized = posixpath.normpath(value.replace("\\", "/"))
    return "" if normalized == "." else normalized


def make_corpus_id(source_kind: SourceKind, canonical_root_or_remote: str) -> str:
    """Build a stable corpus ID from source kind and canonical root identity."""
    root = canonical_root_or_remote.rstrip("/") if "://" in canonical_root_or_remote else _normalize_path(canonical_root_or_remote)
    return _sha256_parts(source_kind, root)[:24]


def make_source_id(corpus_id: str, relative_path: str) -> str:
    """Build a stable source ID from its corpus and normalized relative path."""
    return _sha256_parts(corpus_id, _normalize_path(relative_path))[:32]


def make_chunk_id(
    source_id: str,
    heading_or_symbol: str,
    *,
    occurrence: int,
    chunk_content_hash: str,
) -> str:
    """Build a content-addressed chunk ID independent of ordinal and line shifts."""
    return _sha256_parts(
        source_id,
        heading_or_symbol,
        str(occurrence),
        chunk_content_hash,
    )[:40]
