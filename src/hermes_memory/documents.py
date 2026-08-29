"""Immutable document records and deterministic identity helpers."""

from __future__ import annotations

import hashlib
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypeAlias


SourceKind = Literal["obsidian", "git"]
ContentKind = Literal["markdown", "code", "text"]
MetadataScalar: TypeAlias = str | int | float | bool | None | bytes | bytearray
MetadataValue: TypeAlias = (
    MetadataScalar
    | Mapping[str, "MetadataValue"]
    | list["MetadataValue"]
    | tuple["MetadataValue", ...]
    | set["MetadataValue"]
    | frozenset["MetadataValue"]
)

_METADATA_IMMUTABLE_SCALAR_TYPES = (str, int, float, bool, type(None), bytes)


def _deep_freeze(value: Any) -> Any:
    """Freeze a value from the closed metadata domain.

    Metadata accepts mappings with plain string keys, JSON scalars, bytes and
    bytearray, lists and tuples, and sets and frozensets, recursively. Mutable
    values are normalized to immutable equivalents; arbitrary objects are
    rejected rather than copied because copying cannot guarantee isolation.
    """
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"metadata mapping keys must be strings, got {type(key).__name__}")
            frozen[key] = _deep_freeze(item)
        return MappingProxyType(frozen)
    if isinstance(value, bytearray):
        return bytes(value)
    if type(value) in (list, tuple):
        return tuple(_deep_freeze(item) for item in value)
    if type(value) in (set, frozenset):
        return frozenset(_deep_freeze(item) for item in value)
    if type(value) in _METADATA_IMMUTABLE_SCALAR_TYPES:
        return value
    raise TypeError(f"unsupported metadata value type: {type(value).__name__}")


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
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "heading_path", tuple(self.heading_path))


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
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "heading_path", tuple(self.heading_path))
        if self.embedding is not None:
            object.__setattr__(self, "embedding", tuple(self.embedding))
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
    root = (
        canonical_root_or_remote.rstrip("/")
        if "://" in canonical_root_or_remote
        else _normalize_path(canonical_root_or_remote)
    )
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
