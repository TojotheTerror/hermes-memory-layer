from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from hermes_memory.documents import (
    AtomicUnit,
    DocumentChunk,
    SourceDocument,
    make_chunk_id,
    make_corpus_id,
    make_source_id,
    sha256_text,
)


FIXTURE = Path(__file__).parent / "fixtures" / "obsidian" / "Operations" / "agent-memory.md"


def _source_with_metadata(metadata):
    return SourceDocument(
        source_id="source",
        corpus_id="corpus",
        source_kind="obsidian",
        content_kind="markdown",
        root=Path("/vault"),
        path=Path("/vault/note.md"),
        relative_path="note.md",
        source_uri="obsidian://vault/note.md",
        revision="revision",
        content_hash="hash",
        text="text",
        metadata=metadata,
    )


def _chunk_with_metadata(metadata):
    return DocumentChunk(
        chunk_id="chunk",
        source_id="source",
        corpus_id="corpus",
        ordinal=0,
        text="text",
        contextual_text="context\n\ntext",
        heading_path=("Heading",),
        symbol=None,
        start_line=1,
        end_line=1,
        content_hash="hash",
        citation="note.md#L1",
        metadata=metadata,
    )


@pytest.mark.parametrize("record_factory", [_source_with_metadata, _chunk_with_metadata])
def test_metadata_does_not_retain_caller_aliases(record_factory):
    metadata = {"owner": "original", "labels": ["initial"]}
    record = record_factory(metadata)

    metadata["owner"] = "changed"
    metadata["labels"].append("changed")

    assert record.metadata["owner"] == "original"
    assert tuple(record.metadata["labels"]) == ("initial",)


@pytest.mark.parametrize("record_factory", [_source_with_metadata, _chunk_with_metadata])
def test_metadata_rejects_nested_mutation(record_factory):
    record = record_factory(
        {"nested": {"owner": "original", "labels": ["initial"], "flags": {"safe"}}}
    )

    with pytest.raises(TypeError):
        record.metadata["nested"]["owner"] = "changed"
    with pytest.raises(AttributeError):
        record.metadata["nested"]["labels"].append("changed")
    assert record.metadata["nested"]["flags"] == frozenset({"safe"})


@pytest.mark.parametrize("record_factory", [_source_with_metadata, _chunk_with_metadata])
def test_metadata_rejects_direct_mutation(record_factory):
    record = record_factory({"owner": "original"})

    with pytest.raises(TypeError):
        record.metadata["owner"] = "changed"


def test_document_identity_and_line_ranges_are_deterministic():
    text = FIXTURE.read_text()
    lines = text.splitlines()
    chunk_text = "\n".join(lines[29:38])

    corpus_id = make_corpus_id("obsidian", "/vaults/ops")
    source_id = make_source_id(corpus_id, "./Operations\\agent-memory.md")
    chunk_hash = sha256_text(chunk_text)
    chunk_id = make_chunk_id(
        source_id,
        "Agent Memory Operations > Storage > Recovery",
        occurrence=1,
        chunk_content_hash=chunk_hash,
    )

    assert corpus_id == "fbdc8ba99540a750a3d8d672"
    assert source_id == "8f79b089e8f4da81670cb110aa2a9945"
    assert chunk_hash == "6a4d4262f0306776db47a645c6aded1e335074810a37200613315c74b251e397"
    assert chunk_id == "944cd6177296d1ea147a5e6570b01bd0ef2449af"
    assert sha256_text(chunk_text.replace("\n", "\r\n")) == chunk_hash

    source = SourceDocument(
        source_id=source_id,
        corpus_id=corpus_id,
        source_kind="obsidian",
        content_kind="markdown",
        root=Path("/vaults/ops"),
        path=Path("/vaults/ops/Operations/agent-memory.md"),
        relative_path="Operations/agent-memory.md",
        source_uri="obsidian://ops/Operations/agent-memory.md",
        revision=sha256_text(text),
        content_hash=sha256_text(text),
        text=text,
    )
    unit = AtomicUnit(
        text=chunk_text,
        heading_path=("Agent Memory Operations", "Storage", "Recovery"),
        symbol=None,
        start_line=30,
        end_line=38,
        token_estimate=55,
    )
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        source_id=source.source_id,
        corpus_id=source.corpus_id,
        ordinal=1,
        text=unit.text,
        contextual_text="Agent Memory Operations > Storage > Recovery\n\n" + unit.text,
        heading_path=unit.heading_path,
        symbol=unit.symbol,
        start_line=unit.start_line,
        end_line=unit.end_line,
        content_hash=chunk_hash,
        citation="Operations/agent-memory.md#L30-L38",
    )

    assert (chunk.start_line, chunk.end_line) == (30, 38)
    assert chunk.citation == "Operations/agent-memory.md#L30-L38"
    assert lines[chunk.start_line - 1] == "### Recovery"
    assert lines[chunk.end_line - 1].startswith("A successful recovery")
    with pytest.raises(FrozenInstanceError):
        chunk.end_line = 39


def test_chunk_identity_is_unambiguous_across_variable_width_parts():
    source_id = "a" * 32
    content_hash = "b" * 64

    recovery_one = make_chunk_id(
        source_id,
        "Recovery1",
        occurrence=2,
        chunk_content_hash=content_hash,
    )
    recovery_twelve = make_chunk_id(
        source_id,
        "Recovery",
        occurrence=12,
        chunk_content_hash=content_hash,
    )

    assert recovery_one != recovery_twelve
