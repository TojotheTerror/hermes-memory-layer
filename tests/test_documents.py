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

    assert corpus_id == "c4d4d74effa2400fe746960b"
    assert source_id == "e0d9feb51baa7d2f42531c7a2d75babd"
    assert chunk_hash == "6a4d4262f0306776db47a645c6aded1e335074810a37200613315c74b251e397"
    assert chunk_id == "19fe8cd8dec3329177c811bbaa0ccb0d22fb7712"
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
    assert lines[chunk.start_line - 1] == "### Recovery"
    assert lines[chunk.end_line - 1].startswith("A successful recovery")
    with pytest.raises(FrozenInstanceError):
        chunk.end_line = 39
