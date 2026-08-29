from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from hermes_memory.config import HermesMemoryConfig, load_config
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


def test_document_embedding_defaults_do_not_migrate_memory_bank(monkeypatch):
    for variable in (
        "MEMORY_EMBEDDING_MODEL",
        "DOCUMENT_EMBEDDING_MODEL",
        "DOCUMENT_EMBEDDING_DIMENSIONS",
    ):
        monkeypatch.delenv(variable, raising=False)

    config = load_config(project="test-project")

    assert config.embedding_model == "text-embedding-005"
    assert config.document_embedding_model == "gemini-embedding-001"
    assert config.document_embedding_dimensions == 768


def test_document_ingestion_defaults_match_the_plan(monkeypatch):
    for variable in (
        "DOCUMENT_CHUNK_MIN_TOKENS",
        "DOCUMENT_CHUNK_TARGET_TOKENS",
        "DOCUMENT_CHUNK_MAX_TOKENS",
        "DOCUMENT_CHUNK_OVERLAP_TOKENS",
        "DOCUMENT_EMBEDDING_CONCURRENCY",
        "DOCUMENT_TOP_K",
        "DOCUMENT_CONTEXT_CHAR_LIMIT",
    ):
        monkeypatch.delenv(variable, raising=False)

    config = load_config(project="test-project")

    assert config.chunk_min_tokens == 250
    assert config.chunk_target_tokens == 600
    assert config.chunk_max_tokens == 900
    assert config.chunk_overlap_tokens == 80
    assert config.embedding_concurrency == 4
    assert config.document_top_k == 4
    assert config.document_context_char_limit == 8_000


def test_document_ingestion_settings_load_from_document_environment(monkeypatch):
    environment = {
        "DOCUMENT_EMBEDDING_MODEL": "custom-document-model",
        "DOCUMENT_EMBEDDING_DIMENSIONS": "256",
        "DOCUMENT_CHUNK_MIN_TOKENS": "100",
        "DOCUMENT_CHUNK_TARGET_TOKENS": "200",
        "DOCUMENT_CHUNK_MAX_TOKENS": "300",
        "DOCUMENT_CHUNK_OVERLAP_TOKENS": "50",
        "DOCUMENT_EMBEDDING_CONCURRENCY": "2",
        "DOCUMENT_TOP_K": "7",
        "DOCUMENT_CONTEXT_CHAR_LIMIT": "1234",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    config = load_config(project="test-project")

    assert config.document_embedding_model == "custom-document-model"
    assert config.document_embedding_dimensions == 256
    assert config.chunk_min_tokens == 100
    assert config.chunk_target_tokens == 200
    assert config.chunk_max_tokens == 300
    assert config.chunk_overlap_tokens == 50
    assert config.embedding_concurrency == 2
    assert config.document_top_k == 7
    assert config.document_context_char_limit == 1234


@pytest.mark.parametrize(
    ("setting", "environment", "environment_value", "expected"),
    [
        ("document_embedding_model", "DOCUMENT_EMBEDDING_MODEL", "env-model", "env-model"),
        ("document_embedding_dimensions", "DOCUMENT_EMBEDDING_DIMENSIONS", "256", 256),
        ("chunk_min_tokens", "DOCUMENT_CHUNK_MIN_TOKENS", "200", 200),
        ("chunk_target_tokens", "DOCUMENT_CHUNK_TARGET_TOKENS", "500", 500),
        ("chunk_max_tokens", "DOCUMENT_CHUNK_MAX_TOKENS", "1000", 1000),
        ("chunk_overlap_tokens", "DOCUMENT_CHUNK_OVERLAP_TOKENS", "70", 70),
        ("embedding_concurrency", "DOCUMENT_EMBEDDING_CONCURRENCY", "2", 2),
        ("document_top_k", "DOCUMENT_TOP_K", "3", 3),
        ("document_context_char_limit", "DOCUMENT_CONTEXT_CHAR_LIMIT", "7000", 7000),
    ],
)
@pytest.mark.parametrize("invalid_override", [None, ""])
def test_explicit_invalid_document_override_does_not_fall_back_to_environment(
    monkeypatch, setting, environment, environment_value, expected, invalid_override
):
    monkeypatch.setenv(environment, environment_value)

    config = load_config(project="test-project")
    assert getattr(config, setting) == expected

    with pytest.raises(ValueError, match=setting):
        load_config(project="test-project", **{setting: invalid_override})


@pytest.mark.parametrize("empty_override", [None, ""])
def test_legacy_embedding_model_empty_override_keeps_existing_precedence(
    monkeypatch, empty_override
):
    monkeypatch.setenv("MEMORY_EMBEDDING_MODEL", "legacy-env-model")

    config = load_config(project="test-project", embedding_model=empty_override)

    assert config.embedding_model == "legacy-env-model"


def test_document_chunk_minimum_cannot_exceed_target():
    with pytest.raises(
        ValueError,
        match="chunk_min_tokens must be less than or equal to chunk_target_tokens",
    ):
        load_config(project="test-project", chunk_min_tokens=601)


def test_document_chunk_target_cannot_exceed_maximum():
    with pytest.raises(
        ValueError,
        match="chunk_target_tokens must be less than or equal to chunk_max_tokens",
    ):
        load_config(project="test-project", chunk_target_tokens=901)


def test_document_chunk_overlap_must_be_less_than_minimum():
    with pytest.raises(
        ValueError,
        match="chunk_overlap_tokens must be less than chunk_min_tokens",
    ):
        load_config(project="test-project", chunk_overlap_tokens=250)


@pytest.mark.parametrize("invalid_dimensions", ["0", "-1"])
def test_document_embedding_dimensions_from_environment_must_be_positive(
    monkeypatch, invalid_dimensions
):
    monkeypatch.setenv("DOCUMENT_EMBEDDING_DIMENSIONS", invalid_dimensions)

    with pytest.raises(
        ValueError,
        match="document_embedding_dimensions must be greater than zero",
    ):
        load_config(project="test-project")


def test_explicit_zero_document_embedding_dimensions_is_not_replaced(monkeypatch):
    monkeypatch.delenv("DOCUMENT_EMBEDDING_DIMENSIONS", raising=False)

    with pytest.raises(
        ValueError,
        match="document_embedding_dimensions must be greater than zero",
    ):
        load_config(project="test-project", document_embedding_dimensions=0)


DOCUMENT_NUMERIC_SETTINGS = (
    "document_embedding_dimensions",
    "chunk_min_tokens",
    "chunk_target_tokens",
    "chunk_max_tokens",
    "chunk_overlap_tokens",
    "embedding_concurrency",
    "document_top_k",
    "document_context_char_limit",
)


@pytest.mark.parametrize("setting", DOCUMENT_NUMERIC_SETTINGS)
@pytest.mark.parametrize("invalid_value", [True, "1"])
def test_direct_document_numeric_settings_require_actual_integers(setting, invalid_value):
    with pytest.raises(TypeError, match=rf"{setting} must be an integer"):
        HermesMemoryConfig(project="test-project", **{setting: invalid_value})


@pytest.mark.parametrize(
    "setting",
    [
        "document_embedding_dimensions",
        "chunk_min_tokens",
        "chunk_target_tokens",
        "chunk_max_tokens",
        "embedding_concurrency",
        "document_top_k",
        "document_context_char_limit",
    ],
)
@pytest.mark.parametrize("invalid_value", [0, -1])
def test_direct_positive_document_numeric_settings_reject_non_positive_values(
    setting, invalid_value
):
    with pytest.raises(ValueError, match=rf"{setting} must be greater than zero"):
        HermesMemoryConfig(project="test-project", **{setting: invalid_value})


def test_direct_document_chunk_overlap_rejects_negative_values():
    with pytest.raises(ValueError, match="chunk_overlap_tokens must be non-negative"):
        HermesMemoryConfig(project="test-project", chunk_overlap_tokens=-1)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"chunk_min_tokens": 601},
            "chunk_min_tokens must be less than or equal to chunk_target_tokens",
        ),
        (
            {"chunk_target_tokens": 901},
            "chunk_target_tokens must be less than or equal to chunk_max_tokens",
        ),
        (
            {"chunk_overlap_tokens": 250},
            "chunk_overlap_tokens must be less than chunk_min_tokens",
        ),
    ],
)
def test_direct_document_chunk_relationships_are_validated(overrides, message):
    with pytest.raises(ValueError, match=message):
        HermesMemoryConfig(project="test-project", **overrides)


def test_direct_document_config_defaults_are_valid():
    config = HermesMemoryConfig(project="test-project")

    assert config.chunk_min_tokens <= config.chunk_target_tokens <= config.chunk_max_tokens
    assert 0 <= config.chunk_overlap_tokens < config.chunk_min_tokens


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


def _chunk_with_metadata(metadata, *, heading_path=("Heading",), embedding=None):
    return DocumentChunk(
        chunk_id="chunk",
        source_id="source",
        corpus_id="corpus",
        ordinal=0,
        text="text",
        contextual_text="context\n\ntext",
        heading_path=heading_path,
        symbol=None,
        start_line=1,
        end_line=1,
        content_hash="hash",
        citation="note.md#L1",
        embedding=embedding,
        metadata=metadata,
    )


class _SelfAliasingMutable:
    def __init__(self):
        self.values = []

    def __deepcopy__(self, memo):
        return self


@pytest.mark.parametrize("record_factory", [_source_with_metadata, _chunk_with_metadata])
def test_metadata_rejects_unsupported_self_aliasing_mutable_objects(record_factory):
    unsupported = _SelfAliasingMutable()

    with pytest.raises(
        TypeError,
        match="unsupported metadata value type: _SelfAliasingMutable",
    ):
        record_factory({"unsupported": unsupported})


def test_atomic_unit_normalizes_heading_path_without_caller_aliases():
    heading_path = ["Heading"]
    unit = AtomicUnit(
        text="text",
        heading_path=heading_path,
        symbol=None,
        start_line=1,
        end_line=1,
        token_estimate=1,
    )

    heading_path.append("Changed")

    assert unit.heading_path == ("Heading",)
    assert isinstance(unit.heading_path, tuple)


def test_document_chunk_normalizes_heading_path_without_caller_aliases():
    heading_path = ["Heading"]
    chunk = _chunk_with_metadata({}, heading_path=heading_path)

    heading_path.append("Changed")

    assert chunk.heading_path == ("Heading",)
    assert isinstance(chunk.heading_path, tuple)


def test_document_chunk_normalizes_embedding_without_caller_aliases():
    embedding = [0.25, 0.75]
    chunk = _chunk_with_metadata({}, embedding=embedding)

    embedding[0] = 1.0

    assert chunk.embedding == (0.25, 0.75)
    assert isinstance(chunk.embedding, tuple)


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


def test_source_document_freezes_bytearray_metadata_without_caller_aliases():
    buffer = bytearray(b"source")
    source = _source_with_metadata({"buffer": buffer})

    buffer[0] = ord("S")

    assert source.metadata["buffer"] == b"source"
    with pytest.raises(TypeError):
        source.metadata["buffer"][0] = ord("S")
    assert isinstance(source.metadata["buffer"], bytes)


def test_document_chunk_freezes_bytearray_metadata_without_caller_aliases():
    buffer = bytearray(b"chunk")
    chunk = _chunk_with_metadata({"buffer": buffer})

    buffer[0] = ord("C")

    assert chunk.metadata["buffer"] == b"chunk"
    with pytest.raises(TypeError):
        chunk.metadata["buffer"][0] = ord("C")
    assert isinstance(chunk.metadata["buffer"], bytes)


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
