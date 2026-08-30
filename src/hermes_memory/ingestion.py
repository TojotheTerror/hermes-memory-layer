"""Incremental Obsidian document ingestion — pure planning and injected apply.

This module computes what an Obsidian ingestion *would* write (planning) and,
separately, applies a plan through injected cloud dependencies. Planning makes
no network calls and constructs no clients; every cloud dependency used by apply
is injectable so the end-to-end path runs entirely against fakes in tests.

The accounting records (``IngestionPlan`` / ``IngestionReport`` and their
per-source detail) are frozen and never carry raw file bodies, embeddings, or
secret content — derived chunk payloads needed for apply are held on planning
records with ``repr=False`` so they never leak through logs or reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .chunking import (
    pack_markdown_units,
    pack_semantic_markdown_units,
    parse_markdown_units,
)
from .config import HermesMemoryConfig, load_config
from .documents import (
    make_chunk_id,
    make_corpus_id,
    make_source_id,
    sha256_text,
)
from .source_discovery import RejectedSource, SourcePolicy, discover_sources


# Public embedding price used only for a rough, body-free cost estimate. Kept as
# a single constant so the estimate is transparent and never implies billing.
_EMBEDDING_COST_PER_MILLION_TOKENS = 0.15

# Default allowlist: Markdown notes only. Non-Markdown vault files (``.obsidian``
# config, attachments) fall to the source-discovery default-deny and are reported
# as rejected rather than silently ingested.
_MARKDOWN_INCLUDE_PATTERNS = ("*.md", "*.markdown", "*.mdown", "*.mkd")

_SOURCE_KIND = "obsidian"
_CONTENT_KIND = "markdown"

StateReader = Callable[..., dict | None]


@dataclass(frozen=True)
class _PlannedChunk:
    """A derived, ready-to-embed chunk. Derived text is repr-suppressed."""

    chunk_id: str
    source_id: str
    corpus_id: str
    ordinal: int
    content_hash: str
    heading_path: tuple[str, ...]
    symbol: str | None
    start_line: int
    end_line: int
    citation: str
    token_estimate: int
    text: str = field(repr=False, compare=False)
    contextual_text: str = field(repr=False, compare=False)


@dataclass(frozen=True)
class PlannedSource:
    """Per-source planning detail. Carries no raw body — only derived chunks."""

    source_id: str
    corpus_id: str
    source_kind: str
    content_kind: str
    relative_path: str
    source_uri: str
    revision: str
    content_hash: str
    status: str
    chunk_count: int
    token_count: int
    prior_revision: str | None = None
    prior_content_hash: str | None = None
    chunks: tuple[_PlannedChunk, ...] = field(default=(), repr=False, compare=False)
    units: tuple = field(default=(), repr=False, compare=False)


@dataclass(frozen=True)
class IngestionPlan:
    """What an Obsidian ingestion would write. Frozen, body-free."""

    discovered: tuple[PlannedSource, ...]
    skipped: tuple[PlannedSource, ...]
    rejected: tuple[RejectedSource, ...]
    chunk_count: int
    request_count: int
    token_count: int
    cost_estimate: float


@dataclass(frozen=True)
class SourceOutcome:
    """Per-source apply outcome. Frozen, body-free.

    ``promotion_status`` is truthful about Memory Bank promotion for THIS
    source only: ``not_requested`` (promotion off), ``not_eligible`` (a
    code/repository source without the explicit opt-in), ``complete`` (the
    promotion call returned), or ``incomplete`` (promotion was requested and
    eligible but could not be completed — the corpus stays valid regardless).
    It never asserts which memories Memory Bank actually created.
    """

    source_id: str
    relative_path: str
    status: str
    chunk_count: int
    token_count: int
    promotion_status: str = "not_requested"


@dataclass(frozen=True)
class IngestionReport:
    """Result of applying a plan. Frozen, body-free."""

    discovered: tuple[SourceOutcome, ...]
    skipped: tuple[SourceOutcome, ...]
    rejected: tuple[RejectedSource, ...]
    chunk_count: int
    request_count: int
    token_count: int
    cost_estimate: float
    promotion_status: str = "not_requested"
    legacy_manifest_entries: int | None = None


def _cost_estimate(token_count: int) -> float:
    return round(token_count / 1_000_000 * _EMBEDDING_COST_PER_MILLION_TOKENS, 9)


def _contextual_text(heading_path: tuple[str, ...], text: str) -> str:
    return "\n".join([*heading_path, text])


def _assemble_chunks(
    packed: list,
    *,
    source_id: str,
    corpus_id: str,
    relative_path: str,
) -> tuple[tuple[_PlannedChunk, ...], int]:
    """Turn packed atomic units into ready-to-embed chunk records."""

    chunks: list[_PlannedChunk] = []
    token_count = 0
    occurrences: dict[str, int] = {}
    for ordinal, unit in enumerate(packed):
        heading_key = unit.symbol or "/".join(unit.heading_path)
        occurrence = occurrences.get(heading_key, 0)
        occurrences[heading_key] = occurrence + 1
        chunk_hash = sha256_text(unit.text)
        chunk_id = make_chunk_id(
            source_id,
            heading_key,
            occurrence=occurrence,
            chunk_content_hash=chunk_hash,
        )
        citation = f"{relative_path}#L{unit.start_line}-L{unit.end_line}"
        chunks.append(
            _PlannedChunk(
                chunk_id=chunk_id,
                source_id=source_id,
                corpus_id=corpus_id,
                ordinal=ordinal,
                content_hash=chunk_hash,
                heading_path=unit.heading_path,
                symbol=unit.symbol,
                start_line=unit.start_line,
                end_line=unit.end_line,
                citation=citation,
                token_estimate=unit.token_estimate,
                text=unit.text,
                contextual_text=_contextual_text(unit.heading_path, unit.text),
            )
        )
        token_count += unit.token_estimate
    return tuple(chunks), token_count


def _plan_source_chunks(
    text: str,
    *,
    source_id: str,
    corpus_id: str,
    relative_path: str,
    cfg: HermesMemoryConfig,
) -> tuple[tuple[_PlannedChunk, ...], int, tuple]:
    """Deterministically chunk a note into ready-to-embed payloads.

    Returns the packed chunks, their token total, and the raw parsed units so a
    semantic re-pack can run at apply time without re-reading the note body.
    """

    units = tuple(parse_markdown_units(text))
    packed = pack_markdown_units(
        list(units),
        target_tokens=cfg.chunk_target_tokens,
        max_tokens=cfg.chunk_max_tokens,
        overlap_tokens=cfg.chunk_overlap_tokens,
    )
    chunks, token_count = _assemble_chunks(
        packed,
        source_id=source_id,
        corpus_id=corpus_id,
        relative_path=relative_path,
    )
    return chunks, token_count, units


def plan_obsidian_ingestion(
    vault_roots: Sequence[str | Path],
    *,
    cfg: HermesMemoryConfig | None = None,
    user_id: str,
    agent_name: str,
    state_reader: StateReader,
    policy: SourcePolicy | None = None,
) -> IngestionPlan:
    """Plan an incremental Obsidian ingestion without any external call.

    Discovers Markdown notes via the source-discovery policy, chunks them
    deterministically, and compares each source's current content hash against
    the stored ``document_sources`` state (via the injected ``state_reader``) to
    decide discovered-vs-skipped. Constructs no clients and makes no network,
    Vertex, or BigQuery calls.
    """

    cfg = cfg or load_config()
    active_policy = policy or SourcePolicy(include_patterns=_MARKDOWN_INCLUDE_PATTERNS)

    discovered: list[PlannedSource] = []
    skipped: list[PlannedSource] = []
    rejected: list[RejectedSource] = []

    for raw_root in vault_roots:
        result = discover_sources(raw_root, active_policy)
        rejected.extend(result.rejected)
        canonical_root = str(result.root)
        corpus_id = make_corpus_id(_SOURCE_KIND, canonical_root)
        for source in result.sources:
            relative_path = source.relative_path
            text = source.content.decode("utf-8")
            content_hash = sha256_text(text)
            source_id = make_source_id(corpus_id, relative_path)
            source_uri = (result.root / relative_path).as_uri()

            state = state_reader(source_id, user_id=user_id, agent_name=agent_name)
            prior_revision = state.get("revision") if state else None
            prior_content_hash = state.get("content_hash") if state else None
            unchanged = bool(
                state and state.get("is_active") and state.get("content_hash") == content_hash
            )

            chunks, token_count, units = _plan_source_chunks(
                text,
                source_id=source_id,
                corpus_id=corpus_id,
                relative_path=relative_path,
                cfg=cfg,
            )
            record = PlannedSource(
                source_id=source_id,
                corpus_id=corpus_id,
                source_kind=_SOURCE_KIND,
                content_kind=_CONTENT_KIND,
                relative_path=relative_path,
                source_uri=source_uri,
                revision=content_hash,
                content_hash=content_hash,
                status="skipped" if unchanged else "discovered",
                chunk_count=0 if unchanged else len(chunks),
                token_count=0 if unchanged else token_count,
                prior_revision=prior_revision,
                prior_content_hash=prior_content_hash,
                chunks=() if unchanged else chunks,
                units=() if unchanged else units,
            )
            if unchanged:
                skipped.append(record)
            else:
                discovered.append(record)

    chunk_count = sum(record.chunk_count for record in discovered)
    token_count = sum(record.token_count for record in discovered)
    return IngestionPlan(
        discovered=tuple(discovered),
        skipped=tuple(skipped),
        rejected=tuple(rejected),
        chunk_count=chunk_count,
        request_count=chunk_count,
        token_count=token_count,
        cost_estimate=_cost_estimate(token_count),
    )


def _embedding_records(
    planned: PlannedSource,
    chunks: tuple[_PlannedChunk, ...],
    vectors: list,
    *,
    embedding_model: str,
    embedding_dimensions: int,
) -> list[dict]:
    """Assemble insert-ready chunk dicts pairing derived chunks with vectors."""

    records: list[dict] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        records.append(
            {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "corpus_id": chunk.corpus_id,
                "source_kind": planned.source_kind,
                "content_kind": planned.content_kind,
                "relative_path": planned.relative_path,
                "ordinal": chunk.ordinal,
                "text": chunk.text,
                "contextual_text": chunk.contextual_text,
                "content_hash": chunk.content_hash,
                "heading_path": chunk.heading_path,
                "symbol": chunk.symbol,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "citation": chunk.citation,
                "embedding": tuple(vector),
                "embedding_model": embedding_model,
                "embedding_dimensions": embedding_dimensions,
                "metadata": {},
            }
        )
    return records


def _source_metadata(planned: PlannedSource, *, promotion_requested: bool = False) -> dict:
    metadata: dict = {}
    if promotion_requested:
        # Truthful scope: record only that promotion was REQUESTED for this
        # source. We never write back a list of "facts" Memory Bank returned,
        # because generation is asynchronous and does not hand back a
        # one-to-one mapping from chunks to created memories.
        metadata["memory_bank_promotion"] = "requested"
    return {
        "source_id": planned.source_id,
        "corpus_id": planned.corpus_id,
        "source_kind": planned.source_kind,
        "content_kind": planned.content_kind,
        "relative_path": planned.relative_path,
        "source_uri": planned.source_uri,
        "revision": planned.revision,
        "content_hash": planned.content_hash,
        "metadata": metadata,
    }


# Source kinds promoted to Memory Bank by default. Personal Markdown notes
# (Obsidian) carry durable preferences worth distilling; code/repository
# sources are kept as retrieval chunks unless the caller opts in explicitly.
_DEFAULT_PROMOTION_KINDS = frozenset({_SOURCE_KIND})


def _promotion_eligible(planned: PlannedSource, *, promote_code_sources: bool) -> bool:
    """Whether a source may be promoted, given the code/repository opt-in."""

    if planned.source_kind in _DEFAULT_PROMOTION_KINDS:
        return True
    return promote_code_sources


def _promotion_payload(planned: PlannedSource, chunk: _PlannedChunk) -> str:
    """Contextual promotion text: provenance + a durable-only instruction.

    The payload names the source kind, path, and heading so generation can
    ground provenance, and explicitly instructs it to keep only durable user
    preferences/principles rather than transient document wording. The raw
    passage follows the instruction — it is context for extraction, never an
    asserted fact.
    """

    heading = " / ".join(chunk.heading_path) if chunk.heading_path else "(document root)"
    return (
        f"Source: {planned.source_kind} document at {planned.relative_path} "
        f"(section: {heading}).\n"
        "Instruction: from the passage below, retain only durable user "
        "preferences and guiding principles as long-lived memory. Do NOT store "
        "transient document wording, phrasing, or one-off details.\n\n"
        f"Passage:\n{chunk.text}"
    )


def _promote_source(
    planned: PlannedSource,
    chunks: tuple[_PlannedChunk, ...],
    *,
    memory_bank_client,
    memory_bank_name: str | None,
    user_id: str,
    agent_name: str,
    cfg: HermesMemoryConfig,
) -> str:
    """Promote one already-written source; never raise, never roll back corpus.

    Returns ``complete`` when the promotion call returns, or ``incomplete``
    when promotion is unconfigured or the call fails. The BigQuery corpus was
    inserted and finalized before this runs, so a failure here leaves it valid.
    """

    if memory_bank_client is None or not memory_bank_name:
        # Requested and eligible but unconfigured: corpus is valid; incomplete.
        return "incomplete"
    texts = [_promotion_payload(planned, chunk) for chunk in chunks]
    if not texts:
        return "complete"
    scope = {"user_id": user_id, "agent_name": agent_name}
    try:
        memory_bank_client(memory_bank_name, texts, scope, cfg=cfg)
    except Exception:
        # Surface as promotion-incomplete rather than raising so the valid
        # document corpus is never rolled back.
        return "incomplete"
    return "complete"


def apply_ingestion_plan(
    plan: IngestionPlan,
    *,
    cfg: HermesMemoryConfig | None = None,
    user_id: str,
    agent_name: str,
    embedding_client,
    insert_chunks: Callable[..., int],
    finalize_source_revision: Callable[..., None],
    semantic_gateway=None,
    memory_bank_client: Callable[..., object] | None = None,
    promote_to_memory_bank: bool = False,
    promote_code_sources: bool = False,
    memory_bank_name: str | None = None,
    legacy_manifest_entries: int | None = None,
) -> IngestionReport:
    """Apply a plan through injected cloud dependencies, per source, in order.

    Per discovered source the order is exactly: validate -> (optional) semantic
    boundary vectors when a ``semantic_gateway`` is provided -> final embedding
    vectors -> insert_chunks -> finalize_source_revision -> optional Memory Bank
    promotion (only when ``promote_to_memory_bank`` is True). Promotion is
    provenance-honest and restricted: Obsidian notes promote by default while
    code/repository sources stay retrieval chunks unless ``promote_code_sources``
    is also set. Promotion runs strictly AFTER the corpus for that source is
    inserted and finalized, so a Memory Bank failure is surfaced as
    promotion-incomplete without rolling back the valid corpus. Skipped and
    rejected sources touch no write path. Every dependency is injected, so no
    real client is constructed and no network call is made.
    """

    cfg = cfg or load_config()
    embedding_model = embedding_client.model
    embedding_dimensions = embedding_client.dimensions

    discovered_outcomes: list[SourceOutcome] = []
    for planned in plan.discovered:
        # validate
        if not planned.chunks:
            raise ValueError(f"discovered source {planned.source_id!r} has no chunks")

        # compute semantic boundary vectors (only when a gateway is provided);
        # this re-packs the source's parsed units through the semantic gateway,
        # which embeds only oversized sections. Absent a gateway, the structural
        # chunks planned deterministically are used as-is.
        chunks = planned.chunks
        if semantic_gateway is not None:
            packed = pack_semantic_markdown_units(
                list(planned.units),
                gateway=semantic_gateway,
                min_tokens=cfg.chunk_min_tokens,
                target_tokens=cfg.chunk_target_tokens,
                max_tokens=cfg.chunk_max_tokens,
                overlap_tokens=cfg.chunk_overlap_tokens,
            )
            chunks, _ = _assemble_chunks(
                packed,
                source_id=planned.source_id,
                corpus_id=planned.corpus_id,
                relative_path=planned.relative_path,
            )

        # final embedding vectors
        results = embedding_client.embed_many([c.text for c in chunks])
        vectors = [tuple(r.values) for r in results]
        records = _embedding_records(
            planned,
            chunks,
            vectors,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
        )

        # Decide promotion eligibility for THIS source before finalizing, so the
        # requested-promotion marker can be recorded in its source metadata.
        eligible = promote_to_memory_bank and _promotion_eligible(
            planned, promote_code_sources=promote_code_sources
        )

        # insert_chunks
        insert_chunks(
            records,
            user_id=user_id,
            agent_name=agent_name,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            cfg=cfg,
        )

        # finalize_source_revision (corpus becomes valid/active here)
        finalize_source_revision(
            planned.source_id,
            [c.chunk_id for c in chunks],
            source=_source_metadata(planned, promotion_requested=eligible),
            user_id=user_id,
            agent_name=agent_name,
            cfg=cfg,
        )

        # optional Memory Bank promotion — strictly AFTER the corpus is valid.
        if not promote_to_memory_bank:
            source_promotion = "not_requested"
        elif not eligible:
            source_promotion = "not_eligible"
        else:
            source_promotion = _promote_source(
                planned,
                chunks,
                memory_bank_client=memory_bank_client,
                memory_bank_name=memory_bank_name,
                user_id=user_id,
                agent_name=agent_name,
                cfg=cfg,
            )

        discovered_outcomes.append(
            SourceOutcome(
                source_id=planned.source_id,
                relative_path=planned.relative_path,
                status="written",
                chunk_count=len(chunks),
                token_count=planned.token_count,
                promotion_status=source_promotion,
            )
        )

    # Report-level promotion status is aggregated truthfully from per-source
    # outcomes: incomplete if any eligible promotion could not complete,
    # complete if at least one completed and none failed, otherwise not_requested.
    promotion_status = _aggregate_promotion_status(promote_to_memory_bank, discovered_outcomes)

    skipped_outcomes = tuple(
        SourceOutcome(
            source_id=record.source_id,
            relative_path=record.relative_path,
            status="skipped",
            chunk_count=0,
            token_count=0,
        )
        for record in plan.skipped
    )

    return IngestionReport(
        discovered=tuple(discovered_outcomes),
        skipped=skipped_outcomes,
        rejected=plan.rejected,
        chunk_count=sum(o.chunk_count for o in discovered_outcomes),
        request_count=sum(o.chunk_count for o in discovered_outcomes),
        token_count=sum(o.token_count for o in discovered_outcomes),
        cost_estimate=_cost_estimate(sum(o.token_count for o in discovered_outcomes)),
        promotion_status=promotion_status,
        legacy_manifest_entries=legacy_manifest_entries,
    )


def _aggregate_promotion_status(promote_to_memory_bank: bool, outcomes: list[SourceOutcome]) -> str:
    """Roll per-source promotion statuses up to a single truthful report status."""

    if not promote_to_memory_bank:
        return "not_requested"
    statuses = {o.promotion_status for o in outcomes}
    if "incomplete" in statuses:
        return "incomplete"
    if "complete" in statuses:
        return "complete"
    # Requested, but nothing was eligible to promote.
    return "not_requested"
