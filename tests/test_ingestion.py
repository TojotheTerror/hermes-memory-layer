"""Incremental Obsidian document ingestion — planning and apply."""

from __future__ import annotations

from pathlib import Path

from hermes_memory import ingestion
from hermes_memory.config import HermesMemoryConfig
from hermes_memory.documents import make_corpus_id, make_source_id, sha256_text


CFG = HermesMemoryConfig(project="test-project", bq_dataset="test_dataset")
USER_ID = "tojo"
AGENT_NAME = "hermes"

NOTE_BODY = """# Daily Note

This is a substantial paragraph of genuine note content that comfortably clears
any minimum-length gate so the discovery policy accepts it as a real source.

## Section Two

A second section with more prose so the note produces at least one packed chunk
carrying real derived text ready for embedding downstream.
"""


def _write_note(root: Path, relative: str, body: str = NOTE_BODY) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _fresh_reader(calls: list | None = None):
    """State reader that reports every source as never-before-seen."""

    def reader(source_id, *, user_id, agent_name):
        if calls is not None:
            calls.append((source_id, user_id, agent_name))
        return None

    return reader


# --- Slice A: pure planning discovers a fresh note --------------------------


def test_plan_discovers_fresh_note_with_body_free_accounting(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    calls: list = []

    plan = ingestion.plan_obsidian_ingestion(
        [root],
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        state_reader=_fresh_reader(calls),
    )

    assert len(plan.discovered) == 1
    assert plan.skipped == ()
    assert plan.rejected == ()
    planned = plan.discovered[0]
    assert planned.relative_path == "notes/daily.md"
    assert planned.status == "discovered"
    assert planned.chunk_count >= 1
    assert planned.token_count > 0

    corpus_id = make_corpus_id("obsidian", str(root.resolve()))
    expected_source_id = make_source_id(corpus_id, "notes/daily.md")
    assert planned.source_id == expected_source_id
    assert planned.content_hash == sha256_text(NOTE_BODY)

    # roll-up accounting
    assert plan.chunk_count == planned.chunk_count
    assert plan.request_count == plan.chunk_count
    assert plan.token_count == planned.token_count
    assert plan.cost_estimate > 0

    # state reader consulted with the computed identity
    assert (expected_source_id, USER_ID, AGENT_NAME) in calls

    # never leak the note body through repr of the accounting dataclasses
    assert "substantial paragraph" not in repr(plan)
    assert "substantial paragraph" not in repr(planned)


# --- Slice B: skip semantics via injected state reader ----------------------


def _stateful_reader(state: dict):
    """State reader backed by a dict of source_id -> stored row."""

    def reader(source_id, *, user_id, agent_name):
        return state.get(source_id)

    return reader


def test_plan_skips_unchanged_source_by_content_hash_match(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    corpus_id = make_corpus_id("obsidian", str(root.resolve()))
    source_id = make_source_id(corpus_id, "notes/daily.md")
    content_hash = sha256_text(NOTE_BODY)
    state = {
        source_id: {
            "source_id": source_id,
            "revision": content_hash,
            "content_hash": content_hash,
            "is_active": True,
        }
    }

    plan = ingestion.plan_obsidian_ingestion(
        [root],
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        state_reader=_stateful_reader(state),
    )

    assert plan.discovered == ()
    assert len(plan.skipped) == 1
    skipped = plan.skipped[0]
    assert skipped.status == "skipped"
    assert skipped.chunk_count == 0
    assert skipped.token_count == 0
    assert skipped.prior_content_hash == content_hash
    # nothing to write -> zero roll-up
    assert plan.chunk_count == 0
    assert plan.request_count == 0
    assert plan.token_count == 0
    assert plan.cost_estimate == 0


def test_plan_rediscovers_source_when_content_hash_differs(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    corpus_id = make_corpus_id("obsidian", str(root.resolve()))
    source_id = make_source_id(corpus_id, "notes/daily.md")
    state = {
        source_id: {
            "source_id": source_id,
            "revision": "old-hash",
            "content_hash": "old-hash",
            "is_active": True,
        }
    }

    plan = ingestion.plan_obsidian_ingestion(
        [root],
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        state_reader=_stateful_reader(state),
    )

    assert plan.skipped == ()
    assert len(plan.discovered) == 1
    discovered = plan.discovered[0]
    assert discovered.status == "discovered"
    assert discovered.prior_content_hash == "old-hash"
    assert discovered.content_hash == sha256_text(NOTE_BODY)
    assert plan.chunk_count >= 1


def test_plan_rediscovers_source_that_exists_but_is_inactive(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    corpus_id = make_corpus_id("obsidian", str(root.resolve()))
    source_id = make_source_id(corpus_id, "notes/daily.md")
    content_hash = sha256_text(NOTE_BODY)
    # same hash but deactivated -> must be re-ingested, not skipped
    state = {
        source_id: {
            "source_id": source_id,
            "revision": content_hash,
            "content_hash": content_hash,
            "is_active": False,
        }
    }

    plan = ingestion.plan_obsidian_ingestion(
        [root],
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        state_reader=_stateful_reader(state),
    )

    assert plan.skipped == ()
    assert len(plan.discovered) == 1


# --- Slice C: apply executes strict per-source order ------------------------


class _FakeEmbeddingClient:
    """Records embed calls; returns deterministic fixed-dimension vectors."""

    def __init__(self, dimensions=3, model="test-embedding-model"):
        self.dimensions = dimensions
        self.model = model
        self.embed_calls: list[list[str]] = []

    def embed_many(self, texts):
        texts = list(texts)
        self.embed_calls.append(texts)
        return [
            _FakeEmbeddingResult(tuple(float(i + 1) for i in range(self.dimensions))) for _ in texts
        ]


class _FakeEmbeddingResult:
    def __init__(self, values):
        self.values = values


class _Recorder:
    """Injected cloud fns recording call order and arguments."""

    def __init__(self):
        self.order: list[str] = []
        self.inserted: list[dict] = []
        self.finalized: list[tuple] = []

    def insert_chunks(
        self, chunks, *, user_id, agent_name, embedding_model, embedding_dimensions, cfg
    ):
        self.order.append("insert_chunks")
        self.inserted.extend(chunks)
        return len(chunks)

    def finalize_source_revision(
        self, source_id, active_chunk_ids, *, source, user_id, agent_name, cfg
    ):
        self.order.append("finalize_source_revision")
        self.finalized.append((source_id, tuple(active_chunk_ids), source))


def test_apply_runs_validate_embed_insert_finalize_in_order(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    plan = ingestion.plan_obsidian_ingestion(
        [root],
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        state_reader=_fresh_reader(),
    )
    embedder = _FakeEmbeddingClient()
    rec = _Recorder()

    report = ingestion.apply_ingestion_plan(
        plan,
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        embedding_client=embedder,
        insert_chunks=rec.insert_chunks,
        finalize_source_revision=rec.finalize_source_revision,
    )

    assert rec.order == ["insert_chunks", "finalize_source_revision"]
    assert embedder.embed_calls, "embedding client must be called"

    assert rec.inserted
    for chunk in rec.inserted:
        assert len(chunk["embedding"]) == embedder.dimensions
        assert chunk["embedding_model"] == embedder.model
        assert chunk["embedding_dimensions"] == embedder.dimensions
        assert chunk["source_kind"] == "obsidian"
        assert chunk["content_kind"] == "markdown"
        assert chunk["relative_path"] == "notes/daily.md"

    assert len(rec.finalized) == 1
    source_id, active_ids, source = rec.finalized[0]
    assert set(active_ids) == {c["chunk_id"] for c in rec.inserted}
    assert source["source_id"] == source_id
    assert source["content_hash"] == sha256_text(NOTE_BODY)

    assert report.chunk_count == len(rec.inserted)
    assert len(report.discovered) == 1
    assert report.discovered[0].status == "written"
    assert report.promotion_status == "not_requested"
    assert "substantial paragraph" not in repr(report)


def test_apply_skips_write_path_for_skipped_sources(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    corpus_id = make_corpus_id("obsidian", str(root.resolve()))
    source_id = make_source_id(corpus_id, "notes/daily.md")
    content_hash = sha256_text(NOTE_BODY)
    state = {
        source_id: {
            "source_id": source_id,
            "revision": content_hash,
            "content_hash": content_hash,
            "is_active": True,
        }
    }
    plan = ingestion.plan_obsidian_ingestion(
        [root],
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        state_reader=_stateful_reader(state),
    )
    embedder = _FakeEmbeddingClient()
    rec = _Recorder()

    report = ingestion.apply_ingestion_plan(
        plan,
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        embedding_client=embedder,
        insert_chunks=rec.insert_chunks,
        finalize_source_revision=rec.finalize_source_revision,
    )

    assert embedder.embed_calls == []
    assert rec.order == []
    assert report.chunk_count == 0
    assert len(report.skipped) == 1
    assert report.skipped[0].status == "skipped"


# --- Slice D: authoritative incremental semantics ---------------------------


class _FakeStore:
    """In-memory document_sources authority wiring plan + apply together.

    ``state_reader`` reads the active source row; ``finalize_source_revision``
    writes the active revision (mirroring the real completeness-before-activation
    contract). This lets run-1 -> run-2 incrementality be exercised end to end
    against fakes with no network or client construction.
    """

    def __init__(self):
        self.sources: dict[str, dict] = {}
        self.insert_calls = 0
        self.finalize_calls = 0
        self.inserted_chunk_ids: list[str] = []

    def state_reader(self, source_id, *, user_id, agent_name):
        return self.sources.get(source_id)

    def insert_chunks(
        self, chunks, *, user_id, agent_name, embedding_model, embedding_dimensions, cfg
    ):
        self.insert_calls += 1
        self.inserted_chunk_ids.extend(c["chunk_id"] for c in chunks)
        return len(chunks)

    def finalize_source_revision(
        self, source_id, active_chunk_ids, *, source, user_id, agent_name, cfg
    ):
        self.finalize_calls += 1
        self.sources[source_id] = {
            "source_id": source_id,
            "revision": source["revision"],
            "content_hash": source["content_hash"],
            "is_active": True,
        }


def _run(root, store, embedder):
    plan = ingestion.plan_obsidian_ingestion(
        [root],
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        state_reader=store.state_reader,
    )
    report = ingestion.apply_ingestion_plan(
        plan,
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        embedding_client=embedder,
        insert_chunks=store.insert_chunks,
        finalize_source_revision=store.finalize_source_revision,
    )
    return plan, report


def test_incremental_run1_writes_run2_unchanged_is_noop(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    store = _FakeStore()
    embedder = _FakeEmbeddingClient()

    # run 1 — fresh note: writes chunks (insert + finalize called)
    plan1, report1 = _run(root, store, embedder)
    assert len(plan1.discovered) == 1
    assert store.insert_calls == 1
    assert store.finalize_calls == 1
    assert report1.chunk_count >= 1
    embed_calls_after_run1 = len(embedder.embed_calls)
    assert embed_calls_after_run1 >= 1

    # run 2 — UNCHANGED note: NO Vertex embedding calls, NO BigQuery writes
    plan2, report2 = _run(root, store, embedder)
    assert plan2.discovered == ()
    assert len(plan2.skipped) == 1
    assert store.insert_calls == 1, "no new insert on unchanged run"
    assert store.finalize_calls == 1, "no new finalize on unchanged run"
    assert len(embedder.embed_calls) == embed_calls_after_run1, "no new embeddings"
    assert report2.chunk_count == 0


def test_incremental_changed_note_replaces_only_that_source(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/alpha.md")
    _write_note(root, "notes/beta.md")
    store = _FakeStore()
    embedder = _FakeEmbeddingClient()

    # run 1 — both fresh
    _run(root, store, embedder)
    assert store.finalize_calls == 2
    inserts_after_run1 = store.insert_calls
    embeds_after_run1 = len(embedder.embed_calls)

    corpus_id = make_corpus_id("obsidian", str(root.resolve()))
    alpha_id = make_source_id(corpus_id, "notes/alpha.md")
    beta_id = make_source_id(corpus_id, "notes/beta.md")
    beta_revision_before = store.sources[beta_id]["revision"]

    # change ONLY alpha
    _write_note(root, "notes/alpha.md", NOTE_BODY + "\n\n## Added\n\nBrand new material.\n")

    plan2, report2 = _run(root, store, embedder)

    # exactly one source rediscovered (alpha), beta skipped
    assert {p.source_id for p in plan2.discovered} == {alpha_id}
    assert {p.source_id for p in plan2.skipped} == {beta_id}
    # only one more insert + finalize (alpha), beta untouched
    assert store.insert_calls == inserts_after_run1 + 1
    assert store.finalize_calls == 3
    assert len(embedder.embed_calls) > embeds_after_run1
    # beta's stored revision is unchanged
    assert store.sources[beta_id]["revision"] == beta_revision_before
    # alpha's stored revision advanced to the new content hash
    assert store.sources[alpha_id]["revision"] != beta_revision_before
    assert report2.chunk_count >= 1
    assert len(report2.skipped) == 1


# --- Slice E: semantic gateway boundary path --------------------------------


class _FakeSemanticGateway:
    """Semantic gateway recording calls; unit-vector embeddings by length."""

    task_type = "SEMANTIC_SIMILARITY"

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed_many(self, texts):
        texts = list(texts)
        self.calls.append(texts)
        # deterministic, finite, same-dimension vectors
        return [_FakeEmbeddingResult((1.0, float(len(t) % 7 + 1), 2.0)) for t in texts]


# One section whose prose exceeds chunk_max_tokens (900 tokens ~= 3600 chars),
# forcing pack_semantic_markdown_units down the gateway path.
_OVERSIZED_BODY = "# Big Note\n\n" + ("word " * 1500) + "\n"


def test_apply_uses_semantic_gateway_only_when_provided(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/big.md", _OVERSIZED_BODY)
    embedder = _FakeEmbeddingClient()
    gateway = _FakeSemanticGateway()

    plan = ingestion.plan_obsidian_ingestion(
        [root],
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        state_reader=_fresh_reader(),
    )
    rec = _Recorder()
    ingestion.apply_ingestion_plan(
        plan,
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        embedding_client=embedder,
        insert_chunks=rec.insert_chunks,
        finalize_source_revision=rec.finalize_source_revision,
        semantic_gateway=gateway,
    )

    # gateway consulted to compute semantic boundaries
    assert gateway.calls, "semantic gateway must be called when provided"
    # still embeds + writes
    assert rec.order == ["insert_chunks", "finalize_source_revision"]
    assert rec.inserted


def test_apply_makes_no_semantic_calls_when_gateway_absent(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/big.md", _OVERSIZED_BODY)
    embedder = _FakeEmbeddingClient()
    gateway = _FakeSemanticGateway()

    plan = ingestion.plan_obsidian_ingestion(
        [root],
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        state_reader=_fresh_reader(),
    )
    rec = _Recorder()
    ingestion.apply_ingestion_plan(
        plan,
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        embedding_client=embedder,
        insert_chunks=rec.insert_chunks,
        finalize_source_revision=rec.finalize_source_revision,
        # no semantic_gateway
    )

    assert gateway.calls == []
    assert rec.order == ["insert_chunks", "finalize_source_revision"]
    assert rec.inserted


# --- Slice F: optional Memory Bank promotion --------------------------------


def test_promotion_not_attempted_unless_requested(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    store = _FakeStore()
    embedder = _FakeEmbeddingClient()
    calls = []

    def bank(name, texts, scope, *, cfg):
        calls.append((name, texts, scope))

    plan = ingestion.plan_obsidian_ingestion(
        [root],
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        state_reader=store.state_reader,
    )
    report = ingestion.apply_ingestion_plan(
        plan,
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        embedding_client=embedder,
        insert_chunks=store.insert_chunks,
        finalize_source_revision=store.finalize_source_revision,
        memory_bank_client=bank,
        memory_bank_name="projects/p/locations/l/reasoningEngines/1",
        # promote_to_memory_bank defaults False
    )
    assert calls == []
    assert report.promotion_status == "not_requested"
    # corpus written regardless
    assert store.finalize_calls == 1


def test_promotion_runs_when_explicitly_requested(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    store = _FakeStore()
    embedder = _FakeEmbeddingClient()
    calls = []

    def bank(name, texts, scope, *, cfg):
        calls.append((name, texts, scope))

    plan = ingestion.plan_obsidian_ingestion(
        [root],
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        state_reader=store.state_reader,
    )
    report = ingestion.apply_ingestion_plan(
        plan,
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        embedding_client=embedder,
        insert_chunks=store.insert_chunks,
        finalize_source_revision=store.finalize_source_revision,
        memory_bank_client=bank,
        memory_bank_name="projects/p/locations/l/reasoningEngines/1",
        promote_to_memory_bank=True,
    )
    assert len(calls) == 1
    assert report.promotion_status == "complete"
    assert store.finalize_calls == 1


def test_promotion_failure_leaves_corpus_valid_and_marks_incomplete(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    store = _FakeStore()
    embedder = _FakeEmbeddingClient()

    def failing_bank(name, texts, scope, *, cfg):
        raise RuntimeError("memory bank unavailable")

    plan = ingestion.plan_obsidian_ingestion(
        [root],
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        state_reader=store.state_reader,
    )
    report = ingestion.apply_ingestion_plan(
        plan,
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        embedding_client=embedder,
        insert_chunks=store.insert_chunks,
        finalize_source_revision=store.finalize_source_revision,
        memory_bank_client=failing_bank,
        memory_bank_name="projects/p/locations/l/reasoningEngines/1",
        promote_to_memory_bank=True,
    )
    # promotion failure does NOT raise; corpus stays finalized/valid
    assert store.finalize_calls == 1
    assert report.chunk_count >= 1
    assert report.promotion_status == "incomplete"


# --- Slice G: plan/apply construct no real clients, make no network calls ----


def test_plan_and_apply_never_construct_real_cloud_clients(tmp_path, monkeypatch):
    from hermes_memory import bigquery_store, config, memory_bank

    def _boom(*a, **k):
        raise AssertionError("real cloud client / network access attempted")

    # any real client construction or network entrypoint must never be reached
    monkeypatch.setattr(bigquery_store, "_bq_client", _boom)
    monkeypatch.setattr(config, "get_vertex_client", _boom)
    monkeypatch.setattr(memory_bank, "get_vertex_client", _boom, raising=False)

    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    store = _FakeStore()
    embedder = _FakeEmbeddingClient()

    plan = ingestion.plan_obsidian_ingestion(
        [root],
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        state_reader=store.state_reader,
    )
    report = ingestion.apply_ingestion_plan(
        plan,
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        embedding_client=embedder,
        insert_chunks=store.insert_chunks,
        finalize_source_revision=store.finalize_source_revision,
    )
    # got here without tripping the boom guards
    assert report.chunk_count >= 1


# --- Slice H: hermes_bridge document_sources authority ----------------------


def test_bridge_uses_document_sources_authority_not_legacy_manifest(tmp_path):
    from hermes_memory import hermes_bridge

    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    note_path = str((root / "notes/daily.md").resolve())

    # Legacy v1 manifest says this note was already ingested (matching hash).
    # Under v1 semantics that would SKIP it. Task 11 must ignore the manifest for
    # gating and use document_sources (empty here) as the authority -> WRITE.
    manifest_reads = []

    def manifest_loader():
        manifest_reads.append(1)
        return {note_path: sha256_text(NOTE_BODY)}

    store = _FakeStore()  # empty document_sources
    embedder = _FakeEmbeddingClient()

    report = hermes_bridge.ingest_obsidian_documents(
        [root],
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        embedding_client=embedder,
        state_reader=store.state_reader,
        insert_chunks=store.insert_chunks,
        finalize_source_revision=store.finalize_source_revision,
        manifest_loader=manifest_loader,
    )

    # document_sources empty -> note is written despite legacy manifest match
    assert store.finalize_calls == 1
    assert report.chunk_count >= 1
    assert len(report.discovered) == 1

    # legacy manifest read EXACTLY once, report-only
    assert sum(manifest_reads) == 1
    assert report.legacy_manifest_entries == 1


def test_bridge_skips_via_document_sources_state(tmp_path):
    from hermes_memory import hermes_bridge

    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    corpus_id = make_corpus_id("obsidian", str(root.resolve()))
    source_id = make_source_id(corpus_id, "notes/daily.md")
    content_hash = sha256_text(NOTE_BODY)

    store = _FakeStore()
    # document_sources already has this active revision -> must skip
    store.sources[source_id] = {
        "source_id": source_id,
        "revision": content_hash,
        "content_hash": content_hash,
        "is_active": True,
    }
    embedder = _FakeEmbeddingClient()

    report = hermes_bridge.ingest_obsidian_documents(
        [root],
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        embedding_client=embedder,
        state_reader=store.state_reader,
        insert_chunks=store.insert_chunks,
        finalize_source_revision=store.finalize_source_revision,
        manifest_loader=lambda: {},
    )

    assert embedder.embed_calls == []
    assert store.insert_calls == 0
    assert store.finalize_calls == 0
    assert len(report.skipped) == 1
    assert report.legacy_manifest_entries == 0
