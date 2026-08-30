"""Provenance-honest Memory Bank promotion (Task 16).

Promotion is opt-in and truthful: the corpus is written to BigQuery
independently of any Memory Bank call, promoted payloads carry provenance
context (source kind/path/heading) plus an instruction to keep only durable
preferences, promotion is restricted to Obsidian unless a separate flag opts
code/repository sources in, and a Memory Bank failure never rolls back the
already-valid corpus — it is surfaced as promotion-incomplete instead.

Every cloud dependency is faked; no real client is constructed and no network,
Vertex, or BigQuery call is made.
"""

from __future__ import annotations

from pathlib import Path

from hermes_memory import ingestion
from hermes_memory.config import HermesMemoryConfig
from hermes_memory.documents import make_corpus_id, make_source_id, sha256_text

CFG = HermesMemoryConfig(project="test-project", bq_dataset="test_dataset")
USER_ID = "tojo"
AGENT_NAME = "hermes"
MB_NAME = "projects/p/locations/l/reasoningEngines/1"

NOTE_BODY = """# Daily Note

I prefer to keep my morning routine deliberate and unhurried, a durable
principle that guides how I plan every working day.

## Section Two

A second section with more prose so the note produces at least one packed chunk
carrying real derived text ready for embedding downstream.
"""


def _write_note(root: Path, relative: str, body: str = NOTE_BODY) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


class _FakeEmbeddingResult:
    def __init__(self, values):
        self.values = values


class _FakeEmbeddingClient:
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


class _FakeStore:
    """In-memory document_sources authority wiring plan + apply together."""

    def __init__(self):
        self.sources: dict[str, dict] = {}
        self.insert_calls = 0
        self.finalize_calls = 0
        self.finalized: list[tuple] = []

    def state_reader(self, source_id, *, user_id, agent_name):
        return self.sources.get(source_id)

    def insert_chunks(
        self, chunks, *, user_id, agent_name, embedding_model, embedding_dimensions, cfg
    ):
        self.insert_calls += 1
        return len(chunks)

    def finalize_source_revision(
        self, source_id, active_chunk_ids, *, source, user_id, agent_name, cfg
    ):
        self.finalize_calls += 1
        self.finalized.append((source_id, source))
        self.sources[source_id] = {
            "source_id": source_id,
            "revision": source["revision"],
            "content_hash": source["content_hash"],
            "is_active": True,
        }


class _RecordingBank:
    """Fake memory_bank_client capturing every promotion call's payload."""

    def __init__(self, fail: bool = False):
        self.calls: list[tuple] = []
        self.fail = fail

    def __call__(self, name, texts, scope, *, cfg):
        self.calls.append((name, list(texts), scope))
        if self.fail:
            raise RuntimeError("memory bank unavailable")
        # A truthful client returns opaque operation metadata, NOT a per-chunk
        # fact list. Nothing here should become a fabricated one-to-one mapping.
        return {"operation": "generate", "mock": True}


def _plan_obsidian(root, store):
    return ingestion.plan_obsidian_ingestion(
        [root],
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        state_reader=store.state_reader,
    )


def _apply(plan, store, embedder, **kw):
    return ingestion.apply_ingestion_plan(
        plan,
        cfg=CFG,
        user_id=USER_ID,
        agent_name=AGENT_NAME,
        embedding_client=embedder,
        insert_chunks=store.insert_chunks,
        finalize_source_revision=store.finalize_source_revision,
        **kw,
    )


def _synthetic_code_plan():
    """Build an IngestionPlan holding one repository/code source directly.

    Repo ingestion is not integrated in this worktree, so a code source is
    constructed by hand to exercise source-kind eligibility hermetically.
    """
    corpus_id = make_corpus_id("git", "/repo")
    relative_path = "src/module.py"
    source_id = make_source_id(corpus_id, relative_path)
    text = "def configure():\n    return {'retries': 3}\n"
    chunk = ingestion._PlannedChunk(
        chunk_id="chunk-code-0",
        source_id=source_id,
        corpus_id=corpus_id,
        ordinal=0,
        content_hash=sha256_text(text),
        heading_path=("configure",),
        symbol="configure",
        start_line=1,
        end_line=2,
        citation=f"{relative_path}#L1-L2",
        token_estimate=8,
        text=text,
        contextual_text=ingestion._contextual_text(("configure",), text),
    )
    planned = ingestion.PlannedSource(
        source_id=source_id,
        corpus_id=corpus_id,
        source_kind="git",
        content_kind="code",
        relative_path=relative_path,
        source_uri="file:///repo/src/module.py",
        revision=chunk.content_hash,
        content_hash=chunk.content_hash,
        status="discovered",
        chunk_count=1,
        token_count=8,
        chunks=(chunk,),
        units=(),
    )
    plan = ingestion.IngestionPlan(
        discovered=(planned,),
        skipped=(),
        rejected=(),
        chunk_count=1,
        request_count=1,
        token_count=8,
        cost_estimate=0.0,
    )
    return plan, source_id


# --- Requirement 1: no promotion call unless explicitly requested -----------


def test_no_memory_bank_call_when_promotion_not_requested(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    store = _FakeStore()
    embedder = _FakeEmbeddingClient()
    bank = _RecordingBank()

    plan = _plan_obsidian(root, store)
    report = _apply(
        plan,
        store,
        embedder,
        memory_bank_client=bank,
        memory_bank_name=MB_NAME,
        # promote_to_memory_bank defaults False
    )

    assert bank.calls == [], "no promotion call may occur unless requested"
    assert report.promotion_status == "not_requested"
    # corpus is written regardless of promotion
    assert store.finalize_calls == 1
    assert report.chunk_count >= 1
    # per-source outcome also reports no promotion requested
    assert report.discovered[0].promotion_status == "not_requested"


# --- Requirement 2: promoted payload carries provenance context -------------


def test_promotion_payload_carries_provenance_and_durable_instruction(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    store = _FakeStore()
    embedder = _FakeEmbeddingClient()
    bank = _RecordingBank()

    plan = _plan_obsidian(root, store)
    report = _apply(
        plan,
        store,
        embedder,
        memory_bank_client=bank,
        memory_bank_name=MB_NAME,
        promote_to_memory_bank=True,
    )

    assert len(bank.calls) == 1
    name, texts, scope = bank.calls[0]
    assert name == MB_NAME
    assert scope == {"user_id": USER_ID, "agent_name": AGENT_NAME}
    assert texts, "promotion must carry at least one contextual payload"

    for payload in texts:
        # provenance context: source type, path, and heading
        assert "obsidian" in payload
        assert "notes/daily.md" in payload
        assert "Daily Note" in payload
        # instruction to keep only durable preferences, not transient wording
        low = payload.lower()
        assert "durable" in low
        assert "preference" in low or "principle" in low
        assert "transient" in low

    assert report.promotion_status == "complete"


def test_promotion_does_not_fabricate_one_to_one_fact_mapping(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    store = _FakeStore()
    embedder = _FakeEmbeddingClient()
    bank = _RecordingBank()

    plan = _plan_obsidian(root, store)
    report = _apply(
        plan,
        store,
        embedder,
        memory_bank_client=bank,
        memory_bank_name=MB_NAME,
        promote_to_memory_bank=True,
    )

    # The report records a status only — never a fabricated list of returned
    # facts, and never the raw note body echoed back as confirmed memory.
    assert report.promotion_status == "complete"
    text = repr(report)
    assert "fact" not in text.lower()
    assert "morning routine" not in text
    # no per-source field claims Memory Bank returned N facts for N chunks
    outcome = report.discovered[0]
    for value in vars(outcome).values():
        assert not (isinstance(value, (list, tuple)) and value and "fact" in repr(value).lower())


# --- Requirement 3: Obsidian default; code/repo needs a separate flag -------


def test_code_source_not_promoted_by_default(tmp_path):
    store = _FakeStore()
    embedder = _FakeEmbeddingClient()
    bank = _RecordingBank()
    plan, source_id = _synthetic_code_plan()

    report = _apply(
        plan,
        store,
        embedder,
        memory_bank_client=bank,
        memory_bank_name=MB_NAME,
        promote_to_memory_bank=True,
        # promote_code_sources defaults False -> code stays a retrieval chunk
    )

    assert bank.calls == [], "code/repo sources must not promote without the flag"
    # corpus still written for the code source
    assert store.finalize_calls == 1
    outcome = report.discovered[0]
    assert outcome.source_id == source_id
    assert outcome.status == "written"
    assert outcome.promotion_status == "not_eligible"


def test_code_source_promoted_with_explicit_flag(tmp_path):
    store = _FakeStore()
    embedder = _FakeEmbeddingClient()
    bank = _RecordingBank()
    plan, source_id = _synthetic_code_plan()

    report = _apply(
        plan,
        store,
        embedder,
        memory_bank_client=bank,
        memory_bank_name=MB_NAME,
        promote_to_memory_bank=True,
        promote_code_sources=True,
    )

    assert len(bank.calls) == 1
    _name, texts, _scope = bank.calls[0]
    for payload in texts:
        assert "git" in payload
        assert "src/module.py" in payload
        low = payload.lower()
        assert "durable" in low
        assert "transient" in low
    assert report.discovered[0].promotion_status == "complete"
    assert report.promotion_status == "complete"


# --- Requirement 4: record REQUESTED promotion in source metadata -----------


def test_requested_promotion_recorded_in_source_metadata(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    store = _FakeStore()
    embedder = _FakeEmbeddingClient()
    bank = _RecordingBank()

    plan = _plan_obsidian(root, store)
    _apply(
        plan,
        store,
        embedder,
        memory_bank_client=bank,
        memory_bank_name=MB_NAME,
        promote_to_memory_bank=True,
    )

    assert store.finalized, "source must be finalized"
    _sid, source = store.finalized[0]
    meta = source["metadata"]
    # truthful: records that promotion was REQUESTED, not that facts were made
    assert meta.get("memory_bank_promotion") == "requested"
    blob = repr(source).lower()
    assert "fact" not in blob


def test_metadata_omits_promotion_when_not_requested(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    store = _FakeStore()
    embedder = _FakeEmbeddingClient()

    plan = _plan_obsidian(root, store)
    _apply(plan, store, embedder)

    _sid, source = store.finalized[0]
    assert "memory_bank_promotion" not in source["metadata"]


def test_code_metadata_not_marked_requested_without_flag(tmp_path):
    store = _FakeStore()
    embedder = _FakeEmbeddingClient()
    bank = _RecordingBank()
    plan, _source_id = _synthetic_code_plan()

    _apply(
        plan,
        store,
        embedder,
        memory_bank_client=bank,
        memory_bank_name=MB_NAME,
        promote_to_memory_bank=True,
    )

    _sid, source = store.finalized[0]
    # code source is not eligible -> promotion was not requested for it
    assert "memory_bank_promotion" not in source["metadata"]


# --- Requirement 5: partial failure keeps corpus valid, marks incomplete ----


def test_promotion_failure_keeps_corpus_valid_and_marks_source_incomplete(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    store = _FakeStore()
    embedder = _FakeEmbeddingClient()
    bank = _RecordingBank(fail=True)

    plan = _plan_obsidian(root, store)
    report = _apply(
        plan,
        store,
        embedder,
        memory_bank_client=bank,
        memory_bank_name=MB_NAME,
        promote_to_memory_bank=True,
    )

    # promotion failure must NOT raise; the corpus is already inserted+finalized
    assert store.insert_calls == 1
    assert store.finalize_calls == 1
    # the finalized source remains active/valid in the document corpus
    corpus_source_id = report.discovered[0].source_id
    assert store.sources[corpus_source_id]["is_active"] is True
    # promotion incompleteness is surfaced per source AND at the report level
    assert report.discovered[0].promotion_status == "incomplete"
    assert report.promotion_status == "incomplete"
    assert report.chunk_count >= 1


def test_promotion_incomplete_when_requested_but_unconfigured(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    store = _FakeStore()
    embedder = _FakeEmbeddingClient()

    plan = _plan_obsidian(root, store)
    report = _apply(
        plan,
        store,
        embedder,
        promote_to_memory_bank=True,
        # no memory_bank_client / memory_bank_name -> cannot promote
    )

    # corpus is valid; promotion cannot proceed -> incomplete, not raised
    assert store.finalize_calls == 1
    assert report.promotion_status == "incomplete"
    assert report.discovered[0].promotion_status == "incomplete"
