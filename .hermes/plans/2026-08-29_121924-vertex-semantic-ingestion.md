# Vertex-Native Semantic Ingestion Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add citation-bearing, incremental Obsidian and Git-repository ingestion to the GCP memory plugin by combining local structure-aware parsing with Vertex AI embeddings, BigQuery document-chunk storage/retrieval, and opt-in Memory Bank fact extraction—without adding another database service.

**Architecture:** Keep Agent Platform Memory Bank as the durable personalized-memory channel and use the existing BigQuery dataset as a separate citation-bearing document-retrieval channel. Parse Markdown and source files locally because Google's managed layout parser does not support Markdown/source-code formats, use `gemini-embedding-001` for semantic boundary scoring and retrieval embeddings, persist deterministic sources/chunks in BigQuery, and merge bounded document hits into the plugin's existing `prefetch()` context. RAG Engine, Vertex AI Search, Document AI, Cloud SQL, Firestore, and a separate vector database are explicitly deferred.

**Tech Stack:** Python 3.11+, Click, existing `vertexai.Client`/Google Gen AI SDK, Vertex AI `gemini-embedding-001`, Agent Platform Memory Bank, BigQuery `ARRAY<FLOAT64>` + `VECTOR_SEARCH`, `markdown-it-py`, optional and spike-gated Tree-sitter language pack, pytest, Git/GitHub metadata, Hermes memory-provider plugin API, Linear FUL team.

---

## 1. Executive decision

### Build now

1. Add **two tables to the existing** `gen-lang-client-0810135629.hermes_memory` dataset:
   - `document_sources`: canonical source identity, revision/hash, lifecycle state, and provenance.
   - `document_chunks`: chunk text, 768-dimensional Vertex embedding, heading/symbol/line metadata, and active state.
2. Add a local structure-aware splitter:
   - Markdown: frontmatter + heading hierarchy + paragraph/list/fence boundaries.
   - Source code: symbol boundaries when the tested parser supports the language; deterministic line-window fallback otherwise.
3. Use Vertex AI embeddings twice, but cache the first result:
   - `SEMANTIC_SIMILARITY` for choosing a boundary only when a structural section exceeds the target size.
   - `RETRIEVAL_DOCUMENT` for stored chunk vectors.
4. Query stored vectors with `RETRIEVAL_QUERY`; permit `CODE_RETRIEVAL_QUERY` only for an explicit code-search mode.
5. Add citation-bearing document hits to `HermesBridge.retrieve_context()` and the `gcp_memory_bank` provider's `prefetch()` output.
6. Keep Memory Bank promotion explicit: `--promote-to-memory-bank` runs approved chunks through the existing `generate_from_contents()` path, while the default only creates the citation-bearing BigQuery corpus.

### Do not build now

- No RAG Engine corpus.
- No Vertex AI Search data store.
- No Document AI processor.
- No Cloud SQL/pgvector, Firestore, Chroma, Qdrant, or Vertex AI Vector Search deployment.
- No BigQuery vector index initially.
- No watcher/daemon, GCS staging bucket, Pub/Sub pipeline, Cloud Run service, or scheduled full-vault ingestion.
- No bulk ingestion of private chats, client corpora, raw captures, finance/tax/medical/legal/identity/household material, credentials, or secrets.
- No automatic Memory Bank promotion based only on path or tags.

This is the smallest useful architecture: one existing managed memory service, one existing analytical database, one local ingestion CLI, and one automatic retrieval integration.

## 2. Research findings and source check

### 2.1 Google has managed chunking, but it is not the right v1 splitter

RAG Engine can ingest Markdown and its configurable transformation is fixed token-size chunking; Google's documented defaults are 1,024 tokens with 256-token overlap.[1][2]

Its billing documentation separately identifies file chunking as fixed-size transformation and embedding generation as a distinct stage.[21]

That is useful generic RAG ingestion, but it does not preserve Obsidian heading semantics or code symbols by itself.

RAG Engine also requires creating a corpus and choosing a vector database, which would add a second retrieval store beside BigQuery and Memory Bank.[16]

Document AI's Gemini layout parser creates context-aware chunks and preserves elements such as headings and tables, but its supported input table lists HTML, PDF, DOCX, PPTX, XLSX, and XLSM—not Markdown or source-code files.[3]

Its RAG Engine integration also enables the Document AI API and applies Document AI quotas and pricing.[5][12]

Document AI's beta schema exposes `semanticChunkingGroupSize` and `breakpointPercentileThreshold`, but both fields are marked “not yet used”; they are not an operational semantic-splitter API.[22]

Agent Search offers layout-aware chunks of 100–500 tokens inside its own data-store lifecycle. Markdown is accepted for ingestion, but it is absent from the Layout Parser format matrix, so the documentation does not promise Markdown-heading-aware parsing.[4][23]

**Conclusion:** do not describe Vertex embeddings as a Google-hosted splitter. The splitter is deterministic Python in this repository; the semantic scoring and final representations are Vertex-native.

### 2.2 Embedding model and task-type decision

Google documents `gemini-embedding-001` as a unified model for English, multilingual, and code tasks, with a 2,048-token maximum sequence length and output dimensionality up to 3,072.[6][14][15]

It supports lower Matryoshka dimensions; this plan chooses **768 dimensions** to keep BigQuery storage modest while staying within Google's recommended dimensions.

Google's current pages describe interface-dependent batch limits: the generic online guide permits multiple inputs, while the legacy `:predict` shape allows only one `gemini-embedding-001` input per request.[14][15] The adapter must therefore test and bind to one SDK interface, preserve order, and remain correct at one input per request rather than assuming a larger batch.

`gemini-embedding-2` is GA with an 8,192-token input window, but Google documents it through the multimodal embedding API and still recommends the text embedding API for text-only semantic search, long-form analysis, and retrieval.[18][19][20] It also lacks the text API's documented `autoTruncate=false` control and enum task-type compatibility used by this plan. Therefore `gemini-embedding-001` remains the initial model; `gemini-embedding-2` becomes an explicit evaluation/migration candidate, not a silent upgrade.

Google explicitly distinguishes:

- `SEMANTIC_SIMILARITY` for comparing text similarity—not retrieval.
- `RETRIEVAL_DOCUMENT` for indexed documents.
- `RETRIEVAL_QUERY` for ordinary search queries.
- `CODE_RETRIEVAL_QUERY` for natural-language queries intended to retrieve code; code blocks remain embedded with `RETRIEVAL_DOCUMENT`.[7]

**Conclusion:** use the right task type at each stage, set `auto_truncate=False`, reject any response marked `truncated=True`, and never mix vectors from different models, dimensions, or task families in one search.

### 2.3 Memory Bank and document retrieval are different channels

Google describes Memory Bank as long-term personalized memory generated from user-agent conversations, scoped into isolated collections of self-contained facts.[8] Its direct-content generation path is appropriate for extracting durable principles from approved note chunks, but Memory Bank is not the canonical store for source files, line ranges, Git revisions, or citation lifecycle.[9]

**Conclusion:**

- Memory Bank remains authoritative for conversational/personal facts.
- BigQuery document chunks remain authoritative for excerpts and citations.
- The plugin may retrieve both, but it must label them separately.
- A Memory Bank fact must never be presented as though it has an exact source citation unless that provenance is independently stored and verified.

### 2.4 BigQuery is sufficient for the pilot

BigQuery supports brute-force exact vector search without a vector index; indexes are an optional scale optimization that trade exactness for approximate search and carry BigQuery compute charges.[10] For a personal vault and initial repository pilot, exact search is simpler and gives a reliable evaluation baseline.

BigQuery can also generate table embeddings through `AI.GENERATE_EMBEDDING`, but application-managed incremental embedding is the initial choice because it preserves explicit task-type pairing, source lifecycle ordering, and per-chunk failure handling without a remote-model setup step.[24]

**Phase-2 vector-index gate:** consider `CREATE VECTOR INDEX` only after the active chunk table is large enough that measured p95 retrieval exceeds the agreed budget (initial proposed gate: 1.5 seconds over 20 repeated warm queries) or bytes scanned become materially wasteful. Do not use a guessed row-count threshold as the sole trigger.

### 2.5 Pricing and promotional-credit handling

The current Google Cloud pricing page lists Gemini Embedding online input at `$0.00015 per 1,000 count` and batch input at `$0.00012 per 1,000 count`; output is not charged.[11] BigQuery vector search is billed as BigQuery compute/bytes scanned, not as an embedding-model call.[10] Promotional-credit applicability is account- and SKU-specific; Google Billing reports expose credits and promotions, so the implementation must report usage/cost estimates but must not claim the user's `$1,000` credit covers a SKU until the Billing account shows that credit applied.[13]

**Cost-control behavior:** dry-run reports file count, structural-unit count, estimated embedding input, expected online request count, and the pricing URL; the real run records actual `billable_character_count`, token count, retry count, and model name returned by the API.

## 3. Current repository context

The implementation starts from these verified behaviors:

- `src/hermes_memory/cli.py:125-185` implements `ingest-obsidian` by scanning whole notes, grouping whole notes under `--batch-chars`, and calling Memory Bank generation. It does not split long notes internally.
- `src/hermes_memory/hermes_bridge.py:60-105` discovers notes and batches them; `:164-218` merges Memory Bank, recent BigQuery memory rows, and local SQLite facts.
- `src/hermes_memory/memory_bank.py:153-166` already exposes `generate_from_contents()` and `:169-199` exposes semantic Memory Bank retrieval.
- `src/hermes_memory/bigquery_store.py:91-239` owns dataset/table setup, memory writes, a provisional vector-search path, SQL recall, and stats.
- `plugins/gcp_memory_bank/__init__.py:177-209` converts merged facts into the automatic prefetch block.
- `src/hermes_memory/config.py:9-45` currently uses `text-embedding-005` as the Memory Bank configuration model. Do **not** silently migrate the existing Memory Bank model; add a separate `document_embedding_model` setting.
- BigQuery schema is represented in three places: `bigquery/schema.sql`, Python DDL constants, and `terraform/main.tf` plus `terraform/schemas/*.json`. New tables must update all three representations and include a parity test.
- `tests/test_imports.py` is the only current test module. Its bridge “mock” test can resolve the hard-coded real Agent Engine ID and touch live/local state, so deterministic client injection is the first test-harness correction before feature tests.

## 4. Data and interface contracts

### 4.1 Python models

Create `src/hermes_memory/documents.py` with immutable records similar to:

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

SourceKind = Literal["obsidian", "git"]
ContentKind = Literal["markdown", "code", "text"]

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
```

Identity rules use one unambiguous canonical framing for every multi-part hash. UTF-8 encode each
part, prefix each encoded part with its fixed-width 8-byte unsigned big-endian byte length, concatenate
the resulting frames without delimiters, then hash the bytes:

```text
frame(parts) = concat(uint64_be(len(utf8(part))) + utf8(part) for part in parts)
corpus_id = sha256(frame([source_kind, canonical_root_or_remote]))[:24]
source_id = sha256(frame([corpus_id, normalized_relative_path]))[:32]
content_hash = sha256(normalized_source_or_chunk_text)
chunk_id = sha256(frame([source_id, heading_or_symbol, str(occurrence), chunk_content_hash]))[:40]
```

`chunk_id` is content-addressed. A changed chunk receives a new ID; an unchanged chunk keeps its ID. After a successful source replacement, older IDs for that `source_id` are marked inactive in the same lifecycle step.

### 4.2 BigQuery schema

Append idempotent DDL to `bigquery/schema.sql` and constants in `bigquery_store.py`:

```sql
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.document_sources` (
  source_id STRING NOT NULL,
  corpus_id STRING NOT NULL,
  user_id STRING NOT NULL,
  agent_name STRING NOT NULL,
  source_kind STRING NOT NULL,
  content_kind STRING NOT NULL,
  relative_path STRING NOT NULL,
  source_uri STRING NOT NULL,
  revision STRING NOT NULL,
  content_hash STRING NOT NULL,
  metadata JSON,
  is_active BOOL NOT NULL,
  first_seen_at TIMESTAMP NOT NULL,
  last_seen_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL
)
CLUSTER BY user_id, agent_name, corpus_id, source_kind;

CREATE TABLE IF NOT EXISTS `{project}.{dataset}.document_chunks` (
  chunk_id STRING NOT NULL,
  source_id STRING NOT NULL,
  corpus_id STRING NOT NULL,
  user_id STRING NOT NULL,
  agent_name STRING NOT NULL,
  ordinal INT64 NOT NULL,
  content STRING NOT NULL,
  contextual_content STRING NOT NULL,
  content_hash STRING NOT NULL,
  heading_path ARRAY<STRING>,
  symbol STRING,
  start_line INT64,
  end_line INT64,
  citation STRING NOT NULL,
  embedding ARRAY<FLOAT64> NOT NULL,
  embedding_model STRING NOT NULL,
  embedding_dimensions INT64 NOT NULL,
  metadata JSON,
  is_active BOOL NOT NULL,
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL
)
CLUSTER BY user_id, agent_name, corpus_id, source_id;
```

Do not add TTL to source chunks. Their lifecycle follows the canonical file; deleted files are deactivated. Existing `memories` TTL remains unchanged.

### 4.3 Chunking defaults

Use configuration, not magic literals:

```text
document_embedding_model = gemini-embedding-001
document_embedding_dimensions = 768
chunk_target_tokens = 600
chunk_min_tokens = 250
chunk_max_tokens = 900
chunk_overlap_tokens = 80
embedding_concurrency = 4
document_top_k = 4
document_context_char_limit = 8_000
```

Reasons:

- 900 stays comfortably below the 2,048-token model limit after contextual prefixes.
- 600 is a retrieval-oriented target rather than a hard cut.
- 80 overlap carries short transitions without replicating whole sections.
- Semantic scoring is only used for oversized sections, limiting request count.
- Defaults are pilots, not universal truth; the evaluation gate decides whether to adjust them.

### 4.4 CLI contracts

Replace the old whole-note batching behavior while retaining command compatibility:

```bash
hermes-memory ingest-obsidian \
  --user tojo --agent hermes \
  --vault ~/Vaults/Hermes_Agent \
  --corpus hermes-agent-vault \
  --include 'Operations/**' --include 'Models/**' --include 'Kanban/**' \
  --strategy vertex-semantic \
  --dry-run

hermes-memory ingest-repo \
  --user tojo --agent hermes \
  --repo /path/to/checkout \
  --ref HEAD \
  --strategy vertex-semantic \
  --dry-run

hermes-memory search-docs \
  --user tojo --agent hermes \
  --query 'how should agents decide what becomes memory?' \
  --corpus hermes-agent-vault \
  --top-k 5
```

Real writes require `--apply`; Memory Bank extraction additionally requires `--promote-to-memory-bank`. `--dry-run` and `--apply` are mutually exclusive. Keep `--limit` for bounded pilots.

### 4.5 Retrieval context format

Return distinct fields from `HermesBridge.retrieve_context()`:

```python
{
    "memory_bank_hits": [...],
    "document_hits": [
        {
            "text": "...",
            "distance": 0.18,
            "citation": "github.com/.../blob/<sha>/path/file.go#L40-L88",
            "source_id": "...",
            "chunk_id": "...",
        }
    ],
    "local_hits": [...],
    "prompt_context": "...",
}
```

The plugin renders:

```markdown
## GCP Memory Bank
- <durable personalized fact>

## GCP Source Context
- <excerpt>
  Source: <stable citation>
```

Never merge a source excerpt into the fact list or strip its citation.

## 5. Source policy and safety gates

### Allowed initial pilot

- Operations documentation.
- Model documentation.
- Kanban/project documentation.
- Approved personal operating-principle notes from the named pilot vault.
- One explicitly selected personal Git repository.

### Denied by default

- `.git`, `.obsidian`, `.trash`, `node_modules`, `vendor`, `.venv`, build/dist/cache directories, binaries, generated files, lock files, and files ignored by Git.
- `.env*`, private keys, credential/config exports, token dumps, auth/session files, service-account JSON, secrets, and files matching secret-detector rules.
- Personal finance, tax, medical, household, identity, and legal files.
- Client/private corpora unless a separate local-only workflow and explicit approval exist.
- Raw chat/transcript bulk ingestion.

A denied file must be counted and reported by reason without printing sensitive content. Detection failures are fail-closed for the file, not fail-open.

## 6. Implementation plan

Each production-code task follows RED → verify failure → GREEN → full regression → commit. Do not create a pile of tests first; complete one vertical behavior slice at a time.

### Task 1: Establish evaluation fixtures and baseline contracts

**Objective:** Capture the expected chunk/citation behavior before replacing the existing whole-note path.

**Files:**
- Create: `tests/fixtures/obsidian/Operations/agent-memory.md`
- Create: `tests/fixtures/repo/main.go`
- Create: `tests/fixtures/repo/README.md`
- Create: `tests/test_documents.py`
- Create: `tests/test_evaluation.py`
- Modify: `tests/test_imports.py`

**Steps:**
1. Replace `test_bridge_mock` with explicitly injected fake Memory Bank, BigQuery, and local-memory dependencies; assert that the test constructs no live Vertex client and reads no real local memory database.
2. Run `pytest tests/test_imports.py -v`; confirm the new isolation assertion fails against the current hard-coded configuration path.
3. Add the minimum dependency injection seam needed to make the existing bridge smoke test deterministic, then verify it passes before adding feature behavior.
4. Write fixtures containing YAML tags, nested headings, lists, a fenced code block, repeated heading names, one oversized section, Go functions/types, and an intentionally ignored `.env.example` fixture containing fake markers only.
5. Write a failing test for deterministic source/chunk IDs and stable line ranges.
6. Run `pytest tests/test_documents.py -v`; expect import/behavior failure because `documents.py` does not exist.
7. Add only the immutable data records and hash helpers needed by that test.
8. Re-run the focused tests, then `pytest -q`; expect all tests green with no credentials or network access.
9. Commit: `test: isolate bridge and define ingestion contracts`.

### Task 2: Add document-specific configuration without migrating Memory Bank

**Objective:** Separate document embedding/chunk settings from the existing Memory Bank model.

**Files:**
- Modify: `src/hermes_memory/config.py:9-45`
- Modify: `tests/test_documents.py`

**Steps:**
1. Write a failing test proving `embedding_model` remains `text-embedding-005` while `document_embedding_model` defaults to `gemini-embedding-001` with 768 dimensions.
2. Run the focused test and confirm the missing fields cause RED.
3. Add validated config fields and environment overrides prefixed `DOCUMENT_`.
4. Reject invalid relationships (`min > target`, `target > max`, overlap >= min, dimension <= 0).
5. Re-run the focused test and full suite.
6. Commit: `feat: add document ingestion configuration`.

### Task 3: Parse Markdown into line-addressable structural units

**Objective:** Preserve Obsidian structure before semantic splitting.

**Files:**
- Create: `src/hermes_memory/chunking.py`
- Modify: `pyproject.toml`
- Create: `tests/test_markdown_chunking.py`

**Steps:**
1. Add one failing test for frontmatter extraction, heading hierarchy, fenced-code atomicity, list grouping, and source line ranges.
2. Run it and confirm failure because `parse_markdown_units()` is missing.
3. Add `markdown-it-py` and use token line maps; use `yaml.safe_load` only if PyYAML is already transitively available, otherwise add explicit `PyYAML`.
4. Implement `parse_markdown_units(text) -> list[AtomicUnit]` without expanding wikilinks or mutating source text.
5. Re-run focused and full tests.
6. Commit: `feat: parse markdown into structural units`.

### Task 4: Implement deterministic packing and overlap

**Objective:** Produce useful chunks without any cloud call.

**Files:**
- Modify: `src/hermes_memory/chunking.py`
- Modify: `tests/test_markdown_chunking.py`

**Steps:**
1. Write a failing test proving sections under `chunk_max_tokens` remain whole, oversized units split, no chunk exceeds max, and overlap never crosses an unrelated top-level heading.
2. Implement a conservative token estimate (`ceil(len(text) / 4)`) used only for dry-run/packing; final API statistics remain authoritative.
3. Implement greedy structural packing around the target and a hard fallback for one oversized atomic unit.
4. Verify deterministic output over two runs.
5. Run full tests.
6. Commit: `feat: add bounded structural chunk packing`.

### Task 5: Add the Vertex embedding gateway

**Objective:** Centralize model/task/dimension/truncation behavior and make it testable offline.

**Files:**
- Create: `src/hermes_memory/embeddings.py`
- Create: `tests/test_embeddings.py`
- Modify: `src/hermes_memory/config.py`

**Steps:**
1. Write a fake-client test expecting `gemini-embedding-001`, `output_dimensionality=768`, the requested task type, and truncation rejection.
2. Add `VertexEmbeddingClient` with injected client, bounded `ThreadPoolExecutor`, exponential retry only for documented transient status codes, and no retry for invalid input/auth.
3. Return `EmbeddingResult(values, token_count, billable_character_count, truncated, model)`.
4. Add an in-process cache keyed by `(model, dimensions, task_type, sha256(text))`; do not introduce Redis or another cache service.
5. Verify one-input-per-request behavior in tests.
6. Run focused and full tests.
7. Commit: `feat: add vertex document embeddings`.

### Task 6: Select semantic boundaries only where structure is insufficient

**Objective:** Use Vertex similarity to refine oversized Markdown sections without making every section nondeterministic.

**Files:**
- Modify: `src/hermes_memory/chunking.py`
- Create: `tests/test_semantic_chunking.py`

**Algorithm:**
1. Keep heading boundaries mandatory.
2. For an oversized section, embed atomic units with `SEMANTIC_SIMILARITY`.
3. In each `[min_tokens, max_tokens]` candidate range, cut at the adjacent-unit boundary with the lowest cosine similarity; break ties by proximity to target, then earlier line number.
4. Add overlap from trailing complete units only.
5. Fall back to deterministic packing if Vertex is unavailable **only during dry-run**. Real `--apply` fails before BigQuery mutation unless the user explicitly selects `--strategy structural`.

**Steps:**
1. Write a failing test with fixed fake vectors whose weakest boundary is known.
2. Implement cosine calculation and tie-breaking as pure functions.
3. Implement the cloud-backed refinement wrapper.
4. Test zero vectors, one unit, API failure, and max-token enforcement.
5. Run full tests.
6. Commit: `feat: refine chunk boundaries with vertex similarity`.

### Task 7: Add BigQuery document tables idempotently

**Objective:** Extend the existing dataset instead of adding a database service.

**Files:**
- Modify: `src/hermes_memory/bigquery_store.py:1-120`
- Modify: `bigquery/schema.sql`
- Modify: `terraform/main.tf`
- Create: `terraform/schemas/document_sources.json`
- Create: `terraform/schemas/document_chunks.json`
- Modify: `tests/test_imports.py`
- Create: `tests/test_document_store.py`

**Steps:**
1. Write failing tests that inspect both new DDL statements, clustering fields, Terraform resources, and JSON schema parity.
2. Add `DDL_DOCUMENT_SOURCES` and `DDL_DOCUMENT_CHUNKS` and include them in `ensure_tables()`.
3. Add the two Terraform table resources and schema JSON files with fields/modes matching the Python and SQL representations exactly.
4. Keep dataset location `US`; do not create indexes.
5. Run `terraform fmt -check` and `terraform validate` without applying infrastructure.
6. Verify idempotent mock execution and full test suite.
7. Commit: `feat: add bigquery document corpus tables`.

### Task 8: Implement restart-safe source/chunk synchronization

**Objective:** Upsert a source revision and deactivate stale chunks only after new chunks are safely written.

**Files:**
- Modify: `src/hermes_memory/bigquery_store.py`
- Modify: `tests/test_document_store.py`

**Steps:**
1. Write a failing store-level test with a fake BigQuery client for unchanged-source skip.
2. Add parameterized `get_source_state()` and `upsert_source()`; do not interpolate user/path values into SQL.
3. Add `insert_chunks()` with deterministic insert IDs and strict dimension/model validation.
4. Add `finalize_source_revision(source_id, active_chunk_ids)` that deactivates unmatched old chunks only after all expected new IDs are present.
5. Add `deactivate_missing_sources(corpus_id, seen_source_ids)` behind `--prune`; never prune on a limited run.
6. Test interrupted run behavior: old active revision remains retrievable.
7. Run full suite.
8. Commit: `feat: synchronize document revisions safely`.

### Task 9: Add exact BigQuery vector retrieval

**Objective:** Return filtered, citation-bearing source excerpts without a vector index.

**Files:**
- Modify: `src/hermes_memory/bigquery_store.py`
- Modify: `tests/test_document_store.py`

**Steps:**
1. Write a failing SQL-shape test for `search_document_chunks()`.
2. Query only `is_active`, matching `user_id`, `agent_name`, `embedding_model`, and `embedding_dimensions`; optionally filter `corpus_id`, `source_kind`, or `content_kind` before distance ranking.
3. Use an embedding query-table CTE and `VECTOR_SEARCH(..., distance_type => 'COSINE', use_brute_force => TRUE)`.
4. Return distance, content, citation, path, heading/symbol, line range, and chunk/source IDs.
5. Test no-client mock behavior and empty results.
6. Run full tests.
7. Commit: `feat: search citation-bearing document chunks`.

### Task 10: Enforce source allowlists, ignores, and secret rejection

**Objective:** Make ingestion safe-by-default without building an enterprise DLP system.

**Files:**
- Create: `src/hermes_memory/source_discovery.py`
- Create: `tests/test_source_discovery.py`

**Steps:**
1. Write failing tests for repeatable include/exclude globs, Git ignored files, default excluded directories, symlink escape, binary detection, max file size, and secret-path rejection.
2. Implement `SourcePolicy` with allowlist-first semantics; no include pattern means no full-vault cloud run unless the user passes `--allow-all-approved`.
3. Use `git check-ignore` for repositories when Git is available; fall back to static exclusions with a warning.
4. Add lightweight secret-content patterns for private-key headers and obvious token assignments; report only path + rule name.
5. Run full tests.
6. Commit: `feat: add safe source discovery policy`.

### Task 11: Build the Obsidian ingestion vertical slice

**Objective:** Discover, chunk, embed, store, and incrementally skip approved Markdown notes.

**Files:**
- Create: `src/hermes_memory/ingestion.py`
- Modify: `src/hermes_memory/hermes_bridge.py:22-105`
- Create: `tests/test_ingestion.py`

**Steps:**
1. Write an end-to-end fake-client test for one note: first run writes chunks; second unchanged run makes no Vertex/BigQuery writes; changed run replaces only that source.
2. Add `IngestionPlan` and `IngestionReport` with discovered/skipped/rejected/chunk/request/token/cost fields.
3. Build `plan_obsidian_ingestion()` with no external calls.
4. Build `apply_ingestion_plan()` in this order: validate → semantic boundary vectors → final vectors → insert chunks → finalize source → optional Memory Bank promotion.
5. Replace the v1 local manifest as authority with BigQuery `document_sources`; retain a one-time read of `obsidian_ingest_manifest.json` only to report legacy state, not to skip document writes.
6. Run full tests.
7. Commit: `feat: add incremental obsidian document ingestion`.

### Task 12: Replace the Obsidian CLI safely

**Objective:** Expose a preview-first personal workflow while preserving the command name.

**Files:**
- Modify: `src/hermes_memory/cli.py:125-185`
- Create: `tests/test_cli_obsidian.py`

**Steps:**
1. Write Click-runner tests for missing allowlist, `--dry-run`, `--apply`, mutually exclusive flags, limit behavior, prune restriction, and promotion confirmation.
2. Add a strict dry-run assertion that no live Vertex/BigQuery/Memory Bank client is constructed, no network method is called, and no manifest/state file is written.
3. Replace `--batch-chars` with deprecated warning; map it nowhere rather than silently changing semantics.
4. Print a deterministic plan table and JSON with `--json`.
5. Require `--apply`; require a second explicit `--promote-to-memory-bank` for fact extraction.
6. Never print file bodies, credentials, embeddings, or rejected secret content.
7. Run focused and full tests.
8. Commit: `feat: expose semantic obsidian ingestion cli`.

### Task 13: Add Git repository discovery and stable citations

**Objective:** Treat a local checkout as canonical while producing commit-pinned GitHub citations.

**Files:**
- Modify: `src/hermes_memory/source_discovery.py`
- Modify: `src/hermes_memory/documents.py`
- Create: `tests/test_repo_discovery.py`

**Steps:**
1. Write failing tests for remote normalization (`git@github.com:owner/repo.git` and HTTPS), current commit SHA, dirty-tree rejection, detached HEAD, path quoting, and line-anchor URLs.
2. Require a clean worktree for `--apply` unless `--allow-dirty` is explicit; dirty citations use a local URI and revision marker rather than pretending GitHub contains the changes.
3. Build citations as `<remote>/blob/<commit>/<quoted-path>#Lx-Ly` only for recognized GitHub remotes.
4. Record language by extension, revision, branch/ref, remote URL, and repository-relative path.
5. Run full tests.
6. Commit: `feat: discover git sources with stable citations`.

### Task 14: Add symbol-aware code units with a tested fallback

**Objective:** Keep functions/types together without making a parser platform mandatory.

**Files:**
- Modify: `src/hermes_memory/chunking.py`
- Modify: `pyproject.toml`
- Create: `tests/test_code_chunking.py`

**Steps:**
1. First run a bounded implementation spike for `tree-sitter-language-pack` on Python and Go, checking current API, Linux wheels, symbol line ranges, and license.[17]
2. If the spike passes, add it as an optional `code` extra and import lazily; ingestion of unsupported languages still works via deterministic line windows.
3. If the spike fails, do not substitute an unreviewed parser package. Implement Python symbols with `ast` and generic line windows; mark Go symbol-awareness deferred in output.
4. Write failing behavior tests for class/function/type boundaries, comments attached to following symbols, one oversized symbol, parse errors, and fallback citations.
5. Use symbol boundaries first; semantic splitting may group small adjacent symbols but never cut a symbol unless it exceeds `chunk_max_tokens`.
6. Run full tests on the base install and, if accepted, the `[code]` extra.
7. Commit: `feat: add symbol-aware repository chunking`.

### Task 15: Add `ingest-repo` and `search-docs` CLI commands

**Objective:** Complete the operator-facing repository and retrieval workflow.

**Files:**
- Modify: `src/hermes_memory/cli.py`
- Create: `tests/test_cli_repo.py`
- Create: `tests/test_cli_search_docs.py`

**Steps:**
1. Write failing Click tests for repo dry-run/apply, dirty tree, explicit ref, include/exclude, language filter, `--query-type docs|code`, and citation output.
2. Implement `ingest-repo` through the same `IngestionPlan` path used by Obsidian.
3. Implement `search-docs`: embed query with `RETRIEVAL_QUERY` or explicit `CODE_RETRIEVAL_QUERY`, search BigQuery, print ranked excerpts and citations.
4. Add `--json` for evaluation tooling.
5. Run full tests.
6. Commit: `feat: add repository ingestion and document search cli`.

### Task 16: Keep Memory Bank promotion explicit and provenance-honest

**Objective:** Extract durable personal principles from approved chunks without claiming false citations.

**Files:**
- Modify: `src/hermes_memory/ingestion.py`
- Modify: `src/hermes_memory/memory_bank.py:153-166`
- Create: `tests/test_memory_promotion.py`

**Steps:**
1. Write a failing fake-client test proving no Memory Bank call occurs without promotion.
2. For promoted chunks, send contextual text that includes source type/path/heading and instructs generation to retain only durable user preferences/principles, not transient document wording.
3. Restrict promotion to Obsidian by default; repository promotion requires a separate explicit flag because code/docs should generally remain retrieval chunks.
4. Record requested promotion in source metadata, but do not assert a one-to-one fact mapping that Memory Bank does not return.
5. Test partial Memory Bank failure: BigQuery source corpus stays valid and report marks promotion incomplete.
6. Run full tests.
7. Commit: `feat: add opt-in memory bank promotion`.

### Task 17: Integrate document retrieval into Hermes prefetch

**Objective:** Make chunks useful to every fleet node through the existing provider.

**Files:**
- Modify: `src/hermes_memory/hermes_bridge.py:164-218`
- Modify: `plugins/gcp_memory_bank/__init__.py:177-209`
- Create: `tests/test_bridge_retrieval.py`
- Create: `tests/test_plugin_prefetch.py`

**Steps:**
1. Write a failing bridge test with one Memory Bank fact and one document hit; assert separate result fields and citation retention.
2. Embed the query once for document retrieval; if document search fails or times out, preserve Memory Bank recall.
3. Merge by channel, not by truncating all content to a fake `fact` field.
4. Enforce `document_top_k` and `document_context_char_limit` before returning to the plugin.
5. Write a failing plugin test for the two Markdown sections and stable citations.
6. Add plugin config fields `document_retrieval_enabled`, `document_top_k`, and `document_context_char_limit`; default enabled only when document tables are available.
7. Run full tests.
8. Commit: `feat: add citation-aware document prefetch`.

### Task 18: Build a small retrieval-quality gate

**Objective:** Decide with measurements whether defaults are useful before indexing more data or adding infrastructure.

**Files:**
- Create: `evaluation/queries.yaml`
- Create: `src/hermes_memory/evaluation.py`
- Create: `tests/test_evaluation.py`
- Modify: `src/hermes_memory/cli.py`

**Steps:**
1. Define 10-15 pilot queries with expected source path and optional heading/symbol; do not include sensitive content in the committed fixture.
2. Add `hermes-memory evaluate-docs --queries evaluation/queries.yaml --json out.json`.
3. Compute Recall@5, MRR, citation validity, p50/p95 latency, bytes processed if available, and zero-truncation rate.
4. Set initial acceptance gates: Recall@5 >= 0.80, citation validity = 1.00, truncation = 0, and p95 <= 1.5 seconds on the pilot. Mark these as pilot gates, not global SLAs.
5. Run unit tests with deterministic fake results.
6. Commit: `test: add document retrieval quality gate`.

### Task 19: Update setup, docs, and rollback

**Objective:** Make the personal deployment understandable and reversible.

**Files:**
- Modify: `README.md`
- Modify: `scripts/setup.sh`
- Modify: `plugins/gcp_memory_bank/plugin.yaml`
- Modify: `bigquery/views.sql` only if a simple active-source/chunk count view proves useful
- Create: `docs/semantic-ingestion.md`

**Steps:**
1. Document architecture/channel separation, source policy, CLI examples, cost reporting, promotion semantics, and the fact that credit eligibility must be checked in Billing.
2. Document migration from the old whole-note command; do not automatically re-promote legacy notes.
3. Document rollback: disable document retrieval in plugin config, leave existing Memory Bank behavior intact, and deactivate/remove document tables only with explicit approval.
4. Add troubleshooting for model quota/auth, BigQuery location, dirty repos, unsupported parser language, stale source revisions, and the documented online-interface batch-limit discrepancy.
5. Update plugin metadata/version only if the shipped provider behavior or dependency contract changed; do not create a second installation path.
6. Run `pytest -q`, package build, import checks, `terraform fmt -check`, and `terraform validate`.
7. Commit: `docs: add semantic ingestion operations guide`.

### Task 20: Perform bounded live pilots and fleet verification

**Objective:** Prove the feature through real cloud writes and real Hermes recall.

**Bounded-sensitive steps requiring explicit user approval:**

1. Apply BigQuery DDL to `gen-lang-client-0810135629:hermes_memory`.
2. Perform the first real Vertex embedding request.
3. Ingest the first 5 approved Obsidian notes.
4. Promote any note chunks to Memory Bank.
5. Restart/relaunch any production Hermes gateway or desktop runtime.

**Verification sequence:**

```bash
# 1. Offline/full tests
pytest -q
python -m build

# 2. Dry-run with allowlisted paths
hermes-memory ingest-obsidian ... --limit 5 --dry-run --json

# 3. After approval, apply only five notes
hermes-memory ingest-obsidian ... --limit 5 --apply --json

# 4. Verify BigQuery exact counts and no truncation
# Query document_sources/document_chunks by corpus_id and active state.

# 5. Search the actual corpus
hermes-memory search-docs --user tojo --agent hermes --query "..." --top-k 5

# 6. Re-run unchanged ingestion; expect zero embedding/write calls

# 7. Repeat with one clean pilot repository
hermes-memory ingest-repo ... --limit 20 --dry-run
hermes-memory ingest-repo ... --limit 20 --apply

# 8. Evaluate
hermes-memory evaluate-docs --queries evaluation/queries.yaml --json evaluation-result.json

# 9. Verify plugin health and real prefetch on Caladan
hermes memory status

# 10. With approval, verify fresh runtimes on Sietch Tabr and Arcanum
```

**Expected proof:** stable citations open the exact note/file location, unchanged rerun performs no writes, changed file replaces only its chunks, deleted source disappears only with `--prune`, Memory Bank continues working when document retrieval is disabled, and document retrieval fails open to Memory Bank rather than blocking the turn.

### Task 21: Independent review, commit, push, and smallest release

**Objective:** Satisfy the user's review and remote-delivery requirements honestly.

**Steps:**
1. Run an independent code review focused on source-policy bypass, SQL parameterization, data leakage, lifecycle loss, prompt-context injection, and retry duplication.
2. Resolve every REQUEST-CHANGES item and re-review until APPROVE.
3. Run security scan, `pytest -q`, package build, dry-run fixtures, and live pilot gate.
4. Commit any review fixes separately.
5. Push each implementation phase to the designated GitHub repository; no code remains local-only.
6. Do not tag a release until explicit approval.
7. After approval, publish the smallest honest versioned release and document limitations (especially unsupported source languages and no vector index).

## 7. Linear FUL workspace build plan

Plan mode forbids external writes in this turn, so the Linear workspace is **designed here but not created yet**. Build it immediately after the user approves execution, before code implementation.

### Project

- **Team:** FUL
- **Name:** `GCP Semantic Memory Ingestion`
- **Summary:** Citation-bearing Obsidian and Git retrieval using Vertex embeddings, BigQuery, and opt-in Memory Bank extraction.
- **Project description:** Link this plan file and the GitHub repository; state the no-new-database-service decision and sensitive-source exclusions.
- **Project labels:** Reuse existing labels where possible. Create at most `GCP Memory`, `Obsidian`, and `Git Repository` if equivalent labels do not exist.

### Milestones and issues

#### Milestone 1 — Contracts and GCP foundation

1. `Define fixtures, IDs, and evaluation queries` — Tasks 1-2.
2. `Implement Markdown structural chunking` — Tasks 3-4.
3. `Implement Vertex embedding and semantic boundary gateway` — Tasks 5-6; blocked by issue 2.
4. `[BOUNDED SENSITIVE] Apply BigQuery document schema` — Tasks 7-9; implementation can proceed offline, live apply requires approval.

#### Milestone 2 — Safe Obsidian ingestion

5. `Enforce source allowlists and secret rejection` — Task 10.
6. `Implement incremental Obsidian ingestion` — Task 11; blocked by 3, 4, 5.
7. `Expose preview-first Obsidian CLI` — Task 12; blocked by 6.
8. `[BOUNDED SENSITIVE] Run five-note Obsidian pilot` — Obsidian portion of Task 20; blocked by 7.

#### Milestone 3 — Git repository ingestion

9. `Discover clean Git sources and stable citations` — Task 13; blocked by 5.
10. `Spike and implement symbol-aware code units` — Task 14; blocked by 9.
11. `Expose ingest-repo and search-docs commands` — Task 15; blocked by 4, 10.
12. `[BOUNDED SENSITIVE] Run one-repository pilot` — repository portion of Task 20; blocked by 11.

#### Milestone 4 — Hermes integration and quality gate

13. `Add opt-in Memory Bank promotion` — Task 16; blocked by 6.
14. `Add citation-aware plugin prefetch` — Task 17; blocked by 3, 4.
15. `Implement retrieval quality evaluation` — Task 18; blocked by 11, 14.
16. `[BOUNDED SENSITIVE] Verify fresh fleet runtimes` — final part of Task 20; blocked by 8, 12, 14.

#### Milestone 5 — Documentation and release

17. `Document operation, cost, privacy, and rollback` — Task 19; blocked by 8 and 12.
18. `Independent security and code-quality review` — Task 21; blocked by 15, 16, 17.
19. `Definition of Done / release review` — blocked by 18 and every bounded pilot issue.

### Definition-of-Done issue checklist

- [ ] No new database/vector service was provisioned.
- [ ] Memory Bank model/config migration did not occur silently.
- [ ] BigQuery source/chunk lifecycle passed update/delete/interruption tests.
- [ ] All cloud SQL is parameterized.
- [ ] Sensitive paths/content are rejected without content disclosure.
- [ ] Obsidian and Git citations resolve to the correct source and lines.
- [ ] Unchanged reruns make zero embedding/write calls.
- [ ] Recall@5, MRR, citation, latency, and truncation metrics are recorded.
- [ ] Plugin fallback preserves Memory Bank when document search fails.
- [ ] `hermes memory status` and real fresh-session recall are verified on approved nodes.
- [ ] Independent review reached APPROVE.
- [ ] Commits are pushed; release tag has explicit approval.
- [ ] Rollback is tested/documented.

### Linear creation and verification procedure

1. Read this plan fresh.
2. Use `get_team`/`list_teams` to resolve FUL and inspect existing labels/projects.
3. Create or update one project—never a duplicate.
4. Create milestones.
5. Create all issues with the exact task links, file paths, commands, acceptance criteria, and bounded-sensitive titles above.
6. In a second pass, attach `blockedBy` relations after issue IDs exist.
7. Link the plan file and GitHub repository as project resources.
8. Read back the project with milestones/resources.
9. Read back at least the first/last issue of every milestone and the DoD issue with relations.
10. Programmatically compare expected milestone count, issue count, labels, and dependency edges to the returned workspace before claiming success.

## 8. Risks, tradeoffs, and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Embedding boundary scores vary by model revision | Chunk boundaries drift on re-ingest | Heading boundaries remain mandatory; cache unit embeddings; store model and content hashes; acceptance fixtures detect drift |
| One input per `gemini-embedding-001` request | Slow initial ingestion | Bounded concurrency, hash cache, semantic calls only for oversized sections, `--limit` pilots |
| BigQuery brute-force scan grows | Latency/cost rises | Measure bytes/p95; add an index only after gate failure |
| Memory Bank extraction loses exact provenance | False citation claims | Keep Memory Bank and source chunks separate; cite only BigQuery chunks |
| Whole-vault command captures sensitive material | Privacy/security incident | Allowlist-first policy, explicit apply, denylist and secret rejection, bounded pilots |
| Source update fails mid-run | Missing or mixed revision | Write/verify new chunks before deactivating old ones |
| Dirty Git checkout produces invalid GitHub link | Citation does not resolve | Reject apply by default; local citation only with explicit dirty override |
| Optional parser fails on a fleet OS | Package/runtime breakage | Lazy optional import; base line-window fallback; ingestion runs on Caladan first |
| Retrieved source text contains instructions | Prompt injection | Label as untrusted source context, cap excerpts, never execute instructions from retrieved documents |
| Promotional credit does not cover a SKU | Unexpected bill | Billing-credit preflight and post-pilot cost check; no eligibility assumptions |

## 9. Phase-2 gates—not current work

Open a later architecture decision only if measured evidence shows a gap:

1. **BigQuery vector index:** p95/bytes gate fails.
2. **RAG Engine:** managed parser/retrieval is worth a second store and duplicate lifecycle.
3. **Document AI:** PDF/DOCX-heavy corpus becomes a real requirement.
4. **Vertex AI Search:** hybrid enterprise search requirements appear.
5. **Cloud SQL/Firestore:** transactional lexicon/editor workflows require frequent point updates and joins that BigQuery handles poorly.
6. **Watcher/cron:** manual incremental ingestion is proven useful and staleness becomes a measured problem.
7. **Lexicon agent:** define only after corpus retrieval and source identity are reliable; do not infer lexicon entries from raw chat by default.

## 10. Open questions requiring user choice before live execution

1. Exact pilot Obsidian vault path and the three allowlisted subtrees/files.
2. Exact pilot Git repository path.
3. Whether approved operating-principle notes should use `--promote-to-memory-bank` in the first pilot or only after document retrieval evaluation.
4. Whether the optional code parser dependency is acceptable if its spike passes on Caladan; the base fallback remains available.
5. Whether the initial quality thresholds (Recall@5 0.80, p95 1.5s) are acceptable.
6. Which Linear project icon/color, if any; default to the FUL team's normal style rather than spending time on cosmetic setup.

## 11. Plan review checklist

- [x] GCP managed alternatives were checked against official documentation.
- [x] Managed chunking was not misrepresented as a standalone Vertex embedding feature.
- [x] Memory Bank and citation-bearing document retrieval are separated.
- [x] No new database service is introduced.
- [x] Existing source files and function boundaries are named exactly.
- [x] TDD, real runtime verification, review, commit, and push are included.
- [x] Sensitive sources and user approval gates are explicit.
- [x] Linear project/milestones/issues/dependencies and verification are specified.
- [x] Phase-2 infrastructure is gated by measurements, not speculation.

## Sources

[1] https://docs.cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/fine-tune-rag-transformations
[2] https://docs.cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/supported-documents
[3] https://docs.cloud.google.com/document-ai/docs/layout-parse-chunk
[4] https://docs.cloud.google.com/generative-ai-app-builder/docs/parse-chunk-documents
[5] https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/rag-engine/layout-parser-integration
[6] https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/embeddings
[7] https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/embeddings/task-types
[8] https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank
[9] https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/generate-memories
[10] https://docs.cloud.google.com/bigquery/docs/vector-search-intro
[11] https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing
[12] https://cloud.google.com/document-ai/pricing
[13] https://docs.cloud.google.com/billing/docs/how-to/reports/savings-and-credits
[14] https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/embeddings/get-text-embeddings
[15] https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/text-embeddings-api
[16] https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/rag-engine/vector-db-choices
[17] https://github.com/xberg-io/tree-sitter-language-pack
[18] https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions
[19] https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/embedding-2
[20] https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/embeddings/get-multimodal-embeddings
[21] https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/rag-engine/rag-engine-billing
[22] https://docs.cloud.google.com/document-ai/docs/reference/rest/v1beta3/ProcessOptions
[23] https://docs.cloud.google.com/generative-ai-app-builder/docs/prepare-data
[24] https://docs.cloud.google.com/bigquery/docs/generate-text-embedding
