"""BigQuery store — dataset, tables, views, inserts, vector search, analytics."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

from .config import HermesMemoryConfig, load_config

# BigQuery's streaming insertAll caps a single request at ~10 MB and a bounded
# row count. Split large chunk inserts so no request exceeds either limit; the
# byte budget stays under 10 MB with headroom for request envelope overhead.
_MAX_INSERT_ROWS = 500
_MAX_INSERT_BYTES = 9_000_000


def _row_json_bytes(row: dict) -> int:
    """Estimate a row's serialized insertAll payload size in bytes."""
    import json as _json

    return len(_json.dumps(row, default=str, ensure_ascii=False).encode("utf-8"))


# DDL lives in bigquery/*.sql for review; Python helpers are thin wrappers
# so `bq` CLI or Terraform can also apply them without Python.

DDL_DATASET = """CREATE SCHEMA IF NOT EXISTS `{project}.{dataset}`
OPTIONS(location="{bq_location}", description="Hermes Agent memory layer — mirror of Memory Bank + analytics");
"""

DDL_MEMORIES = """CREATE TABLE IF NOT EXISTS `{project}.{dataset}.memories` (
  memory_id STRING NOT NULL OPTIONS(description="Memory Bank resource name or UUID"),
  fact STRING NOT NULL OPTIONS(description="Extracted fact text"),
  scope JSON OPTIONS(description='Scope e.g. {{"user_id":"tojo","agent_name":"hermes"}}'),
  user_id STRING OPTIONS(description="Denormalized user_id for partitioning"),
  agent_name STRING,
  embedding ARRAY<FLOAT64> OPTIONS(description="text-embedding-005 vector (768d) — null until backfill"),
  metadata JSON,
  source STRING OPTIONS(description="session|direct|revision"),
  session_name STRING,
  created_at TIMESTAMP NOT NULL OPTIONS(description="Insertion time"),
  updated_at TIMESTAMP NOT NULL,
  expires_at TIMESTAMP OPTIONS(description="TTL expiry"),
  revision_id STRING
) OPTIONS(description="Hermes memories — BigQuery mirror of Memory Bank");
"""

DDL_SESSIONS = """CREATE TABLE IF NOT EXISTS `{project}.{dataset}.sessions` (
  session_name STRING NOT NULL,
  session_id STRING NOT NULL,
  user_id STRING NOT NULL,
  agent_name STRING,
  events JSON OPTIONS(description="Ordered SessionEvents JSON array"),
  event_count INT64,
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL
) OPTIONS(description="Agent Platform Sessions event log (BigQuery copy)");
"""

DDL_REVISIONS = """CREATE TABLE IF NOT EXISTS `{project}.{dataset}.memory_revisions` (
  revision_name STRING NOT NULL,
  memory_id STRING NOT NULL,
  fact STRING NOT NULL,
  extracted_memories JSON OPTIONS(description="Intermediate extraction JSON"),
  scope JSON,
  created_at TIMESTAMP NOT NULL,
  ttl_expires_at TIMESTAMP
) OPTIONS(description="Memory revision audit trail");
"""

DDL_DOCUMENT_SOURCES = """CREATE TABLE IF NOT EXISTS `{project}.{dataset}.document_sources` (
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
"""

DDL_DOCUMENT_CHUNKS = """CREATE TABLE IF NOT EXISTS `{project}.{dataset}.document_chunks` (
  chunk_id STRING NOT NULL,
  source_id STRING NOT NULL,
  corpus_id STRING NOT NULL,
  user_id STRING NOT NULL,
  agent_name STRING NOT NULL,
  source_kind STRING NOT NULL,
  content_kind STRING NOT NULL,
  relative_path STRING NOT NULL,
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
"""

DDL_VIEWS = {
    "recent_memories": """CREATE OR REPLACE VIEW `{project}.{dataset}.recent_memories` AS
SELECT memory_id, fact, JSON_VALUE(scope, '$.user_id') AS user_id, agent_name, created_at, expires_at
FROM `{project}.{dataset}.memories`
ORDER BY created_at DESC;
""",
    "memory_stats": """CREATE OR REPLACE VIEW `{project}.{dataset}.memory_stats` AS
SELECT JSON_VALUE(scope,'$.user_id') AS user_id, COUNT(*) AS memory_count,
       MIN(created_at) AS first_memory_at, MAX(created_at) AS last_memory_at
FROM `{project}.{dataset}.memories`
GROUP BY user_id;
""",
    "user_timeline": """CREATE OR REPLACE VIEW `{project}.{dataset}.user_timeline` AS
SELECT user_id, session_name, event_count, created_at FROM `{project}.{dataset}.sessions`
UNION ALL
SELECT JSON_VALUE(scope,'$.user_id') AS user_id, memory_id AS session_name, 1 AS event_count, created_at
FROM `{project}.{dataset}.memories`
ORDER BY created_at DESC;
""",
}

_PROJECT_ID_PATTERN = re.compile(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", flags=re.ASCII)
_DATASET_ID_PATTERN = re.compile(r"[A-Za-z0-9_]{1,1024}", flags=re.ASCII)


@dataclass(frozen=True, slots=True)
class DocumentChunkSearchResult:
    """Immutable citation-bearing document vector-search hit."""

    distance: float
    content: str
    contextual_content: str
    citation: str
    source_path: str
    heading_path: tuple[str, ...]
    symbol: str | None
    start_line: int | None
    end_line: int | None
    chunk_id: str
    source_id: str
    corpus_id: str


_DOCUMENT_SEARCH_RESULT_FIELDS = {
    "distance",
    "content",
    "contextual_content",
    "citation",
    "source_path",
    "heading_path",
    "symbol",
    "start_line",
    "end_line",
    "chunk_id",
    "source_id",
    "corpus_id",
}


def _document_search_result_from_row(row) -> DocumentChunkSearchResult:
    """Validate a query row without including source values in failures."""
    try:
        values = dict(row)
        if set(values) != _DOCUMENT_SEARCH_RESULT_FIELDS:
            raise TypeError
        distance = values["distance"]
        if isinstance(distance, bool) or not isinstance(distance, Real):
            raise TypeError
        normalized_distance = float(distance)
        if not math.isfinite(normalized_distance):
            raise ValueError
        string_fields = (
            "content",
            "contextual_content",
            "citation",
            "source_path",
            "chunk_id",
            "source_id",
            "corpus_id",
        )
        if any(not isinstance(values[name], str) for name in string_fields):
            raise TypeError
        heading_path = values["heading_path"]
        if heading_path is None:
            normalized_heading_path = ()
        elif isinstance(heading_path, (list, tuple)) and all(
            isinstance(heading, str) for heading in heading_path
        ):
            normalized_heading_path = tuple(heading_path)
        else:
            raise TypeError
        if values["symbol"] is not None and not isinstance(values["symbol"], str):
            raise TypeError
        for name in ("start_line", "end_line"):
            if values[name] is not None and type(values[name]) is not int:
                raise TypeError
        return DocumentChunkSearchResult(
            distance=normalized_distance,
            content=values["content"],
            contextual_content=values["contextual_content"],
            citation=values["citation"],
            source_path=values["source_path"],
            heading_path=normalized_heading_path,
            symbol=values["symbol"],
            start_line=values["start_line"],
            end_line=values["end_line"],
            chunk_id=values["chunk_id"],
            source_id=values["source_id"],
            corpus_id=values["corpus_id"],
        )
    except (KeyError, TypeError, ValueError):
        raise ValueError("malformed document search result") from None


def _validate_ddl_identifiers(cfg: HermesMemoryConfig) -> None:
    """Reject unsafe or malformed identifiers before interpolating BigQuery DDL."""
    if _PROJECT_ID_PATTERN.fullmatch(cfg.project) is None:
        raise ValueError(f"Invalid BigQuery project: {cfg.project!r}")
    if _DATASET_ID_PATTERN.fullmatch(cfg.bq_dataset) is None:
        raise ValueError(f"Invalid BigQuery bq_dataset: {cfg.bq_dataset!r}")


def _validate_query_embedding(query_embedding, dimensions: int) -> list[float]:
    """Normalize a finite real vector without exposing values in errors."""
    try:
        values = tuple(query_embedding)
    except TypeError as exc:
        raise TypeError("query_embedding must be an iterable of real numbers") from exc
    if len(values) != dimensions:
        raise ValueError(f"query_embedding must contain exactly {dimensions} values")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise TypeError("query_embedding values must be real numbers")
    embedding = [float(value) for value in values]
    if not all(math.isfinite(value) for value in embedding):
        raise ValueError("query_embedding values must be finite")
    return embedding


def _bq_client(cfg: HermesMemoryConfig):
    try:
        from google.cloud import bigquery

        # Don't pin location on client — BigQuery is global endpoint, dataset location is per-resource.
        return bigquery.Client(project=cfg.project)
    except ImportError as e:
        raise RuntimeError(
            "google-cloud-bigquery not installed. pip install google-cloud-bigquery"
        ) from e
    except Exception as e:
        print(f"[hermes-memory] BigQuery client unavailable: {e}")
        return None


def ensure_dataset(cfg: HermesMemoryConfig | None = None):
    cfg = cfg or load_config()
    client = _bq_client(cfg)
    if client is None:
        print(f"[mock] ensure_dataset {cfg.project}.{cfg.bq_dataset} ({cfg.bq_location})")
        return
    from google.cloud import bigquery

    dataset_id = f"{cfg.project}.{cfg.bq_dataset}"
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = cfg.bq_location
    dataset.description = "Hermes Agent memory layer — mirror of Memory Bank + analytics"
    client.create_dataset(dataset, exists_ok=True)
    print(f"Dataset ready: {dataset_id}")


def ensure_tables(cfg: HermesMemoryConfig | None = None):
    cfg = cfg or load_config()
    _validate_ddl_identifiers(cfg)
    client = _bq_client(cfg)
    if client is None:
        print("[mock] ensure_tables skipped")
        return
    tables = [
        ("memories", DDL_MEMORIES),
        ("sessions", DDL_SESSIONS),
        ("revisions", DDL_REVISIONS),
        ("document_sources", DDL_DOCUMENT_SOURCES),
        ("document_chunks", DDL_DOCUMENT_CHUNKS),
    ]
    for name, ddl in tables:
        sql = ddl.format(project=cfg.project, dataset=cfg.bq_dataset, bq_location=cfg.bq_location)
        client.query(sql).result()
        print(f"Table ready: {cfg.bq_dataset}.{name}")
    for vname, vddl in DDL_VIEWS.items():
        sql = vddl.format(project=cfg.project, dataset=cfg.bq_dataset)
        client.query(sql).result()
        print(f"View ready: {cfg.bq_dataset}.{vname}")


def get_source_state(
    source_id: str,
    *,
    user_id: str,
    agent_name: str,
    cfg: HermesMemoryConfig | None = None,
    client=None,
) -> dict | None:
    """Return the current source row for an identity, if it exists."""
    cfg = cfg or load_config()
    _validate_ddl_identifiers(cfg)
    client = client or _bq_client(cfg)
    if client is None:
        return None

    from google.cloud import bigquery as _bq

    table = f"`{cfg.project}.{cfg.bq_dataset}.document_sources`"
    sql = f"""SELECT source_id, revision, content_hash, is_active
FROM {table}
WHERE source_id = @source_id AND user_id = @user_id AND agent_name = @agent_name
LIMIT 1"""
    job_config = _bq.QueryJobConfig(
        query_parameters=[
            _bq.ScalarQueryParameter("source_id", "STRING", source_id),
            _bq.ScalarQueryParameter("user_id", "STRING", user_id),
            _bq.ScalarQueryParameter("agent_name", "STRING", agent_name),
        ]
    )
    rows = list(client.query(sql, job_config=job_config).result())
    return dict(rows[0]) if rows else None


def upsert_source(
    source: dict,
    *,
    user_id: str,
    agent_name: str,
    cfg: HermesMemoryConfig | None = None,
) -> bool:
    """Return false without writing when the active source revision is unchanged."""
    cfg = cfg or load_config()
    _validate_ddl_identifiers(cfg)
    client = _bq_client(cfg)
    state = get_source_state(
        source["source_id"],
        user_id=user_id,
        agent_name=agent_name,
        cfg=cfg,
        client=client,
    )
    if (
        state
        and state.get("is_active")
        and state.get("revision") == source["revision"]
        and state.get("content_hash") == source["content_hash"]
    ):
        return False
    if client is None:
        raise RuntimeError("BigQuery client unavailable")

    import json as _json
    from google.cloud import bigquery as _bq

    table = f"`{cfg.project}.{cfg.bq_dataset}.document_sources`"
    sql = f"""MERGE {table} AS target
USING (SELECT @source_id AS source_id, @user_id AS user_id, @agent_name AS agent_name) AS incoming
ON target.source_id = incoming.source_id
  AND target.user_id = incoming.user_id
  AND target.agent_name = incoming.agent_name
WHEN MATCHED AND NOT target.is_active THEN UPDATE SET
  corpus_id = @corpus_id,
  source_kind = @source_kind,
  content_kind = @content_kind,
  relative_path = @relative_path,
  source_uri = @source_uri,
  revision = @revision,
  content_hash = @content_hash,
  metadata = PARSE_JSON(@metadata),
  last_seen_at = CURRENT_TIMESTAMP(),
  updated_at = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN INSERT
  (source_id, corpus_id, user_id, agent_name, source_kind, content_kind,
   relative_path, source_uri, revision, content_hash, metadata, is_active,
   first_seen_at, last_seen_at, updated_at)
VALUES
  (@source_id, @corpus_id, @user_id, @agent_name, @source_kind, @content_kind,
   @relative_path, @source_uri, @revision, @content_hash, PARSE_JSON(@metadata), FALSE,
   CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())"""
    job_config = _bq.QueryJobConfig(
        query_parameters=[
            _bq.ScalarQueryParameter("source_id", "STRING", source["source_id"]),
            _bq.ScalarQueryParameter("corpus_id", "STRING", source["corpus_id"]),
            _bq.ScalarQueryParameter("user_id", "STRING", user_id),
            _bq.ScalarQueryParameter("agent_name", "STRING", agent_name),
            _bq.ScalarQueryParameter("source_kind", "STRING", source["source_kind"]),
            _bq.ScalarQueryParameter("content_kind", "STRING", source["content_kind"]),
            _bq.ScalarQueryParameter("relative_path", "STRING", source["relative_path"]),
            _bq.ScalarQueryParameter("source_uri", "STRING", source["source_uri"]),
            _bq.ScalarQueryParameter("revision", "STRING", source["revision"]),
            _bq.ScalarQueryParameter("content_hash", "STRING", source["content_hash"]),
            _bq.ScalarQueryParameter(
                "metadata", "STRING", _json.dumps(source.get("metadata") or {})
            ),
        ]
    )
    client.query(sql, job_config=job_config).result()
    return True


def _validate_embedding_dimensions(value: object) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError("embedding dimensions must be a positive integer")


def insert_chunks(
    chunks: list[dict],
    *,
    user_id: str,
    agent_name: str,
    embedding_model: str,
    embedding_dimensions: int,
    cfg: HermesMemoryConfig | None = None,
) -> int:
    """Insert inactive chunks using chunk identities as retry-stable insert IDs."""
    _validate_embedding_dimensions(embedding_dimensions)
    if not chunks:
        return 0
    cfg = cfg or load_config()
    _validate_ddl_identifiers(cfg)
    seen_chunk_ids: set = set()
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if chunk_id in seen_chunk_ids:
            # Chunk IDs double as retry-stable insert IDs; duplicates would
            # collapse rows and corrupt insert-count and completeness accounting.
            raise ValueError("duplicate chunk_id in insert batch")
        seen_chunk_ids.add(chunk_id)
        declared_dimensions = chunk.get("embedding_dimensions")
        _validate_embedding_dimensions(declared_dimensions)
        if chunk.get("embedding_model") != embedding_model:
            raise ValueError(f"chunk {chunk.get('chunk_id')!r} has wrong embedding model")
        if declared_dimensions != embedding_dimensions:
            raise ValueError(f"chunk {chunk.get('chunk_id')!r} has wrong embedding dimensions")
        embedding = chunk.get("embedding")
        if embedding is None or len(embedding) != embedding_dimensions:
            raise ValueError(f"chunk {chunk.get('chunk_id')!r} has wrong embedding vector length")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in embedding):
            raise ValueError("chunk embedding must contain only finite numeric values")
    client = _bq_client(cfg)
    if client is None:
        raise RuntimeError("BigQuery client unavailable")

    from datetime import datetime, timezone
    import json as _json

    timestamp = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "chunk_id": chunk["chunk_id"],
            "source_id": chunk["source_id"],
            "corpus_id": chunk["corpus_id"],
            "user_id": user_id,
            "agent_name": agent_name,
            "source_kind": chunk["source_kind"],
            "content_kind": chunk["content_kind"],
            "relative_path": chunk["relative_path"],
            "ordinal": chunk["ordinal"],
            "content": chunk["text"],
            "contextual_content": chunk["contextual_text"],
            "content_hash": chunk["content_hash"],
            "heading_path": list(chunk.get("heading_path") or ()),
            "symbol": chunk.get("symbol"),
            "start_line": chunk.get("start_line"),
            "end_line": chunk.get("end_line"),
            "citation": chunk["citation"],
            "embedding": list(chunk["embedding"]),
            "embedding_model": embedding_model,
            "embedding_dimensions": embedding_dimensions,
            "metadata": _json.dumps(chunk.get("metadata") or {}),
            "is_active": False,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        for chunk in chunks
    ]
    table = f"{cfg.project}.{cfg.bq_dataset}.document_chunks"
    inserted = 0
    for batch_rows, batch_ids in _batched_insert_rows(rows):
        errors = client.insert_rows_json(table, batch_rows, row_ids=batch_ids)
        if errors:
            raise RuntimeError(f"BigQuery chunk insert failed: {errors}")
        inserted += len(batch_rows)
    return inserted


def _batched_insert_rows(rows: list[dict]):
    """Yield (rows, row_ids) batches bounded by row count and payload bytes.

    Each batch stays within `_MAX_INSERT_ROWS` and `_MAX_INSERT_BYTES`; a single
    row that alone exceeds the byte cap is emitted as its own batch rather than
    dropped, so the yielded batches always form an exact, order-preserving
    partition of `rows` with no drops or duplicates.
    """
    batch: list[dict] = []
    batch_ids: list[str] = []
    batch_bytes = 0
    for row in rows:
        row_bytes = _row_json_bytes(row)
        if batch and (
            len(batch) >= _MAX_INSERT_ROWS or batch_bytes + row_bytes > _MAX_INSERT_BYTES
        ):
            yield batch, batch_ids
            batch, batch_ids, batch_bytes = [], [], 0
        batch.append(row)
        batch_ids.append(row["chunk_id"])
        batch_bytes += row_bytes
    if batch:
        yield batch, batch_ids


def finalize_source_revision(
    source_id: str,
    active_chunk_ids: list[str],
    *,
    source: dict,
    user_id: str,
    agent_name: str,
    cfg: HermesMemoryConfig | None = None,
) -> None:
    """Atomically activate complete chunks and their source revision metadata."""
    if source["source_id"] != source_id:
        raise ValueError("source_id does not match source metadata")

    cfg = cfg or load_config()
    _validate_ddl_identifiers(cfg)
    client = _bq_client(cfg)
    if client is None:
        raise RuntimeError("BigQuery client unavailable")

    import json as _json
    from google.cloud import bigquery as _bq

    chunk_ids = list(dict.fromkeys(active_chunk_ids))
    chunks = f"`{cfg.project}.{cfg.bq_dataset}.document_chunks`"
    sources = f"`{cfg.project}.{cfg.bq_dataset}.document_sources`"
    sql = f"""BEGIN TRANSACTION;
ASSERT (
  SELECT COUNT(DISTINCT chunk_id)
  FROM {chunks}
  WHERE source_id = @source_id
    AND user_id = @user_id
    AND agent_name = @agent_name
    AND chunk_id IN UNNEST(@active_chunk_ids)
) = ARRAY_LENGTH(@active_chunk_ids)
AS 'expected chunks incomplete; source revision not finalized';
UPDATE {chunks}
SET is_active = chunk_id IN UNNEST(@active_chunk_ids),
    updated_at = CURRENT_TIMESTAMP()
WHERE source_id = @source_id
  AND user_id = @user_id
  AND agent_name = @agent_name
  AND (is_active OR chunk_id IN UNNEST(@active_chunk_ids));
MERGE {sources} AS target
USING (SELECT @source_id AS source_id, @user_id AS user_id, @agent_name AS agent_name) AS incoming
ON target.source_id = incoming.source_id
  AND target.user_id = incoming.user_id
  AND target.agent_name = incoming.agent_name
WHEN MATCHED THEN UPDATE SET
  corpus_id = @corpus_id,
  source_kind = @source_kind,
  content_kind = @content_kind,
  relative_path = @relative_path,
  source_uri = @source_uri,
  revision = @revision,
  content_hash = @content_hash,
  metadata = PARSE_JSON(@metadata),
  is_active = TRUE,
  last_seen_at = CURRENT_TIMESTAMP(),
  updated_at = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN INSERT
  (source_id, corpus_id, user_id, agent_name, source_kind, content_kind,
   relative_path, source_uri, revision, content_hash, metadata, is_active,
   first_seen_at, last_seen_at, updated_at)
VALUES
  (@source_id, @corpus_id, @user_id, @agent_name, @source_kind, @content_kind,
   @relative_path, @source_uri, @revision, @content_hash, PARSE_JSON(@metadata), TRUE,
   CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP());
COMMIT TRANSACTION;"""
    job_config = _bq.QueryJobConfig(
        query_parameters=[
            _bq.ScalarQueryParameter("source_id", "STRING", source_id),
            _bq.ScalarQueryParameter("corpus_id", "STRING", source["corpus_id"]),
            _bq.ScalarQueryParameter("user_id", "STRING", user_id),
            _bq.ScalarQueryParameter("agent_name", "STRING", agent_name),
            _bq.ScalarQueryParameter("source_kind", "STRING", source["source_kind"]),
            _bq.ScalarQueryParameter("content_kind", "STRING", source["content_kind"]),
            _bq.ScalarQueryParameter("relative_path", "STRING", source["relative_path"]),
            _bq.ScalarQueryParameter("source_uri", "STRING", source["source_uri"]),
            _bq.ScalarQueryParameter("revision", "STRING", source["revision"]),
            _bq.ScalarQueryParameter("content_hash", "STRING", source["content_hash"]),
            _bq.ScalarQueryParameter(
                "metadata", "STRING", _json.dumps(source.get("metadata") or {})
            ),
            _bq.ArrayQueryParameter("active_chunk_ids", "STRING", chunk_ids),
        ]
    )
    client.query(sql, job_config=job_config).result()


def deactivate_missing_sources(
    corpus_id: str,
    seen_source_ids: list[str],
    *,
    user_id: str,
    agent_name: str,
    prune: bool = False,
    limited: bool = False,
    cfg: HermesMemoryConfig | None = None,
) -> None:
    """Deactivate unseen corpus sources only for an explicit, complete prune run."""
    if not prune:
        raise ValueError("deactivate_missing_sources requires prune=True")
    if limited:
        raise ValueError("cannot prune during a limited run")

    cfg = cfg or load_config()
    _validate_ddl_identifiers(cfg)
    client = _bq_client(cfg)
    if client is None:
        raise RuntimeError("BigQuery client unavailable")

    from google.cloud import bigquery as _bq

    sources = f"`{cfg.project}.{cfg.bq_dataset}.document_sources`"
    chunks = f"`{cfg.project}.{cfg.bq_dataset}.document_chunks`"
    sql = f"""BEGIN TRANSACTION;
UPDATE {chunks}
SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP()
WHERE corpus_id = @corpus_id
  AND user_id = @user_id
  AND agent_name = @agent_name
  AND source_id NOT IN UNNEST(@seen_source_ids);
UPDATE {sources}
SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP()
WHERE corpus_id = @corpus_id
  AND user_id = @user_id
  AND agent_name = @agent_name
  AND source_id NOT IN UNNEST(@seen_source_ids);
COMMIT TRANSACTION;"""
    job_config = _bq.QueryJobConfig(
        query_parameters=[
            _bq.ScalarQueryParameter("corpus_id", "STRING", corpus_id),
            _bq.ArrayQueryParameter(
                "seen_source_ids", "STRING", list(dict.fromkeys(seen_source_ids))
            ),
            _bq.ScalarQueryParameter("user_id", "STRING", user_id),
            _bq.ScalarQueryParameter("agent_name", "STRING", agent_name),
        ]
    )
    client.query(sql, job_config=job_config).result()


def insert_memory(
    fact: str,
    scope: dict,
    cfg: HermesMemoryConfig | None = None,
    memory_id: str | None = None,
    embedding: list[float] | None = None,
    metadata: dict | None = None,
    source: str = "direct",
    session_name: str | None = None,
):
    cfg = cfg or load_config()
    client = _bq_client(cfg)
    import uuid

    mem_id = memory_id or str(uuid.uuid4())
    if client is None:
        print(f"[mock] insert_memory: {mem_id} fact={fact[:60]} scope={scope}")
        return {"memory_id": mem_id, "fact": fact, "scope": scope}
    # Parameterized query — string concatenation SQL breaks on newlines,
    # quotes, and other special characters in arbitrary fact text.
    import json as _json
    from google.cloud import bigquery as _bq

    table = f"`{cfg.project}.{cfg.bq_dataset}.memories`"
    sql = f"""INSERT INTO {table} (memory_id, fact, scope, user_id, agent_name, source, session_name, metadata, created_at, updated_at, expires_at)
    VALUES (@memory_id, @fact, PARSE_JSON(@scope), @user_id, @agent_name, @source, @session_name, PARSE_JSON(@metadata), CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL @ttl_days DAY))"""
    job_config = _bq.QueryJobConfig(
        query_parameters=[
            _bq.ScalarQueryParameter("memory_id", "STRING", mem_id),
            _bq.ScalarQueryParameter("fact", "STRING", fact),
            _bq.ScalarQueryParameter("scope", "STRING", _json.dumps(scope)),
            _bq.ScalarQueryParameter("user_id", "STRING", scope.get("user_id", "")),
            _bq.ScalarQueryParameter("agent_name", "STRING", scope.get("agent_name", "")),
            _bq.ScalarQueryParameter("source", "STRING", source),
            _bq.ScalarQueryParameter("session_name", "STRING", session_name),
            _bq.ScalarQueryParameter(
                "metadata", "STRING", _json.dumps(metadata) if metadata else "null"
            ),
            _bq.ScalarQueryParameter("ttl_days", "INT64", cfg.ttl_days),
        ]
    )
    try:
        client.query(sql, job_config=job_config).result()
        print(f"Inserted memory {mem_id}")
    except Exception as e:
        print(f"BigQuery insert failed: {e}")
        # fallback: log but don't crash
        return {"memory_id": mem_id, "fact": fact, "scope": scope, "error": str(e)}
    return {"memory_id": mem_id, "fact": fact, "scope": scope}


def insert_session(
    session_name: str,
    user_id: str,
    events: list[dict],
    cfg: HermesMemoryConfig | None = None,
    agent_name: str | None = None,
):
    cfg = cfg or load_config()
    client = _bq_client(cfg)
    session_id = session_name.split("/")[-1]
    if client is None:
        print(f"[mock] insert_session: {session_name} events={len(events)}")
        return {"session_name": session_name, "session_id": session_id, "user_id": user_id}
    import json as _json
    from google.cloud import bigquery as _bq

    table = f"`{cfg.project}.{cfg.bq_dataset}.sessions`"
    sql = f"""INSERT INTO {table} (session_name, session_id, user_id, agent_name, events, event_count, created_at, updated_at)
    VALUES (@session_name, @session_id, @user_id, @agent_name, PARSE_JSON(@events), @event_count, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())"""
    job_config = _bq.QueryJobConfig(
        query_parameters=[
            _bq.ScalarQueryParameter("session_name", "STRING", session_name),
            _bq.ScalarQueryParameter("session_id", "STRING", session_id),
            _bq.ScalarQueryParameter("user_id", "STRING", user_id),
            _bq.ScalarQueryParameter("agent_name", "STRING", agent_name),
            _bq.ScalarQueryParameter("events", "STRING", _json.dumps(events)),
            _bq.ScalarQueryParameter("event_count", "INT64", len(events)),
        ]
    )
    try:
        client.query(sql, job_config=job_config).result()
        print(f"Inserted session {session_name}")
    except Exception as e:
        print(f"BigQuery session insert failed: {e}")
    return {"session_name": session_name, "session_id": session_id, "user_id": user_id}


def vector_search(
    query_embedding: list[float],
    scope: dict | None = None,
    top_k: int = 8,
    cfg: HermesMemoryConfig | None = None,
) -> list[dict]:
    """BigQuery VECTOR_SEARCH — requires embeddings populated. Mock fallback returns SQL string."""
    cfg = cfg or load_config()
    if scope and scope.get("user_id"):
        where = f"WHERE JSON_VALUE(scope, '$.user_id') = '{scope['user_id']}'"
    else:
        where = ""
    sql = f"""SELECT base.memory_id, base.fact, base.scope, distance
FROM VECTOR_SEARCH(
  (SELECT * FROM `{cfg.project}.{cfg.bq_dataset}.memories` {where}),
  'embedding',
  (SELECT {query_embedding} AS query),
  top_k => {top_k}, distance_type => 'COSINE')
ORDER BY distance ASC"""
    client = _bq_client(cfg)
    if client is None:
        print(f"[mock] vector_search SQL:\n{sql[:400]}...")
        return [{"fact": "[mock] bq vector hit", "distance": 0.12, "sql": sql}]
    # Real execution would need query_embedding as a proper vector table;
    # for now return the SQL for caller to adapt if needed.
    print("VECTOR_SEARCH requires populated embeddings — returning SQL for reference.")
    return [{"sql": sql}]


def search_document_chunks(
    query_embedding: Iterable[Real],
    *,
    user_id: str,
    agent_name: str,
    corpus_id: str | None = None,
    source_kind: str | None = None,
    content_kind: str | None = None,
    top_k: int | None = None,
    cfg: HermesMemoryConfig | None = None,
) -> list[DocumentChunkSearchResult]:
    """Return exact vector matches from active, tenant-scoped document chunks."""
    from google.cloud import bigquery

    cfg = cfg or load_config()
    _validate_ddl_identifiers(cfg)
    embedding = _validate_query_embedding(query_embedding, cfg.document_embedding_dimensions)
    if top_k is None:
        top_k = cfg.document_top_k
    if type(top_k) is not int:
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    optional_predicates = []
    query_parameters = [
        bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", embedding),
        bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        bigquery.ScalarQueryParameter("agent_name", "STRING", agent_name),
        bigquery.ScalarQueryParameter("embedding_model", "STRING", cfg.document_embedding_model),
        bigquery.ScalarQueryParameter(
            "embedding_dimensions", "INT64", cfg.document_embedding_dimensions
        ),
    ]
    for name, value in (
        ("corpus_id", corpus_id),
        ("source_kind", source_kind),
        ("content_kind", content_kind),
    ):
        if value is not None:
            optional_predicates.append(f"    AND chunks.{name} = @{name}")
            query_parameters.append(bigquery.ScalarQueryParameter(name, "STRING", value))
    query_parameters.append(bigquery.ScalarQueryParameter("top_k", "INT64", top_k))

    optional_sql = "\n".join(optional_predicates)
    chunks_table = f"`{cfg.project}.{cfg.bq_dataset}.document_chunks`"
    # Source finalization and pruning atomically mirror source activation into chunk rows.
    # Denormalized source fields keep every required predicate in this legal base-table query.
    sql = f"""SELECT
  distance,
  base.content,
  base.contextual_content,
  base.citation,
  base.relative_path AS source_path,
  base.heading_path,
  base.symbol,
  base.start_line,
  base.end_line,
  base.chunk_id,
  base.source_id,
  base.corpus_id
FROM VECTOR_SEARCH(
  (SELECT
    chunks.embedding,
    chunks.content,
    chunks.contextual_content,
    chunks.citation,
    chunks.relative_path,
    chunks.heading_path,
    chunks.symbol,
    chunks.start_line,
    chunks.end_line,
    chunks.chunk_id,
    chunks.source_id,
    chunks.corpus_id
  FROM {chunks_table} AS chunks
  WHERE chunks.is_active = TRUE
    AND chunks.user_id = @user_id
    AND chunks.agent_name = @agent_name
    AND chunks.embedding_model = @embedding_model
    AND chunks.embedding_dimensions = @embedding_dimensions
{optional_sql}),
  'embedding',
  (SELECT @query_embedding AS embedding),
  query_column_to_search => 'embedding',
  top_k => @top_k,
  distance_type => 'COSINE',
  options => '{{"use_brute_force":true}}'
)
ORDER BY distance ASC, base.chunk_id ASC"""
    client = _bq_client(cfg)
    if client is None:
        return []
    rows = client.query(
        sql, job_config=bigquery.QueryJobConfig(query_parameters=query_parameters)
    ).result()
    return [_document_search_result_from_row(row) for row in rows]


def query_memories_sql(user_id: str, limit: int = 20, cfg: HermesMemoryConfig | None = None) -> str:
    cfg = cfg or load_config()
    return f"SELECT memory_id, fact, created_at FROM `{cfg.project}.{cfg.bq_dataset}.memories` WHERE user_id = '{user_id}' ORDER BY created_at DESC LIMIT {limit}"


def fetch_stats(user_id: str | None = None, cfg: HermesMemoryConfig | None = None) -> list[dict]:
    cfg = cfg or load_config()
    client = _bq_client(cfg)
    if client is None:
        return [{"user_id": user_id or "tojo", "memory_count": 0, "mock": True}]
    where = f"WHERE user_id = '{user_id}'" if user_id else ""
    sql = f"SELECT * FROM `{cfg.project}.{cfg.bq_dataset}.memory_stats` {where}"
    try:
        rows = list(client.query(sql).result())
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"stats query failed (tables may not exist yet): {e}")
        return []
