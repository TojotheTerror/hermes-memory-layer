"""BigQuery store — dataset, tables, views, inserts, vector search, analytics."""

from __future__ import annotations

import re

from .config import HermesMemoryConfig, load_config

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


def _validate_ddl_identifiers(cfg: HermesMemoryConfig) -> None:
    """Reject unsafe or malformed identifiers before interpolating BigQuery DDL."""
    if _PROJECT_ID_PATTERN.fullmatch(cfg.project) is None:
        raise ValueError(f"Invalid BigQuery project: {cfg.project!r}")
    if _DATASET_ID_PATTERN.fullmatch(cfg.bq_dataset) is None:
        raise ValueError(f"Invalid BigQuery bq_dataset: {cfg.bq_dataset!r}")


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
    if not chunks:
        return 0
    cfg = cfg or load_config()
    _validate_ddl_identifiers(cfg)
    for chunk in chunks:
        if chunk.get("embedding_model") != embedding_model:
            raise ValueError(f"chunk {chunk.get('chunk_id')!r} has wrong embedding model")
        if chunk.get("embedding_dimensions") != embedding_dimensions:
            raise ValueError(f"chunk {chunk.get('chunk_id')!r} has wrong embedding dimensions")
        embedding = chunk.get("embedding")
        if embedding is None or len(embedding) != embedding_dimensions:
            raise ValueError(f"chunk {chunk.get('chunk_id')!r} has wrong embedding vector length")
    client = _bq_client(cfg)
    if client is None:
        raise RuntimeError("BigQuery client unavailable")

    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "chunk_id": chunk["chunk_id"],
            "source_id": chunk["source_id"],
            "corpus_id": chunk["corpus_id"],
            "user_id": user_id,
            "agent_name": agent_name,
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
            "metadata": chunk.get("metadata") or {},
            "is_active": False,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        for chunk in chunks
    ]
    table = f"{cfg.project}.{cfg.bq_dataset}.document_chunks"
    row_ids = [chunk["chunk_id"] for chunk in chunks]
    errors = client.insert_rows_json(table, rows, row_ids=row_ids)
    if errors:
        raise RuntimeError(f"BigQuery chunk insert failed: {errors}")
    return len(rows)


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
