"""BigQuery store — dataset, tables, views, inserts, vector search, analytics."""
from __future__ import annotations

import datetime
import json
from typing import Any

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


def _bq_client(cfg: HermesMemoryConfig):
    try:
        from google.cloud import bigquery
        # Don't pin location on client — BigQuery is global endpoint, dataset location is per-resource.
        return bigquery.Client(project=cfg.project)
    except ImportError as e:
        raise RuntimeError("google-cloud-bigquery not installed. pip install google-cloud-bigquery") from e
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
    client = _bq_client(cfg)
    if client is None:
        print("[mock] ensure_tables skipped")
        return
    for name, ddl in [("memories", DDL_MEMORIES), ("sessions", DDL_SESSIONS), ("revisions", DDL_REVISIONS)]:
        sql = ddl.format(project=cfg.project, dataset=cfg.bq_dataset, bq_location=cfg.bq_location)
        client.query(sql).result()
        print(f"Table ready: {cfg.bq_dataset}.{name}")
    for vname, vddl in DDL_VIEWS.items():
        sql = vddl.format(project=cfg.project, dataset=cfg.bq_dataset)
        client.query(sql).result()
        print(f"View ready: {cfg.bq_dataset}.{vname}")


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
    # Use DML INSERT (reliable vs streaming insert eventual-consistency quirks)
    import json as _json
    # Escape single quotes for SQL string literal
    safe_fact = fact.replace("'", "\\'")
    safe_scope = _json.dumps(scope).replace("'", "\\'")
    safe_meta = _json.dumps(metadata).replace("'", "\\'") if metadata else None
    meta_expr = f"JSON '{safe_meta}'" if safe_meta else "NULL"
    table = f"`{cfg.project}.{cfg.bq_dataset}.memories`"
    sql = f"""INSERT INTO {table} (memory_id, fact, scope, user_id, agent_name, source, session_name, metadata, created_at, updated_at, expires_at)
    VALUES ('{mem_id}', '{safe_fact}', JSON '{safe_scope}', '{scope.get('user_id','')}', '{scope.get('agent_name','')}', '{source}', {f"'{session_name}'" if session_name else 'NULL'}, {meta_expr}, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL {cfg.ttl_days} DAY))"""
    try:
        client.query(sql).result()
        print(f"Inserted memory {mem_id}")
    except Exception as e:
        print(f"BigQuery insert failed: {e}")
        # fallback: log but don't crash
        return {"memory_id": mem_id, "fact": fact, "scope": scope, "error": str(e)}
    return {"memory_id": mem_id, "fact": fact, "scope": scope}


def insert_session(session_name: str, user_id: str, events: list[dict], cfg: HermesMemoryConfig | None = None, agent_name: str | None = None):
    cfg = cfg or load_config()
    client = _bq_client(cfg)
    session_id = session_name.split("/")[-1]
    if client is None:
        print(f"[mock] insert_session: {session_name} events={len(events)}")
        return {"session_name": session_name, "session_id": session_id, "user_id": user_id}
    import json as _json
    safe_events = _json.dumps(events).replace("'", "\\'")
    agent_expr = f"'{agent_name}'" if agent_name else "NULL"
    table = f"`{cfg.project}.{cfg.bq_dataset}.sessions`"
    sql = f"""INSERT INTO {table} (session_name, session_id, user_id, agent_name, events, event_count, created_at, updated_at)
    VALUES ('{session_name}', '{session_id}', '{user_id}', {agent_expr}, JSON '{safe_events}', {len(events)}, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())"""
    try:
        client.query(sql).result()
        print(f"Inserted session {session_name}")
    except Exception as e:
        print(f"BigQuery session insert failed: {e}")
    return {"session_name": session_name, "session_id": session_id, "user_id": user_id}


def vector_search(query_embedding: list[float], scope: dict | None = None, top_k: int = 8, cfg: HermesMemoryConfig | None = None) -> list[dict]:
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
