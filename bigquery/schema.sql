-- BigQuery DDL — idempotent, apply via bq query or terraform
-- Dataset creation is separate (bq mk --location=US or Terraform google_bigquery_dataset)

-- memories: mirror of Memory Bank facts + BigQuery-native fields
CREATE TABLE IF NOT EXISTS `{{project}}.{{dataset}}.memories` (
  memory_id STRING NOT NULL OPTIONS(description="Memory Bank resource name or UUID"),
  fact STRING NOT NULL,
  scope JSON OPTIONS(description='{"user_id":"tojo","agent_name":"hermes"}'),
  user_id STRING,
  agent_name STRING,
  embedding ARRAY<FLOAT64> OPTIONS(description="text-embedding-005 768d"),
  metadata JSON,
  source STRING OPTIONS(description="session|direct|revision"),
  session_name STRING,
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL,
  expires_at TIMESTAMP,
  revision_id STRING
) OPTIONS(description="Hermes memories — BigQuery mirror");

CREATE TABLE IF NOT EXISTS `{{project}}.{{dataset}}.sessions` (
  session_name STRING NOT NULL,
  session_id STRING NOT NULL,
  user_id STRING NOT NULL,
  agent_name STRING,
  events JSON,
  event_count INT64,
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL
) OPTIONS(description="Sessions event log");

CREATE TABLE IF NOT EXISTS `{{project}}.{{dataset}}.memory_revisions` (
  revision_name STRING NOT NULL,
  memory_id STRING NOT NULL,
  fact STRING NOT NULL,
  extracted_memories JSON,
  scope JSON,
  created_at TIMESTAMP NOT NULL,
  ttl_expires_at TIMESTAMP
) OPTIONS(description="Revision audit trail");

-- document_sources: canonical source identity and revision lifecycle
CREATE TABLE IF NOT EXISTS `{{project}}.{{dataset}}.document_sources` (
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

-- document_chunks: citation-bearing text and retrieval embeddings
CREATE TABLE IF NOT EXISTS `{{project}}.{{dataset}}.document_chunks` (
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
  embedding_task_type STRING NOT NULL,
  embedding_dimensions INT64 NOT NULL,
  metadata JSON,
  is_active BOOL NOT NULL,
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL
)
CLUSTER BY user_id, agent_name, corpus_id, source_id;
