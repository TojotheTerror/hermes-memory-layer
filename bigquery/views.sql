-- Views — re-runnable
CREATE OR REPLACE VIEW `{{project}}.{{dataset}}.recent_memories` AS
SELECT memory_id, fact, JSON_VALUE(scope, '$.user_id') AS user_id, agent_name, created_at, expires_at
FROM `{{project}}.{{dataset}}.memories` ORDER BY created_at DESC;

CREATE OR REPLACE VIEW `{{project}}.{{dataset}}.memory_stats` AS
SELECT JSON_VALUE(scope,'$.user_id') AS user_id, COUNT(*) AS memory_count,
       MIN(created_at) AS first_memory_at, MAX(created_at) AS last_memory_at
FROM `{{project}}.{{dataset}}.memories` GROUP BY user_id;

CREATE OR REPLACE VIEW `{{project}}.{{dataset}}.user_timeline` AS
SELECT user_id, session_name, event_count, created_at FROM `{{project}}.{{dataset}}.sessions`
UNION ALL
SELECT JSON_VALUE(scope,'$.user_id') AS user_id, memory_id AS session_name, 1 AS event_count, created_at FROM `{{project}}.{{dataset}}.memories`
ORDER BY created_at DESC;

-- Active document corpus status — one row per (scope, corpus): count of active
-- sources and their active chunks. Useful for verifying an ingest run, checking
-- rollback (deactivated sources drop out), and a body-free size/cost sanity read.
CREATE OR REPLACE VIEW `{{project}}.{{dataset}}.active_document_sources` AS
SELECT
  s.user_id,
  s.agent_name,
  s.corpus_id,
  s.source_kind,
  COUNT(DISTINCT s.source_id) AS active_source_count,
  COUNT(c.chunk_id) AS active_chunk_count,
  MAX(s.last_seen_at) AS last_seen_at
FROM `{{project}}.{{dataset}}.document_sources` AS s
LEFT JOIN `{{project}}.{{dataset}}.document_chunks` AS c
  ON c.source_id = s.source_id AND c.is_active
WHERE s.is_active
GROUP BY s.user_id, s.agent_name, s.corpus_id, s.source_kind;
