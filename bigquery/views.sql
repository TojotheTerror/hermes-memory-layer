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
