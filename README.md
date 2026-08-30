# Hermes Agent Memory Layer
### Agent Platform · BigQuery · Memory Bank (GCP)

Persistent, scoped long-term memory for the Hermes Agent fleet (Caladan / Arcanum / Sietch Tabr) — bridged from local SQLite reflex probes to GCP-managed intelligence.

```
Hermes Agent (local, Lemonade, SQLite) ──┐
                                         ├──► Agent Platform Sessions ──► Memory Bank
                                         │         (AppendEvent)            (gemini-2.5-flash extraction
Hermes fleet probes                      │                                   text-embedding-005 search)
(reflex + reasoner, 0-token Go)          │                                    scope {user_id, agent_name}
                                         │
                                         └──► BigQuery (hermes_memory)
                                                  memories │ sessions │ memory_revisions
                                                  VECTOR_SEARCH │ analytics views │ TTL audit
```

#### Dual-retrieval (per turn)
1. **Memory Bank** `RetrieveMemories(similarity_search_params={query, top_k=8})` — semantic, managed, deduped
2. **BigQuery** `VECTOR_SEARCH + WHERE scope.user_id = ?` — auditable, analytical, filterable
3. **Merge** ranked + deduped → injected into Hermes prompt context

#### Write paths
- **Conversation → memory**: `session.events.append()` → `memories.generate(vertex_session_source=session)` → BigQuery mirror insert
- **Explicit fact**: `memories.generate(direct_contents_source=...)` or `CreateMemory`
- **Event streaming**: `ingest_events` (batched, triggers background generation)

---

## Quick start

```bash
# 1. Infra (idempotent)
export PROJECT_ID=gen-lang-client-0810135629
export LOCATION=us-central1
export BQ_LOCATION=US
./scripts/setup.sh                    # creates dataset + tables + views

# 2. Python
pip install -e ".[adk]"
export GOOGLE_CLOUD_PROJECT=$PROJECT_ID
export GOOGLE_CLOUD_LOCATION=$LOCATION
export GOOGLE_CLOUD_AGENT_ENGINE_ID=<memory-bank-id>  # after first create

# 3. Create Memory Bank (first time)
python -c "from hermes_memory.memory_bank import get_client, create_memory_bank; print(create_memory_bank())"

# 4. CLI
hermes-memory init --project $PROJECT_ID --location $LOCATION
hermes-memory search --user tojo --query "fleet topology"
hermes-memory stats --user tojo

# 5. Demo (works without live GCP — mock mode)
python examples/demo.py
```

## ADK integration

```python
from hermes_memory.adk_integration import build_memory_agent

agent, runner = build_memory_agent(project=PROJECT_ID, location=LOCATION)
# agent has: LoadMemoryTool + BigQuery tool + add_session_to_memory callback
```

See `examples/adk_agent_example.py` for full runtime deploy via `AdkApp` + `client.agent_engines.create()`.

## BigQuery schema

| Table | Purpose |
|---|---|
| `hermes_memory.memories` | Mirror of Memory Bank facts + embedding (REPEATED FLOAT), metadata JSON, TTL |
| `hermes_memory.sessions` | Session event log (JSON), user_id, timestamps |
| `hermes_memory.memory_revisions` | Audit trail from Memory Bank revisions API |
| `hermes_memory.document_sources` | Citation-bearing document source identity + revision lifecycle (active/inactive) |
| `hermes_memory.document_chunks` | Document chunks + retrieval embeddings (`gemini-embedding-001` @768) |
| Views | `recent_memories`, `memory_stats`, `user_timeline`, `active_document_sources` |

Vector search: `VECTOR_SEARCH(TABLE hermes_memory.memories, 'embedding', (SELECT text_embedding AS query ...), top_k=>8)`

## Semantic document ingestion (Vertex-native)

A **separate channel** from conversational Memory Bank: allowlisted Obsidian
notes and clean Git repos are chunked (Markdown structural + Python
symbol-aware via stdlib `ast`, no tree-sitter), embedded with Vertex
`gemini-embedding-001` @768 dims (`auto_truncate=False`), and stored as
**citation-bearing chunks** in BigQuery (`document_sources` / `document_chunks`,
US). Retrieval is exact **brute-force COSINE** `VECTOR_SEARCH` — **no vector
index**. The `gcp_memory_bank` plugin surfaces these as a document channel that
**fails open** to Memory Bank.

```bash
hermes-memory init --project "$PROJECT_ID" --location us-central1 --bq-location US
hermes-memory ingest-obsidian --user tojo --agent hermes --vault ~/Vaults/Hermes_Agent --limit 5 --json   # dry run
hermes-memory ingest-obsidian --user tojo --agent hermes --vault ~/Vaults/Hermes_Agent --limit 5 --apply   # apply
hermes-memory ingest-repo --user tojo --agent hermes --repo /path/to/checkout --ref HEAD --include "src/**" --language python --apply
hermes-memory search-docs --user tojo --agent hermes --query "..." --top-k 5
hermes-memory evaluate-docs --queries evaluation/queries.yaml --json evaluation-result.json
```

Channel separation: **Memory Bank = personal/conversational; BigQuery =
citation-bearing documents.** Promotion of durable principles to Memory Bank is
**opt-in** (`--promote-to-memory-bank`), Obsidian-default, and provenance-honest.
Migration from the old whole-note command retains command compatibility but
**does not auto-re-promote legacy notes**. **Rollback:** set
`document_retrieval_enabled: false` in plugin config (Memory Bank untouched);
tables carry `deletion_protection` + `prevent_destroy` and are removed only with
explicit approval.

**GenAI credit eligibility must be checked in GCP Billing** — not asserted here.

→ **Full operations guide:** [`docs/semantic-ingestion.md`](docs/semantic-ingestion.md)
(architecture, source policy, cost reporting, rollback, troubleshooting, limitations).

## Cost notes

- Memory Bank: **$0.25 / 1,000 stored memories** (generation billed as Gemini + embeddings)
- BigQuery: on-demand $6.25/TB scanned, storage $0.02/GB/mo — dataset is tiny (<100 MB)
- Generation model: `gemini-2.5-flash` (configurable), Memory embedding: `text-embedding-005`
- Document embedding: `gemini-embedding-001` @768 dims — rough per-run estimate printed by ingest (`cost_estimate=$…`, ~$0.15/1M tokens); estimate only, verify credits in Billing

## Hermes bridge

`HermesBridge` syncs local `~/.hermes/memory.db` (if present) to cloud without blocking the agent turn:
- `bridge.sync_session(session_id, user_id)` — ships events
- `bridge.retrieve_context(user_id, query, top_k=8)` — dual retrieval + merge
- `bridge.explicit_remember(user_id, fact)` — direct memory write

Offline-safe: all cloud calls have timeout + fallback to local SQLite.

## Terraform

```bash
cd terraform && terraform init && terraform apply -var="project_id=$PROJECT_ID"
```

## Repo layout

```
src/hermes_memory/  — pip package
bigquery/           — DDL (dataset, tables, views)
docs/               — semantic-ingestion.md (ingestion ops guide)
terraform/          — infra-as-code (document tables: deletion_protection + prevent_destroy)
scripts/setup.sh    — idempotent gcloud/bq bootstrap
examples/           — demo.py (mock), adk_agent_example.py
tests/              — import + schema tests
```

## Security

- IAM: `roles/aiplatform.memoryViewer` / `memoryEditor` scoped per `user_id` via IAM Conditions
- Org policy: `gcp.resourceLocations` to pin Memory Bank + BigQuery to `us-central1` / `US`
- Hermes Gateway E2EE unchanged; cloud memory is scoped, never cross-user

---
*Built for Hermes Agent v0.20.5 — Plan → Research → Implement workflow. v0.1.0 smallest honest useful release.*
