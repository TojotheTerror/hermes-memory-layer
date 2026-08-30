# Vertex-Native Semantic Ingestion — Progress Ledger

**Repo:** `hermes-memory-layer`
**Integration branch:** `feat/vertex-semantic-ingestion`
**Integration HEAD:** `e081823` (Tasks 1–19 ALL integrated — offline build COMPLETE; 716 tests green)
**Governing plan:** `.hermes/plans/2026-08-29_121924-vertex-semantic-ingestion.md`
**Last updated:** 2026-08-30

## Durability rule
Progress is durable ONLY after push + remote-verify. `/tmp` worktrees are
disposable scratch; every candidate commit is pushed to its own origin branch
so a `/tmp` wipe loses nothing. This was validated after an actual `/tmp`
cleanup — all branches recovered from origin intact.

## Standing guardrails (user-set)
No live BigQuery DDL · no billed Vertex calls · no source ingestion · no
Memory Bank promotion · no runtime restart · no release tag. Task 20 (live
pilots) and Task 21 (release tag) require explicit user go-ahead. Every
implementation commit must be pushed, never local-only. Reviews must reach a
real APPROVE — no self-approval, no merge on green CI alone.

## Task status

| # | Task | State | Candidate branch / SHA |
|---|------|-------|------------------------|
| 1 | Contracts, fixtures, isolated bridge | INTEGRATED | in `48a8da2` (via `5b73223`) |
| 2 | Document configuration | INTEGRATED | in `efa30f3` (via `cbae690`) |
| 3 | Markdown structural parsing | INTEGRATED | `28362b0` (via `c7291ad`) |
| 4 | Deterministic packing/overlap | INTEGRATED | in `48a8da2` (`2f0db4c`) |
| 7 | BigQuery document tables | INTEGRATED | via `4fc2e2d` |
| 5 | Vertex embedding gateway | INTEGRATED | in `ac0ac09` (`cebec48`) |
| 6 | Vertex semantic boundaries | INTEGRATED | in `fa65a69` (chunking cluster) |
| 8 | Restart-safe sync | INTEGRATED | in `fa65a69` (bigquery cluster) |
| 9 | Exact filtered vector retrieval | INTEGRATED | in `fa65a69` (bigquery cluster) |
| 10 | Source allowlist/policy | INTEGRATED | in `fa65a69` (source cluster) |
| 13 | Git discovery/stable citations | INTEGRATED | in `fa65a69` (source cluster) |
| 14 | Symbol-aware code units | INTEGRATED | in `fa65a69` (chunking cluster) |
| 18 | Retrieval-quality evaluation gate | INTEGRATED | in `ac0ac09` (`ed47e25`) |
| 11 | Incremental Obsidian ingestion slice | INTEGRATED | in `79ec231` (`057d0de`) |
| 12 | Preview-first Obsidian CLI | IN PROGRESS (TDD) | touches `cli.py` |
| 15 | ingest-repo / search-docs commands | HELD (needs 12's cli.py) | — |
| 16 | Provenance-honest Memory Bank promotion | IN PROGRESS (TDD) | `ingestion.py`+`memory_bank.py` |
| 17 | Citation-aware retrieval into prefetch | IN PROGRESS (TDD) | `hermes_bridge.py`+plugin |
| 19 | Setup, docs, rollback guidance | PENDING | — |
| 20 | Live GCP pilots + fleet verification | GATED (user go-ahead) | — |
| 21 | Independent review, PR/CI, merge, release tag | GATED (user go-ahead) | — |

## User decision log
- 2026-08-30: User APPROVED proceeding into the Task 20 live pilot (billed
  Vertex + BigQuery DDL + 5-note Obsidian ingestion + Memory Bank promotion +
  runtime verification), CONDITIONAL on the code being fully built and
  independently reviewed first. Task 21 release tag still requires a separate
  explicit go-ahead. Confirm exact commands at the moment of each billed/
  irreversible step before running it.
- 2026-08-30: Offline build COMPLETE (Tasks 1-19 integrated @ a6f56f5, 716
  tests). Pilot scope decisions: (a) RETRIEVAL-ONLY first pilot — NO Memory
  Bank promotion this round (promote in a later run); (b) NO gateway restart —
  verify document retrieval out-of-process via `search-docs` against live
  BigQuery; (c) user gave go to begin, confirming each billed step.
- 2026-08-30: TASK 20 LIVE PILOT SUCCEEDED (retrieval-only, no restart).
  Two live-only bugs (mocks missed) found + fixed via TDD during the pilot:
  (1) insert_chunks passed metadata dict raw to streaming insert -> "metadata
  is not a record"; (2) streaming-buffer rows block finalize's UPDATE
  activation. Fix: write chunks via load_table_from_json (managed storage,
  dict JSON) — commit f1daf6f. Chose Option B for DDL: created only
  document_sources + document_chunks via `bq mk` (never touched live
  memories/sessions). Verified live: 17 chunks + 2 sources active, 768-dim
  embeddings, `search-docs` returns ranked COSINE results with line-anchored
  citations. Total spend ~$0.0015. Pilot artifacts: BigQuery tables live;
  pilot vault ~/Vaults/_pilot_hermes_docs (2 approved notes, copies).
- 2026-08-30: Load-job fix (f1daf6f) INDEPENDENTLY VALIDATED — dual APPROVE
  (quality/security + spec-compliance) at HEAD 050d0f7. Spec review noted the
  load job is all-or-nothing per call (stronger than streaming) and WRITE_APPEND
  preserves the active revision (restart-safe). Task 20 pilot fully closed.
  KNOWN NON-BLOCKING FOLLOW-UP (backlog, not a release blocker): the load-job
  path lost streaming row_ids dedup, so a source inserted-but-not-finalized then
  retried could accumulate physical DUPLICATE INACTIVE rows. These NEVER
  activate/retrieve (finalize's completeness-ASSERT + activate-by-id gate what
  goes live), so it is storage hygiene only. Harden later via a
  `DELETE WHERE source_id=@sid AND is_active=FALSE` (or dedup pass) before the
  load — deferred to keep the data-path change reviewed-and-minimal.

## Integration reconciliation clusters
- **Chunking:** Tasks 4 (integrated) / 6 / 14 — cherry-pick only the isolated
  Task 6 commit from its composite branch; reconcile with corrected Task 14.
- **BigQuery:** Tasks 8 / 9 — Task 8 writer must populate Task 9's denormalized
  `source_kind`, `content_kind`, `relative_path` chunk columns.
- **Source discovery:** Tasks 10 / 13 — reconcile immutable validated-byte
  records with the dirty-read stable-object binding.
- Task 18 is isolated to evaluation/CLI files.

## Active batch
`deleg_25f28524` (2 subagents): independent spec-compliance + quality/security
review of the Task 11 candidate `057d0de`. On dual-APPROVE: integrate onto the
branch with full-suite + Ruff verification and push. Loop any REQUEST-CHANGES.

## Next steps (in order)
1. Consume `deleg_25f28524` verdicts (ground-truth against git, not summaries).
2. On APPROVE, integrate Task 11; re-verify full suite from clean checkout; push.
3. Fan out now-unblocked Tasks 12/15/16/17 in parallel (share settled `ingestion.py`):
   - 12 preview-first Obsidian CLI · 15 ingest-repo/search-docs · 16 Memory Bank
     promotion · 17 citation-aware plugin prefetch. Each: TDD → spec review →
     quality review → integrate.
4. Task 19 docs/rollback (needs 12).
5. STOP at Task 20 (live pilots) and Task 21 (release tag) — explicit user go-ahead.
