# Vertex-Native Semantic Ingestion — Progress Ledger

**Repo:** `hermes-memory-layer`
**Integration branch:** `feat/vertex-semantic-ingestion`
**Integration HEAD:** `ac0ac091c4a47de350de899e77703ac902b5570f` (Tasks 1–5, 7, 18 integrated)
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
| 6 | Vertex semantic boundaries | APPROVED, awaits chunking cluster | `feat/vertex-semantic-boundaries` @ `f0c1709` |
| 8 | Restart-safe sync | corrected, final re-review running | `fix/task8-empty-batch-dimensions` @ `37bba66` |
| 9 | Exact filtered vector retrieval | APPROVED, awaits BigQuery cluster | `fix/vector-search-valid-base-query` @ `0e2c788` |
| 10 | Source allowlist/policy | APPROVED, awaits source cluster | `fix/source-policy-stable-read` @ `0ab7625` |
| 13 | Git discovery/stable citations | corrected, final re-review running | `fix/git-discovery-stable-citations` @ `480af31` |
| 14 | Symbol-aware code units | corrected, final re-review running | `fix/task14-source-preservation` @ `d9c7982` |
| 18 | Retrieval-quality evaluation gate | INTEGRATED | in `ac0ac09` (`ed47e25`) |
| 11 | Incremental Obsidian ingestion slice | PENDING | — |
| 12 | Preview-first Obsidian CLI | PENDING | — |
| 15 | ingest-repo / search-docs commands | PENDING | — |
| 16 | Provenance-honest Memory Bank promotion | PENDING | — |
| 17 | Citation-aware retrieval into prefetch | PENDING | — |
| 19 | Setup, docs, rollback guidance | PENDING | — |
| 20 | Live GCP pilots + fleet verification | GATED (user go-ahead) | — |
| 21 | Independent review, PR/CI, merge, release tag | GATED (user go-ahead) | — |

## Integration reconciliation clusters
- **Chunking:** Tasks 4 (integrated) / 6 / 14 — cherry-pick only the isolated
  Task 6 commit from its composite branch; reconcile with corrected Task 14.
- **BigQuery:** Tasks 8 / 9 — Task 8 writer must populate Task 9's denormalized
  `source_kind`, `content_kind`, `relative_path` chunk columns.
- **Source discovery:** Tasks 10 / 13 — reconcile immutable validated-byte
  records with the dirty-read stable-object binding.
- Task 18 is isolated to evaluation/CLI files.

## Active batch
`deleg_c4bfa08e` (8 subagents): correct Tasks 8/13/14; re-review Tasks
5/6/9/10/18. On completion: integrate each APPROVE onto the branch with
full-suite + Ruff verification and push; loop any REQUEST-CHANGES; then advance
to Tasks 11–19.

## Next steps (in order)
1. Consume `deleg_c4bfa08e` verdicts (ground-truth against git, not summaries).
2. Integrate APPROVE'd tasks; re-verify full suite; push each.
3. Re-review the three corrected tasks (8/13/14).
4. Build Tasks 11, 12, 15, 16, 17 (each: TDD → spec review → quality review → integrate).
5. Task 19 docs/rollback.
6. STOP at Task 20 and Task 21 — request explicit user go-ahead.
