---
title: Agent Memory Operations
tags:
  - hermes
  - agent-memory
  - runbook
aliases: [Memory Runbook]
---

# Agent Memory Operations

This note describes the deterministic test corpus used for memory operations.

## Storage

### Layout

The service keeps three durable layers:

- Memory Bank facts for personalized recall.
- BigQuery rows for analytics and citations.
- Local memory for offline continuity.

```yaml
memory:
  provider: gcp_memory_bank
  retrieval_limit: 5
```

### Recovery

1. Confirm the local queue is readable.
2. Compare the active source revision.
3. Replay only the missing chunk identifiers.

#### Verification

A successful recovery reports the same source identifier before and after replay.

## Oversized Incident Narrative

At 09:00 the operator opened the incident log and recorded the current source revision, active chunk count, and retrieval scope before touching any state. This sentence is deliberately detailed so the section exceeds the pilot chunk target while remaining harmless synthetic prose.

At 09:05 the first diagnostic compared the canonical root with the stored corpus identity and found that both values matched. The operator noted that deterministic identity must not depend on traversal order, temporary directories, wall-clock time, or a cloud response.

At 09:10 the second diagnostic normalized line endings in a copied note and verified that the content digest remained stable. No production file was read, no credential was requested, and every identifier in this fixture is a fake test marker.

At 09:15 the team reviewed heading ancestry and kept the top-level incident heading attached to each atomic paragraph. Nested headings remained mandatory boundaries, while paragraph and list boundaries remained candidates for later structural packing.

At 09:20 an intentionally long paragraph described why a source citation needs both a relative path and an inclusive line range. A reader should be able to open the cited file, inspect the exact excerpt, and understand which heading or symbol supplied its context.

At 09:25 the dry-run report listed planned sources, planned chunks, skipped paths, and estimated embedding requests without mutating a remote service. The report contained no raw secret values and used only synthetic names from this committed fixture.

At 09:30 the operator simulated a retry after a transient embedding failure. The planned chunk identifiers did not change because retries reuse normalized content, heading occurrence, source identity, and content hashes rather than generating random values.

At 09:35 a second simulated worker processed the same paragraphs in reverse discovery order. Sorting by normalized relative path and source order produced the same source identifier, the same ordinal sequence, and the same inclusive line ranges.

At 09:40 the review checked overlap behavior. Any overlap must use complete trailing atomic units, must not cross an unrelated top-level heading, and must preserve the original line metadata rather than inventing a synthetic location.

At 09:45 the team tested an unchanged rerun. The source digest matched the active revision, so the dry-run planned zero embedding requests and zero writes while preserving all active chunks and their existing citations.

At 09:50 the team changed one sentence in a temporary copy. Only the affected content-addressed chunk received a new identifier; unaffected chunks retained their identifiers even when later line numbers shifted in the edited source.

At 09:55 the replacement simulation inserted every expected new chunk before deactivating stale identifiers. An interrupted attempt left the previous active revision retrievable, demonstrating that lifecycle updates must not create an empty source window.

At 10:00 the evaluator issued a query about replaying missing chunks. The expected answer came from the concise Recovery section, not this narrative, proving that an oversized section can coexist with a more precise retrieval target.

At 10:05 the evaluator issued a query about canonical roots. The expected result retained the Operations/agent-memory.md path, the Oversized Incident Narrative heading, and the exact inclusive lines containing the matching explanation.

At 10:10 the operator reviewed list handling and fenced code handling from the earlier Layout section. A structural parser must keep list items and the complete YAML fence as atomic units instead of cutting through markers or dropping their heading ancestry.

At 10:15 the source policy rejected credential-shaped and ignored paths before parsing. Detection failures remained fail-closed for an individual file, and reports counted rejection reasons without printing the contents of a denied file.

At 10:20 the repository pilot used a synthetic Go file with interfaces, structs, functions, and methods. Symbol boundaries were expected when a tested parser was available, with deterministic line-window fallback when it was not.

At 10:25 the operator compared retrieval channels. Memory Bank facts stayed separate from citation-bearing document excerpts so that a generalized personal fact could never be presented as though it had an exact source line citation.

At 10:30 the evaluator measured recall, reciprocal rank, citation validity, truncation, and latency using fake deterministic results. These pilot gates informed defaults but were not described as global service-level guarantees.

At 10:35 a bounded concurrency check preserved input order even when fake embedding calls completed out of order. Each result remained paired with the normalized text and task type that produced it, with no reliance on undocumented batch behavior.

At 10:40 the report displayed model name, output dimensions, request count, and estimated characters. It avoided promising promotional credit eligibility because billing-account status must be verified separately from repository behavior.

At 10:45 the final dry-run repeated every operation without cloud, gateway, Memory Bank, BigQuery, or production local-memory access. Injected fakes supplied all three bridge channels and recorded exactly which arguments the test passed.

At 10:50 the reviewer confirmed that the fixture contains only synthetic operational prose. Names, paths, environment values, tokens, and identifiers are fake markers intended for unit tests and cannot authenticate to any service.

At 10:55 the incident closed after focused and full tests returned green. The retained evidence included deterministic identifiers, stable inclusive line ranges, explicit dependency calls, and a clean diff limited to the approved task.

## Runtime

### Recovery

- Stop new writes in the synthetic test harness.
- Drain the fake queue.
- Resume after the fixture reports `healthy`.

#### Verification

The runtime recovery heading intentionally repeats the earlier Recovery heading.
