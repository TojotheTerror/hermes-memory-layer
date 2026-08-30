"""CLI: hermes-memory init | search | remember | stats | sync"""

from __future__ import annotations

import json
from pathlib import Path

import click

from .config import load_config
from .hermes_bridge import HermesBridge


@click.group()
def main():
    """Hermes Agent memory layer — Memory Bank + BigQuery."""
    pass


@main.command()
@click.option("--project", default=None, help="GCP project id")
@click.option("--location", default="us-central1")
@click.option("--bq-location", default="US")
def init(project, location, bq_location):
    """Create BigQuery dataset + tables + views (idempotent)."""
    cfg = load_config(project=project, location=location, bq_location=bq_location)
    click.echo(
        f"Project: {cfg.project}  Location: {cfg.location}  BQ: {cfg.bq_location}  Dataset: {cfg.bq_dataset}"
    )
    from .bigquery_store import ensure_dataset, ensure_tables

    ensure_dataset(cfg)
    ensure_tables(cfg)
    click.echo(
        'Init done. Next: create a Memory Bank with python -c "from hermes_memory.memory_bank import create_memory_bank; print(create_memory_bank())"'
    )


@main.command("evaluate-docs")
@click.option(
    "--queries",
    "queries_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--json",
    "json_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
def evaluate_docs(queries_path: Path, json_path: Path):
    """Measure document retrieval against the initial pilot gates."""
    from . import evaluation

    try:
        queries = evaluation.load_queries(queries_path)
        execute_search = evaluation.make_runtime_query_executor()
        report = evaluation.evaluate_queries(queries, execute_search)
        evaluation.write_report_json(report, json_path)
    except Exception:
        raise click.ClickException("document evaluation failed") from None
    if not report.pilot_gate.passed:
        raise click.exceptions.Exit(1)


@main.command()
@click.option("--user", "user_id", required=True, help="user_id scope")
@click.option("--query", required=True, help="Search query")
@click.option("--top-k", default=8, type=int)
def search(user_id, query, top_k):
    """Dual-retrieval search (Memory Bank + BigQuery + local)."""
    bridge = HermesBridge()
    result = bridge.retrieve_context(user_id=user_id, query=query, top_k=top_k)
    click.echo(json.dumps(result, indent=2, default=str))
    if result.get("prompt_context"):
        click.echo("\n--- Prompt context ---")
        click.echo(result["prompt_context"])


@main.command()
@click.option("--user", "user_id", required=True)
@click.option("--fact", required=True, help="Fact to remember")
def remember(user_id, fact):
    """Explicitly remember a fact (Memory Bank + BigQuery)."""
    bridge = HermesBridge()
    result = bridge.explicit_remember(user_id=user_id, fact=fact)
    click.echo(json.dumps(result, indent=2, default=str))


@main.command()
@click.option("--user", "user_id", default=None, help="Filter by user_id")
def stats(user_id):
    """Show memory stats from BigQuery."""
    from .bigquery_store import fetch_stats

    rows = fetch_stats(user_id=user_id)
    click.echo(json.dumps(rows, indent=2, default=str))


@main.command("sync-local")
@click.option("--limit", default=10, type=int, help="How many local memories to preview")
def sync_local(limit):
    """Preview local Hermes SQLite memories (no cloud write)."""
    from .hermes_bridge import read_local_memories

    rows = read_local_memories(limit=limit)
    click.echo(json.dumps(rows, indent=2, default=str))
    click.echo(f"\nFound {len(rows)} local memories.")


@main.command("seed")
@click.option("--user", "user_id", required=True, help="user_id scope")
@click.option("--agent", "agent_name", default="hermes", help="agent_name scope")
@click.option("--dry-run", is_flag=True, help="Preview without writing")
def seed(user_id, agent_name, dry_run):
    """Ingest curated MEMORY.md / USER.md facts into Memory Bank + BigQuery (deduped)."""
    from .bigquery_store import query_memories_sql, _bq_client
    from .hermes_bridge import read_curated_memory_files

    facts = read_curated_memory_files()
    click.echo(f"Found {len(facts)} curated facts in MEMORY.md / USER.md")

    bridge = HermesBridge()

    # dedupe against what's already in BigQuery for this scope
    existing: set[str] = set()
    try:
        client = _bq_client(bridge.cfg)
        if client is not None:
            sql = query_memories_sql(user_id, limit=1000, cfg=bridge.cfg)
            for r in client.query(sql).result():
                existing.add(str(r["fact"]).strip().lower())
    except Exception as e:
        click.echo(f"[seed] warn: could not fetch existing facts for dedupe: {e}")

    new_facts = [f for f in facts if f["fact"].strip().lower() not in existing]
    click.echo(
        f"{len(new_facts)} new (not already in corpus), {len(facts) - len(new_facts)} already present"
    )

    if dry_run:
        for f in new_facts:
            click.echo(
                f"  [dry-run] would write ({f['kind']} / {f['source_file']}): {f['fact'][:80]}..."
            )
        return

    written = 0
    for f in new_facts:
        result = bridge.explicit_remember(
            user_id=user_id,
            fact=f["fact"],
            agent_name=agent_name,
            metadata={"kind": f["kind"], "source_file": f["source_file"], "seed": True},
        )
        errs = {k: v for k, v in result.items() if k.endswith("_error")}
        if errs:
            click.echo(f"  [FAIL] {f['fact'][:60]}... -> {errs}")
        else:
            written += 1
            click.echo(f"  [OK] {f['fact'][:60]}...")

    click.echo(
        f"\nSeeded {written}/{len(new_facts)} new facts for scope user_id={user_id}, agent_name={agent_name}"
    )


def _dry_run_state_reader(source_id, *, user_id, agent_name):
    """Preview-safe state reader: reports no prior state, touching no network.

    A dry run never queries BigQuery ``document_sources``; it presents the full
    plan as if nothing were previously ingested. This keeps the preview honest
    (it never claims a skip it did not verify) while constructing no client.
    """
    return None


# --- Apply-time dependency seams --------------------------------------------
#
# Each seam constructs exactly one real cloud dependency and is only ever called
# on an ``--apply`` run. Tests monkeypatch these to inject fakes, so a dry run
# (which never calls any seam) constructs no client and makes no network call.


def _build_embedding_client(cfg):
    from google import genai

    from .embeddings import VertexEmbeddingClient

    sdk_client = genai.Client(vertexai=True, project=cfg.project, location=cfg.location)
    return VertexEmbeddingClient(
        client=sdk_client,
        model=cfg.document_embedding_model,
        dimensions=cfg.document_embedding_dimensions,
        task_type="RETRIEVAL_DOCUMENT",
    )


def _build_state_reader(cfg):
    from . import bigquery_store

    def reader(source_id, *, user_id, agent_name):
        return bigquery_store.get_source_state(
            source_id, user_id=user_id, agent_name=agent_name, cfg=cfg
        )

    return reader


def _build_insert_chunks(cfg):
    from . import bigquery_store

    return bigquery_store.insert_chunks


def _build_finalize_source_revision(cfg):
    from . import bigquery_store

    return bigquery_store.finalize_source_revision


def _build_semantic_gateway(cfg):
    # No semantic boundary gateway wired for the personal CLI yet; structural
    # chunks are used as-is. Returning None keeps apply on the deterministic path.
    return None


def _build_memory_bank(cfg):
    from .hermes_bridge import HermesBridge
    from .memory_bank import generate_from_contents

    bridge = HermesBridge()
    return generate_from_contents, bridge.memory_bank_name


def _deactivate_missing_sources(
    corpus_id, seen_source_ids, *, user_id, agent_name, prune, limited, cfg
):
    from . import bigquery_store

    bigquery_store.deactivate_missing_sources(
        corpus_id,
        seen_source_ids,
        user_id=user_id,
        agent_name=agent_name,
        prune=prune,
        limited=limited,
        cfg=cfg,
    )


def _sorted_sources(sources):
    return sorted(sources, key=lambda s: s.relative_path)


def _sorted_rejected(rejected):
    return sorted(rejected, key=lambda r: (r.path, r.rule))


def _plan_payload(plan, *, mode, user_id, agent_name):
    """Build the deterministic, body-free serializable view of a plan."""
    return {
        "mode": mode,
        "user_id": user_id,
        "agent_name": agent_name,
        "counts": {
            "discovered": len(plan.discovered),
            "skipped": len(plan.skipped),
            "rejected": len(plan.rejected),
            "chunks": plan.chunk_count,
            "requests": plan.request_count,
            "tokens": plan.token_count,
        },
        "cost_estimate": plan.cost_estimate,
        "discovered": [
            {
                "relative_path": s.relative_path,
                "status": s.status,
                "chunk_count": s.chunk_count,
                "token_count": s.token_count,
            }
            for s in _sorted_sources(plan.discovered)
        ],
        "skipped": [
            {
                "relative_path": s.relative_path,
                "status": s.status,
                "chunk_count": s.chunk_count,
                "token_count": s.token_count,
            }
            for s in _sorted_sources(plan.skipped)
        ],
        "rejected": [{"path": r.path, "rule": r.rule} for r in _sorted_rejected(plan.rejected)],
    }


def _render_plan(plan, *, mode, user_id, agent_name):
    """Render a deterministic, body-free plan table (never prints note bodies)."""
    lines: list[str] = []
    lines.append(f"Obsidian ingestion {mode} — user_id={user_id} agent_name={agent_name}")
    if mode == "preview":
        lines.append("(dry run: no BigQuery/Vertex/Memory Bank client constructed; no writes)")
    lines.append(
        f"discovered={len(plan.discovered)} skipped={len(plan.skipped)} "
        f"rejected={len(plan.rejected)}"
    )
    lines.append(
        f"chunks={plan.chunk_count} requests={plan.request_count} "
        f"tokens={plan.token_count} cost_estimate=${plan.cost_estimate:.6f}"
    )
    if plan.discovered:
        lines.append("discovered sources:")
        for s in _sorted_sources(plan.discovered):
            lines.append(
                f"  [{s.status}] {s.relative_path} (chunks={s.chunk_count}, tokens={s.token_count})"
            )
    if plan.skipped:
        lines.append("skipped sources:")
        for s in _sorted_sources(plan.skipped):
            lines.append(f"  [skipped] {s.relative_path}")
    if plan.rejected:
        lines.append("rejected sources:")
        for r in _sorted_rejected(plan.rejected):
            lines.append(f"  [rejected:{r.rule}] {r.path}")
    return "\n".join(lines)


def _limit_plan(plan, limit):
    """Return a new plan capping discovered sources to the first ``limit``.

    Sources are ordered deterministically by relative_path before truncation so
    a limited run is reproducible. Skipped/rejected are preserved untouched.
    """
    from .ingestion import IngestionPlan, _cost_estimate

    if limit is None or limit >= len(plan.discovered):
        return plan
    kept = tuple(_sorted_sources(plan.discovered)[:limit])
    chunk_count = sum(s.chunk_count for s in kept)
    token_count = sum(s.token_count for s in kept)
    return IngestionPlan(
        discovered=kept,
        skipped=plan.skipped,
        rejected=plan.rejected,
        chunk_count=chunk_count,
        request_count=chunk_count,
        token_count=token_count,
        cost_estimate=_cost_estimate(token_count),
    )


def _report_payload(report, *, user_id, agent_name):
    """Deterministic, body-free serializable view of an apply report."""
    return {
        "mode": "apply",
        "user_id": user_id,
        "agent_name": agent_name,
        "counts": {
            "written": len(report.discovered),
            "skipped": len(report.skipped),
            "rejected": len(report.rejected),
            "chunks": report.chunk_count,
            "requests": report.request_count,
            "tokens": report.token_count,
        },
        "cost_estimate": report.cost_estimate,
        "promotion_status": report.promotion_status,
        "written": [
            {
                "relative_path": o.relative_path,
                "status": o.status,
                "chunk_count": o.chunk_count,
                "token_count": o.token_count,
            }
            for o in sorted(report.discovered, key=lambda o: o.relative_path)
        ],
        "skipped": [
            {"relative_path": o.relative_path, "status": o.status}
            for o in sorted(report.skipped, key=lambda o: o.relative_path)
        ],
        "rejected": [{"path": r.path, "rule": r.rule} for r in _sorted_rejected(report.rejected)],
    }


def _render_report(report, *, user_id, agent_name):
    """Render a deterministic, body-free apply report (never prints bodies)."""
    lines: list[str] = []
    lines.append(f"Obsidian ingestion applied — user_id={user_id} agent_name={agent_name}")
    lines.append(
        f"written={len(report.discovered)} skipped={len(report.skipped)} "
        f"rejected={len(report.rejected)}"
    )
    lines.append(
        f"chunks={report.chunk_count} requests={report.request_count} "
        f"tokens={report.token_count} cost_estimate=${report.cost_estimate:.6f}"
    )
    lines.append(f"promotion_status={report.promotion_status}")
    if report.discovered:
        lines.append("written sources:")
        for o in sorted(report.discovered, key=lambda o: o.relative_path):
            lines.append(
                f"  [{o.status}] {o.relative_path} (chunks={o.chunk_count}, tokens={o.token_count})"
            )
    return "\n".join(lines)


@main.command("ingest-obsidian")
@click.option("--user", "user_id", required=True, help="user_id scope")
@click.option("--agent", "agent_name", default="hermes", help="agent_name scope")
@click.option(
    "--vault",
    "vaults",
    multiple=True,
    help="Vault path to ingest (allowlist). Repeatable; at least one is REQUIRED.",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Cap the number of new/changed sources processed on an --apply run.",
)
@click.option("--apply", "apply_writes", is_flag=True, help="Perform writes (required to write).")
@click.option(
    "--promote-to-memory-bank",
    "promote",
    is_flag=True,
    help="Also extract facts into Memory Bank (requires --apply).",
)
@click.option(
    "--prune",
    is_flag=True,
    help="Deactivate sources absent from the vault (requires --apply; not with --limit).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the plan/report as deterministic JSON.")
@click.option(
    "--batch-chars",
    default=None,
    type=int,
    help="DEPRECATED and ignored — semantic chunking replaces manual batching.",
)
def ingest_obsidian(
    user_id, agent_name, vaults, limit, apply_writes, promote, prune, as_json, batch_chars
):
    """Preview or apply semantic ingestion of allow-listed Obsidian vaults.

    Preview-first: with no ``--apply`` this only plans and prints what *would*
    be written, constructing no client and making no network call. ``--apply``
    performs the writes; fact extraction requires a separate explicit
    ``--promote-to-memory-bank``.
    """
    from . import ingestion
    from .config import load_config

    if batch_chars is not None:
        click.echo(
            "warning: --batch-chars is deprecated and ignored; semantic chunking "
            "replaces manual batching.",
            err=True,
        )

    vault_list = [v for v in vaults if v and v.strip()]
    if not vault_list:
        raise click.UsageError(
            "at least one --vault is required (no default vault allowlist); "
            "pass --vault PATH for each vault to ingest."
        )

    if promote and not apply_writes:
        raise click.UsageError("--promote-to-memory-bank requires --apply (it performs writes).")
    if prune and not apply_writes:
        raise click.UsageError("--prune requires --apply (it deactivates sources).")
    if prune and limit is not None:
        raise click.UsageError(
            "--prune cannot be combined with --limit: a limited run sees only part of "
            "the vault and must not deactivate the sources it did not examine."
        )

    cfg = load_config()

    if not apply_writes:
        # Preview: no seam is called, so no client is constructed and no network
        # call is made. State is read through the preview-safe reader only.
        plan = ingestion.plan_obsidian_ingestion(
            vault_list,
            cfg=cfg,
            user_id=user_id,
            agent_name=agent_name,
            state_reader=_dry_run_state_reader,
        )
        if as_json:
            payload = _plan_payload(plan, mode="preview", user_id=user_id, agent_name=agent_name)
            click.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            click.echo(_render_plan(plan, mode="preview", user_id=user_id, agent_name=agent_name))
        return

    # --- apply path: real dependencies built through injectable seams ---------
    state_reader = _build_state_reader(cfg)
    plan = ingestion.plan_obsidian_ingestion(
        vault_list,
        cfg=cfg,
        user_id=user_id,
        agent_name=agent_name,
        state_reader=state_reader,
    )
    limited = limit is not None
    plan = _limit_plan(plan, limit)

    memory_bank_client = None
    memory_bank_name = None
    if promote:
        memory_bank_client, memory_bank_name = _build_memory_bank(cfg)

    report = ingestion.apply_ingestion_plan(
        plan,
        cfg=cfg,
        user_id=user_id,
        agent_name=agent_name,
        embedding_client=_build_embedding_client(cfg),
        insert_chunks=_build_insert_chunks(cfg),
        finalize_source_revision=_build_finalize_source_revision(cfg),
        semantic_gateway=_build_semantic_gateway(cfg),
        memory_bank_client=memory_bank_client,
        promote_to_memory_bank=promote,
        memory_bank_name=memory_bank_name,
    )

    if prune:
        # Deactivate sources absent from the vault, per corpus. Only reachable
        # with an explicit --prune on a non-limited run (guarded above).
        by_corpus: dict[str, list[str]] = {}
        for planned in (*plan.discovered, *plan.skipped):
            by_corpus.setdefault(planned.corpus_id, []).append(planned.source_id)
        for corpus_id, seen in by_corpus.items():
            _deactivate_missing_sources(
                corpus_id,
                seen,
                user_id=user_id,
                agent_name=agent_name,
                prune=True,
                limited=limited,
                cfg=cfg,
            )

    if as_json:
        payload = _report_payload(report, user_id=user_id, agent_name=agent_name)
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        click.echo(_render_report(report, user_id=user_id, agent_name=agent_name))


if __name__ == "__main__":
    main()
