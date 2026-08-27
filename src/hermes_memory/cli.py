"""CLI: hermes-memory init | search | remember | stats | sync"""
from __future__ import annotations

import json
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
    click.echo(f"Project: {cfg.project}  Location: {cfg.location}  BQ: {cfg.bq_location}  Dataset: {cfg.bq_dataset}")
    from .bigquery_store import ensure_dataset, ensure_tables
    ensure_dataset(cfg)
    ensure_tables(cfg)
    click.echo("Init done. Next: create a Memory Bank with python -c \"from hermes_memory.memory_bank import create_memory_bank; print(create_memory_bank())\"")


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
    click.echo(f"{len(new_facts)} new (not already in corpus), {len(facts) - len(new_facts)} already present")

    if dry_run:
        for f in new_facts:
            click.echo(f"  [dry-run] would write ({f['kind']} / {f['source_file']}): {f['fact'][:80]}...")
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

    click.echo(f"\nSeeded {written}/{len(new_facts)} new facts for scope user_id={user_id}, agent_name={agent_name}")


@main.command("ingest-obsidian")
@click.option("--user", "user_id", required=True, help="user_id scope")
@click.option("--agent", "agent_name", default="hermes", help="agent_name scope")
@click.option("--vault", "vaults", multiple=True, help="Vault path(s); repeatable. Default = canonical set")
@click.option("--min-chars", default=200, type=int, help="Skip notes shorter than this after stripping frontmatter")
@click.option("--batch-chars", default=6000, type=int, help="Approx chars per Memory Bank generate() call")
@click.option("--limit", default=None, type=int, help="Cap number of new/changed notes processed (testing)")
@click.option("--dry-run", is_flag=True, help="Preview batches without writing")
def ingest_obsidian(user_id, agent_name, vaults, min_chars, batch_chars, limit, dry_run):
    """Ingest Obsidian vault notes into Memory Bank via real Gemini extraction (deduped by content hash)."""
    from .hermes_bridge import (
        DEFAULT_OBSIDIAN_VAULTS,
        batch_notes,
        discover_obsidian_notes,
        _load_obsidian_manifest,
        _save_obsidian_manifest,
    )
    from .memory_bank import generate_from_contents

    vault_list = list(vaults) if vaults else DEFAULT_OBSIDIAN_VAULTS
    click.echo(f"Scanning vaults:\n  " + "\n  ".join(vault_list))
    notes = discover_obsidian_notes(vault_list, min_chars=min_chars)
    click.echo(f"Found {len(notes)} substantive notes (>= {min_chars} chars, config/trash dirs excluded)")

    manifest = _load_obsidian_manifest()
    new_notes = [n for n in notes if manifest.get(n["path"]) != n["hash"]]
    click.echo(f"{len(new_notes)} new/changed, {len(notes) - len(new_notes)} already ingested (unchanged)")

    if limit:
        new_notes = new_notes[:limit]
        click.echo(f"Limiting this run to first {limit} notes")

    batches = batch_notes(new_notes, batch_chars=batch_chars)
    click.echo(f"Grouped into {len(batches)} batch(es) (~{batch_chars} chars each -> ~{len(batches)} Memory Bank generate calls)")

    if dry_run:
        for i, b in enumerate(batches):
            click.echo(f"  [dry-run] batch {i + 1}/{len(batches)}: {len(b)} notes -> {[n['rel'] for n in b]}")
        return

    bridge = HermesBridge()
    scope = {"user_id": user_id, "agent_name": agent_name}
    ingested_notes = 0
    failed_batches = 0
    for i, b in enumerate(batches):
        texts = [f"## {n['rel']} (vault: {__import__('pathlib').Path(n['vault']).name})\n\n{n['body']}" for n in b]
        try:
            generate_from_contents(bridge.memory_bank_name, texts, scope, cfg=bridge.cfg)
            for n in b:
                manifest[n["path"]] = n["hash"]
            ingested_notes += len(b)
            click.echo(f"  [OK] batch {i + 1}/{len(batches)}: {len(b)} notes")
        except Exception as e:
            failed_batches += 1
            click.echo(f"  [FAIL] batch {i + 1}/{len(batches)}: {e}")
        _save_obsidian_manifest(manifest)  # save progress after every batch so interrupts don't lose state

    click.echo(
        f"\nIngested {ingested_notes}/{len(new_notes)} notes across {len(batches) - failed_batches}/{len(batches)} "
        f"successful batches for scope user_id={user_id}, agent_name={agent_name}"
    )


if __name__ == "__main__":
    main()
