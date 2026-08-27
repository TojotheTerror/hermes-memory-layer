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


if __name__ == "__main__":
    main()
