#!/usr/bin/env bash
set -euo pipefail
PROJECT_ID="${PROJECT_ID:-${GOOGLE_CLOUD_PROJECT:-gen-lang-client-0810135629}}"
LOCATION="${LOCATION:-us-central1}"
BQ_LOCATION="${BQ_LOCATION:-US}"
DATASET="${BQ_DATASET:-hermes_memory}"

echo "==> Hermes memory layer setup"
echo "    PROJECT_ID=$PROJECT_ID  LOCATION=$LOCATION  BQ_LOCATION=$BQ_LOCATION  DATASET=$DATASET"

echo "==> Enabling APIs (idempotent)..."
gcloud services enable aiplatform.googleapis.com bigquery.googleapis.com --project="$PROJECT_ID"

echo "==> Creating BigQuery dataset $PROJECT_ID.$DATASET ($BQ_LOCATION)..."
bq --project_id="$PROJECT_ID" mk --location="$BQ_LOCATION" --dataset --description "Hermes Agent memory layer" "$PROJECT_ID:$DATASET" 2>&1 | grep -v "Already Exists" || true

echo "==> Creating tables..."
for tbl in memories sessions memory_revisions document_sources document_chunks; do
  echo "  - $tbl"
done
# Use Python helper for DDL (handles exists_ok). This provisions BOTH the
# Memory Bank mirror tables AND the citation-bearing document tables
# (document_sources / document_chunks) that back semantic ingestion. In
# Terraform, the document tables carry deletion_protection + prevent_destroy;
# see docs/semantic-ingestion.md for the rollback procedure.
python3 -c "from hermes_memory.bigquery_store import ensure_dataset, ensure_tables; ensure_dataset(); ensure_tables()" 2>&1 || {
  echo "Python helper not installed — run: pip install -e ."
  echo "Falling back to bq DDL via helper SQL files in bigquery/"
}

echo "==> Done. Next:"
echo "    python -c \"from hermes_memory.memory_bank import create_memory_bank; print(create_memory_bank())\""
echo "    # then export GOOGLE_CLOUD_AGENT_ENGINE_ID=<id>"
echo "    hermes-memory search --user tojo --query 'fleet topology'"
echo "    # Semantic document ingestion (preview-first; see docs/semantic-ingestion.md):"
echo "    hermes-memory ingest-obsidian --user tojo --agent hermes --vault ~/Vaults/Hermes_Agent --limit 5 --json  # dry run"
