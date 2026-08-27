#!/usr/bin/env bash
# Fetches the hermes-memory-agent SA key from GCP Secret Manager into
# memory-backed tmpfs (/dev/shm) rather than persistent disk.
#
# Requires: gcloud CLI, authenticated as a principal with
#           roles/secretmanager.secretAccessor on hermes-memory-agent-sa-key
#           (currently: rpgmonkey@gmail.com via `gcloud auth login`).
#
# Prints the tmpfs path to stdout on success. Callers should
# `export GOOGLE_APPLICATION_CREDENTIALS="$(fetch-sa-key.sh)"`.
set -euo pipefail

PROJECT="gen-lang-client-0810135629"
SECRET_NAME="hermes-memory-agent-sa-key"
DEST_DIR="/dev/shm/hermes-gcp-keys"
DEST_FILE="${DEST_DIR}/hermes-memory-agent.json"
GCLOUD_BIN="${GCLOUD_BIN:-$(command -v gcloud || echo /home/tojotheterror/google-cloud-sdk/bin/gcloud)}"

mkdir -p "$DEST_DIR"
chmod 700 "$DEST_DIR"

"$GCLOUD_BIN" secrets versions access latest \
  --secret="$SECRET_NAME" \
  --project="$PROJECT" \
  > "$DEST_FILE"

chmod 600 "$DEST_FILE"

echo "$DEST_FILE"
