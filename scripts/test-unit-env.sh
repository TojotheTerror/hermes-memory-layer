#!/usr/bin/env bash
# Simulates the systemd unit's exact PATH/env for testing fetch-sa-key.sh
set -euo pipefail
exec env -i \
  HOME="$HOME" \
  PATH="/home/tojotheterror/.hermes/hermes-agent/venv/bin:/home/tojotheterror/.hermes/hermes-agent/node_modules/.bin:/home/tojotheterror/.hermes/node/bin:/home/tojotheterror/.hermes/node:/home/tojotheterror/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  VIRTUAL_ENV="/home/tojotheterror/.hermes/hermes-agent/venv" \
  HERMES_HOME="/home/tojotheterror/.hermes" \
  /home/tojotheterror/hermes-memory-layer/scripts/fetch-sa-key.sh
