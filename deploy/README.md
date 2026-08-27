# Deploying the Secret Manager credential fetch (Caladan / Sietch — systemd hosts)

Hermes' own gateway CLI (`hermes_cli.gateway.refresh_systemd_unit_if_needed`)
self-heals `~/.config/systemd/user/hermes-gateway.service` back to its
generated template whenever it detects drift. **Do not edit that file
directly** — the edit will be silently reverted on the next gateway
start/reconcile.

Instead, install a systemd drop-in, which Hermes does not touch:

```bash
mkdir -p ~/.config/systemd/user/hermes-gateway.service.d
cp hermes-gateway-10-gcp-credentials.conf.example \
   ~/.config/systemd/user/hermes-gateway.service.d/10-gcp-credentials.conf
# edit the path inside if fetch-sa-key.sh lives somewhere else on this host
systemctl --user daemon-reload
systemctl --user restart hermes-gateway
```

Verify:

```bash
systemctl --user cat hermes-gateway.service   # confirm drop-in is merged in
GWPID=$(pgrep -f "hermes_cli.main gateway run" | head -1)
tr '\0' '\n' < /proc/$GWPID/environ | grep GOOGLE_APPLICATION_CREDENTIALS
hermes memory status   # Provider: gcp_memory_bank, Status: available
```

Requires `gcloud` CLI on the host, authenticated as a principal with
`roles/secretmanager.secretAccessor` on the `hermes-memory-agent-sa-key`
secret in project `gen-lang-client-0810135629`.

## Arcanum (Windows — no systemd, no service wrapper)

Arcanum runs the gateway as a desktop-app child process under interactive
logon, not a service. This drop-in approach does not apply as-is; deferred
as separate follow-up work (needs a logon-triggered fetch + a Windows-side
tmpfs-equivalent, since `%TEMP%` is disk-backed).
