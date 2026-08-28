# Deploying the Secret Manager credential fetch (Caladan / Sietch — systemd hosts)

**Status: Caladan ✅ migrated. Sietch ✅ migrated (2026-08-27, verified
durable across 3 restart cycles).** Both hosts follow the exact same
pattern below — no host-specific doc needed.

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

> Note: the drop-in *filename* just needs to live under
> `hermes-gateway.service.d/` and end in `.conf` — systemd merges every
> file in that directory regardless of name. Caladan uses
> `10-gcp-credentials.conf`; Sietch's pre-existing drop-in was named
> `gcp-memory.conf` and was edited in place (targeted patch — only the
> `GOOGLE_APPLICATION_CREDENTIALS` line changed, plus one new
> `ExecStartPre`) rather than replaced, since it already had the other
> `GOOGLE_CLOUD_*`/`BQ_*` env vars set correctly. Match your host's
> existing drop-in name; don't invent a second one.

Verify:

```bash
systemctl --user cat hermes-gateway.service   # confirm drop-in is merged in
GWPID=$(pgrep -f "hermes_cli.main gateway run" | head -1)
tr '\0' '\n' < /proc/$GWPID/environ | grep GOOGLE_APPLICATION_CREDENTIALS
hermes memory status   # Provider: gcp_memory_bank, Status: available
```

Repeat the restart + verify cycle **3 times** before deleting the old
on-disk key — always re-derive `$GWPID` fresh each time rather than
reusing a stale PID, to avoid a false-negative from checking a process
that already exited (see Lessons Learned below).

Requires `gcloud` CLI on the host, authenticated as a principal with
`roles/secretmanager.secretAccessor` on the `hermes-memory-agent-sa-key`
secret in project `gen-lang-client-0810135629`. On hosts without root
(e.g. Sietch, no passwordless sudo), install `gcloud` with the no-sudo
tar.gz method:

```bash
curl -fsSL -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz -C "$HOME"
"$HOME"/google-cloud-sdk/install.sh --path-update false --command-completion false --usage-reporting false --quiet
"$HOME"/google-cloud-sdk/bin/gcloud auth login --no-launch-browser   # USER-RUN, OAuth — cannot be scripted/delegated
```

## Lessons learned (carried forward from Caladan → applied on Sietch)

1. Never edit `hermes-gateway.service` directly — Hermes' self-heal
   silently reverts it. Drop-ins are the only durable extension point.
2. When re-verifying after a restart, always re-derive the gateway's
   PID fresh (`pgrep` again) rather than reusing a PID from a prior
   check — checking a stale/exited PID's `/proc/<pid>/environ` gives a
   false negative, not a true failure.
3. Prefer a targeted patch to an existing, structurally-correct drop-in
   over a wholesale overwrite, when the file already has other env vars
   set correctly for the host.
4. Only delete the old on-disk credential file after the new path has
   been proven durable across multiple restart cycles, not after a
   single successful start.

## Arcanum (Windows — no systemd, no service wrapper)

Arcanum runs the gateway as a desktop-app child process under interactive
logon, not a service. This drop-in approach does not apply as-is; deferred
as separate follow-up work (needs a logon-triggered fetch + a Windows-side
tmpfs-equivalent, since `%TEMP%` is disk-backed).
