# Implementation Spec — Sietch & Arcanum Secret Manager Credential Migration

Status: DRAFT — awaiting approval to execute
Author: Hermes (agent), session `20260827_183626_4a0e21`
Depends on: Task #2 (Caladan), already complete and pushed (`00a199d`, `0457dbb`)
Scope: bring Sietch and Arcanum onto the same "no persistent on-disk SA key,
fetched from Secret Manager at process start" pattern already proven durable
on Caladan.

---

## 0. Non-negotiable constraints carried into this spec

- Nothing stays local-only — every script/config produced here gets committed
  and pushed to `github.com/TojotheTerror/hermes-memory-layer`.
- No sudo password is ever requested or piped through chat. Any step needing
  Sietch's sudo is called out explicitly as **USER-RUN**.
- No OAuth consent screen is driven by the agent. The one-time
  `gcloud auth login --no-launch-browser` step is **USER-RUN**.
- Any bounded sensitive operation (service restart, credential swap) that
  could disrupt a live gateway gets explicit go-ahead before executing, even
  though the user has pre-authorized "continue until done" for routine work.
- Smallest honest useful version — no new third-party drivers (ImDisk),
  no Task Scheduler, no WSL bridging. Baseline hardening only; fancier
  options are logged as optional future work, not required for "done."

---

## 1. SIETCH — WHAT, HOW, WHERE, WHEN

### 1.1 Current confirmed state (from investigation)
- OS: Ubuntu, `apt`/`snap`/`curl` all present, good connectivity.
- `hermes-gateway.service.d/gcp-memory.conf` **already exists** and is
  structurally correct — currently sets:
  ```
  GOOGLE_APPLICATION_CREDENTIALS=/home/tojotheterror/.hermes/gcp-keys/hermes-memory-agent.json
  ```
  (an on-disk key, the thing we're migrating away from), plus the four
  non-secret `GOOGLE_CLOUD_*`/`BQ_*` env vars (keep those as-is — no change
  needed).
- No `google-cloud-sdk` directory present yet (`gcloud` not installed).
- `sudo -n true` fails — **no passwordless sudo**.
- `loginctl show-user … Linger` = **yes**, and
  `systemctl --user is-enabled hermes-gateway.service` = **enabled** — i.e.
  the user service is already set up to survive logout/reboot correctly;
  no linger/enablement work needed, one less thing to break.

### 1.2 Step-by-step plan

**Step S1 — Install gcloud CLI (no sudo required). USER-RUN or agent-run, either works.**
Because the tar.gz method needs no root (confirmed against Google's own
install docs), the agent *can* run this directly over SSH — flagging it here
as agent-run by default unless you'd rather do it yourself:
```bash
ssh sietch "curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz \
  && tar -xf google-cloud-cli-linux-x86_64.tar.gz -C /home/tojotheterror \
  && rm google-cloud-cli-linux-x86_64.tar.gz \
  && /home/tojotheterror/google-cloud-sdk/install.sh --path-update false --command-completion false --usage-reporting false --quiet"
```
Lands at `/home/tojotheterror/google-cloud-sdk/bin/gcloud`, mirroring
Caladan's exact layout so `fetch-sa-key.sh`'s existing fallback path
(`command -v gcloud || echo /home/tojotheterror/google-cloud-sdk/bin/gcloud`)
works unmodified on Sietch too — **zero script changes needed**, just
copy the file as-is.
`--path-update false` avoids touching `.bashrc` non-interactively (SSH
non-login shells don't need it; the script hardcodes the fallback path).

**Step S2 — Authenticate. USER-RUN, mandatory, cannot be delegated.**
```bash
ssh sietch "/home/tojotheterror/google-cloud-sdk/bin/gcloud auth login --no-launch-browser"
```
This prints a URL. You open it on your phone/laptop, sign in as
`rpgmonkey@gmail.com` (the identity that already holds
`roles/secretmanager.secretAccessor` on the secret — confirmed via Caladan's
setup), copy the short authorization code, paste it back into the SSH
session. Agent will provide the exact command but will not read or handle
the resulting code/URL — this runs in a session you drive interactively.

**Step S3 — Verify accessor role resolves for this identity (agent-run, read-only).**
```bash
ssh sietch "/home/tojotheterror/google-cloud-sdk/bin/gcloud secrets versions access latest \
  --secret=hermes-memory-agent-sa-key --project=gen-lang-client-0810135629 | head -c 50"
```
Confirms IAM + auth actually work end-to-end before touching the live
service. Expect valid JSON key material start (`{"type": "service_account"`);
truncated output only, never printed in full to avoid any credential
material landing in terminal scrollback/logs.

**Step S4 — Ship `fetch-sa-key.sh` to Sietch (agent-run).**
```bash
scp ~/hermes-memory-layer/scripts/fetch-sa-key.sh sietch:~/hermes-memory-layer-scripts-fetch-sa-key.sh
ssh sietch "mkdir -p ~/hermes-memory-layer/scripts && mv ~/hermes-memory-layer-scripts-fetch-sa-key.sh ~/hermes-memory-layer/scripts/fetch-sa-key.sh && chmod +x ~/hermes-memory-layer/scripts/fetch-sa-key.sh"
```
(Sietch doesn't need the full repo cloned for this — a minimal
`~/hermes-memory-layer/scripts/` mirror is enough; cloning the actual repo
is a nice-to-have, not required. Noting as future cleanup, not blocking.)

**Step S5 — Edit the existing drop-in (agent-run, single targeted edit).**
Replace just the `GOOGLE_APPLICATION_CREDENTIALS` line and add
`ExecStartPre`, leaving the other four `Environment=` lines untouched:
```ini
[Service]
Environment="GOOGLE_CLOUD_PROJECT=gen-lang-client-0810135629"
Environment="GOOGLE_CLOUD_LOCATION=us-central1"
Environment="GOOGLE_CLOUD_AGENT_ENGINE_ID=8113170407277723648"
Environment="BQ_LOCATION=US"
Environment="BQ_DATASET=hermes_memory"
ExecStartPre=/home/tojotheterror/hermes-memory-layer/scripts/fetch-sa-key.sh
Environment="GOOGLE_APPLICATION_CREDENTIALS=/dev/shm/hermes-gcp-keys/hermes-memory-agent.json"
```
Applied via `patch`-style edit on the live file, not a wholesale overwrite,
to avoid accidentally dropping a line we didn't intend to touch.

**Step S6 — Reload + restart (BOUNDED SENSITIVE OP — needs your explicit go-ahead before execution, per your stated constraint on live-service restarts).**
```bash
ssh sietch "systemctl --user daemon-reload && systemctl --user restart hermes-gateway"
```

**Step S7 — Verify, same rigor as Caladan (multi-cycle, not single-shot trust).**
```bash
ssh sietch "systemctl --user cat hermes-gateway.service"   # confirm drop-in merged
ssh sietch "GWPID=\$(pgrep -f 'hermes_cli.main gateway run' | head -1) && tr '\\0' '\\n' < /proc/\$GWPID/environ | grep GOOGLE_APPLICATION_CREDENTIALS"
ssh sietch "hermes memory status"   # expect: Provider gcp_memory_bank, Status available
```
Repeat the restart+verify cycle **twice more** (3 total, same bar as
Caladan) before declaring durable — specifically checking against the
*current* PID each time to avoid the transient-race false-negative
already seen once on Caladan.

**Step S8 — Delete the old on-disk key (only after S7 passes 3x).**
```bash
ssh sietch "rm -f ~/.hermes/gcp-keys/hermes-memory-agent.json"
```

**Step S9 — Commit + push.**
- Update `deploy/README.md` to remove Sietch from "not yet done" and fold
  it into the main durable-pattern section (it needs no new doc — it now
  follows the exact same steps as Caladan).
- `git add scripts/ deploy/` (if `fetch-sa-key.sh` moved, it's already
  tracked — no change) — most likely just a `deploy/README.md` edit.

### 1.3 When
Sequenced: S1 → S2 (blocks everything after it, USER-RUN) → S3 → S4 → S5 →
[approval gate] → S6 → S7 (x3) → S8 → S9. Estimated agent-active time once
you complete S2: ~10 minutes. S2 itself is on your schedule.

---

## 2. ARCANUM — WHAT, HOW, WHERE, WHEN

### 2.1 Current confirmed state (from investigation, corrected this segment)
- `gcloud` already installed and authenticated; `secretAccessor` already
  granted — no auth work needed here at all.
- Current on-disk key: `C:\Users\rpgmo\AppData\Local\hermes\gcp-keys\hermes-memory-agent.json`,
  referenced via a **User** environment variable
  `GOOGLE_APPLICATION_CREDENTIALS` (confirmed via
  `[Environment]::GetEnvironmentVariable(...,'User')`).
- Launch shortcut confirmed at exact path:
  `C:\Users\rpgmo\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Hermes.lnk`
  (this is what the user actually clicks — Start Menu entry, not a Desktop
  icon; no taskbar pin found).
- No service/scheduled task wrapper exists; Hermes.exe launches as a normal
  child of `explorer.exe` under interactive logon.
- WSL2 (`Ubuntu-24.04`) present with real systemd — evaluated and **rejected**
  as the mechanism (adds a cross-boundary hop for no benefit since Hermes
  itself is a native Windows process, not WSL-hosted).
- BitLocker on for `C:` — softens but doesn't eliminate the on-disk-key
  concern; not sufficient alone, still migrating to fetch-at-launch +
  delete-on-exit.

### 2.2 Design decision
Replace the direct `Hermes.exe` launch with a **wrapper script** that:
1. Fetches the SA key from Secret Manager into `%TEMP%` (process-scoped,
   not `%LOCALAPPDATA%` — narrower default ACL, cleared more aggressively
   by Windows than a persistent AppData folder).
2. Sets `GOOGLE_APPLICATION_CREDENTIALS` **for its own process only** (not
   the persistent User env var — this was the actual latent risk in the
   current setup: a machine-wide persistent env var pointing at a
   permanent secret file is broader exposure than a process-scoped one).
3. Launches `Hermes.exe` as a child (inherits the env var).
4. Waits for Hermes to exit, then deletes the temp key file in a
   `finally` block — runs even on Ctrl+C/terminating errors, per
   PowerShell's documented `try/finally` semantics.
5. The Start Menu shortcut is repointed at the wrapper `.ps1` (via a
   thin `.cmd` launcher, since `.lnk` targets a `.ps1` awkwardly with
   execution-policy prompts otherwise — see S-A2 below).

This needs **no elevation, no Task Scheduler, no service wrapper** — it's a
drop-in replacement for "double-click the icon," which is exactly today's
usage pattern. Smallest useful version, per your stated preference.

### 2.3 Step-by-step plan

**Step A1 — Write the wrapper script (agent-run, local then shipped).**
`scripts/arcanum-launch-hermes.ps1`:
```powershell
$ErrorActionPreference = "Stop"
$project = "gen-lang-client-0810135629"
$secret  = "hermes-memory-agent-sa-key"
$tempDir = Join-Path $env:TEMP "hermes-gcp-keys"
$keyPath = Join-Path $tempDir "hermes-memory-agent.json"
$hermesExe = "$env:LOCALAPPDATA\Programs\hermes-agent-desktop\Hermes.exe"  # confirm exact path before final commit — see Edge Case A-1

New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

try {
    & gcloud secrets versions access latest --secret=$secret --project=$project `
        | Out-File -FilePath $keyPath -Encoding utf8 -NoNewline
    $env:GOOGLE_APPLICATION_CREDENTIALS = $keyPath

    $proc = Start-Process -FilePath $hermesExe -PassThru
    Wait-Process -Id $proc.Id
}
finally {
    if (Test-Path $keyPath) {
        Remove-Item -Path $keyPath -Force -ErrorAction SilentlyContinue
    }
}
```

**Step A2 — Repoint the launch shortcut (agent-run via PowerShell over SSH).**
`.lnk` files launching `.ps1` directly trigger execution-policy prompts by
default. Standard, low-friction fix: keep the `.lnk`, but change its
**Target** to:
```
powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "C:\Users\rpgmo\hermes-memory-layer\scripts\arcanum-launch-hermes.ps1"
```
`-ExecutionPolicy Bypass` is scoped to this single invocation only (doesn't
change the system-wide policy) — this is Microsoft's documented pattern for
exactly this situation, not a security loosening beyond the one launch.

**Step A3 — Ship the script + edit the shortcut.**
```powershell
scp ~/hermes-memory-layer/scripts/arcanum-launch-hermes.ps1 arcanum:C:/Users/rpgmo/hermes-memory-layer/scripts/arcanum-launch-hermes.ps1
```
Shortcut edit via COM `WScript.Shell` (`$sh.CreateShortcut(...).TargetPath`)
run through the standalone-`.ps1`-over-SSH pattern already established to
avoid the nested-quoting bug.

**Step A4 — Test launch (BOUNDED — first live test of a changed launch path, flag before running).**
Manually trigger the shortcut (or run the wrapper directly over SSH for a
non-interactive smoke test first, then confirm interactively) and verify:
- Hermes.exe starts normally, gateway reports `gcp_memory_bank` available.
- Key file appears in `%TEMP%\hermes-gcp-keys\` while running.
- Key file is **gone** within a few seconds of closing Hermes normally.
- Key file is **also gone** after a forced-kill test (`Stop-Process`), to
  confirm the `finally` block's terminating-error coverage claim actually
  holds in practice, not just in theory.

**Step A5 — Remove the persistent User env var + old on-disk key (only after A4 passes both sub-tests).**
```powershell
[Environment]::SetEnvironmentVariable("GOOGLE_APPLICATION_CREDENTIALS", $null, "User")
Remove-Item "C:\Users\rpgmo\AppData\Local\hermes\gcp-keys\hermes-memory-agent.json" -Force
```

**Step A6 — Commit + push** `scripts/arcanum-launch-hermes.ps1` and a new
`deploy/arcanum-README.md` documenting the wrapper-launch pattern (parallel
to the systemd-drop-in README, since Arcanum's mechanism is genuinely
different and deserves its own doc rather than a forced-fit into the
systemd doc).

### 2.4 When
A1 → A2/A3 (can run same session, no auth blocker) → [approval gate before
A4] → A4 (both sub-tests) → A5 → A6. Estimated total: ~15–20 minutes
including the two launch tests. No USER-RUN step exists here at all —
unlike Sietch, this entire path is agent-executable, since gcloud is
already authenticated. Only the *first live launch test* (A4) needs your
go-ahead, as a bounded sensitive change to how your daily-driver Hermes
instance starts.

---

## 3. Edge cases considered

| # | Edge case | Handling |
|---|---|---|
| S-1 | Sietch's SSH session dies mid `gcloud auth login --no-launch-browser` before code paste | Non-destructive — no partial credential state written; just re-run S2 |
| S-2 | `hermes-gateway.service` has `Restart=always` and crash-loops between S6 and a completed S7 verification | `fetch-sa-key.sh` is idempotent (safe to call repeatedly) and fails loudly (`set -euo pipefail`) if the secret fetch fails, which surfaces as a clear systemd start failure rather than a silent bad state |
| S-3 | Secret Manager API transient error during `ExecStartPre` | Same failure-loud behavior as S-2; service simply fails to start, visible via `systemctl --user status`; no automatic silent fallback to a stale key exists (by design — matches Caladan's proven behavior) |
| S-4 | Sietch's `gcloud` install directory collides with a future manual install | Using the identical path convention as Caladan (`~/google-cloud-sdk`) makes this a non-issue and keeps fleet-wide consistency |
| A-1 | Exact `Hermes.exe` path in the wrapper script is a placeholder pending confirmation | **Must verify exact path via `Get-Process Hermes | Select Path` while it's running, before A2 is finalized** — flagged explicitly, not guessed |
| A-2 | User double-clicks the *old* pinned taskbar/search shortcut instead of Start Menu after migration | Confirmed no taskbar pin exists; Start Menu is the only entry point found — low risk, but worth a manual visual check post-migration |
| A-3 | `Wait-Process` never returns because Hermes.exe detaches/respawns itself (e.g. installer self-update relaunch) | Needs verification during A4 — if Hermes replaces its own PID on update, `Wait-Process` would return early and delete the key while the *new* process is still running. Flagging as a real risk to test for, not asserting it's safe |
| A-4 | Antivirus/Windows Defender flags the wrapper's `-ExecutionPolicy Bypass` invocation | Scoped bypass on a user-authored local script is standard and low-risk, but noting as a possible false-positive source if seen during A4 |
| A-5 | Multiple simultaneous Hermes launches (double-click twice) both fetch and both try to delete the same temp file | `Remove-Item -ErrorAction SilentlyContinue` avoids a crash on double-delete; worst case is a harmless race, not a security issue since both instances have legitimate access |
| Cross | Fleet drift — Caladan/Sietch/Arcanum diverge again later because someone edits one node's mechanism without updating the others | Mitigated by keeping all three documented in the same `deploy/` folder with per-platform READMEs cross-referencing each other; **recommend a periodic drift-check** (see Lessons Learned) as genuine future work, not solved by this spec alone |

---

## 4. Review — lessons learned (carried forward from Task #2, applied here)

1. **Never trust "the process is running correctly" as proof of durable
   config.** Caladan's self-heal bug was caught only because the *file on
   disk* was re-checked, not just the live running process's env. Applied
   here: Sietch verification explicitly re-checks the *merged systemd unit*
   and repeats across 3 restart cycles, not one.
2. **A "successful" first grep/check right after a restart can be a false
   negative from a race condition**, not a real failure — re-check against
   the *current* PID before concluding something's broken. Applied to
   Sietch's S7.
3. **Platform-specific self-heal/regeneration logic can silently undo
   direct edits** — this was true of Hermes' systemd unit generator on
   Linux; there is no evidence of an equivalent regenerator for the Start
   Menu shortcut or `.ps1` script on Windows, but this should be explicitly
   watched for during any future Hermes desktop auto-update, since an
   updater *could* plausibly reset shortcuts. Not currently a known risk,
   flagged as a thing to watch, not a certainty.
4. **Nested-quoting bugs in inline PowerShell-over-SSH are a recurring,
   known failure mode** — every Arcanum step in this spec that touches
   PowerShell is written as a standalone `.ps1` file shipped via `scp`,
   per the pattern already fixed twice this project.
5. **Sudo/OAuth interactivity boundaries must be respected even when it
   slows things down** — Sietch's plan explicitly carves out S2 as
   USER-RUN rather than trying to find a workaround (e.g., scripting
   expect-style password entry), consistent with your explicit
   instruction and general security hygiene.
6. **"Smallest useful version" prevented over-engineering** — ImDisk RAM
   disks and Task Scheduler/WSL bridging were seriously evaluated and
   explicitly rejected in favor of `%TEMP%` + `try/finally` and a plain
   wrapper script, matching your stated dislike of enterprise-style
   solutions for a homelab-scale risk.
7. **Documentation must be written for the "next mistake" a future rollout
   would make**, not just as a record — `deploy/README.md` was written
   specifically to stop a *third* node from repeating Caladan's base-unit
   mistake, and the new `deploy/arcanum-README.md` should do the same for
   any future Windows node.

---

## 5. Open items requiring your decision before execution

1. Sietch S1 (gcloud install) — do you want the agent to run it directly
   over SSH (no sudo needed, so agent-run is safe), or would you prefer to
   run it yourself? Defaulting to **agent-run** unless you say otherwise.
2. Sietch S6 and Arcanum A4 are flagged as bounded sensitive ops per your
   standing constraint — I will pause and ask for explicit go-ahead
   immediately before each, even though general execution is
   pre-authorized.
3. Arcanum A-1 (exact `Hermes.exe` path) needs a live confirmation step
   before A2 is finalized — trivial, but sequenced first in Arcanum's
   execution to avoid writing a wrapper against a guessed path.

---

## 6. Definition of done

- [ ] Sietch: gateway running with `GOOGLE_APPLICATION_CREDENTIALS` pointed
      at a `/dev/shm` tmpfs path fetched via `ExecStartPre`, verified
      durable across 3 restart cycles, old on-disk key deleted, changes
      committed + pushed.
- [ ] Arcanum: Hermes launched via wrapper script that fetches the key to
      `%TEMP%`, sets a process-scoped env var, and deletes the key on exit
      (verified for both normal exit and forced-kill); persistent User env
      var and old on-disk key removed; changes committed + pushed.
- [ ] `deploy/README.md` updated to reflect Sietch as complete;
      `deploy/arcanum-README.md` added.
- [ ] All three nodes (Caladan, Sietch, Arcanum) now follow a documented,
      consistent "fetch-at-start, no persistent on-disk key" pattern.
