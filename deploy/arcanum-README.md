# Deploying the GCP Secret Manager credential fetch on Arcanum (Windows desktop)

**Status: ✅ migrated (2026-08-28), live-verified — normal exit and forced-kill
both confirmed to clean up correctly.**

Arcanum has no `systemd` (it's Windows), so it uses a genuinely different
mechanism from Caladan/Sietch's systemd drop-in (see `deploy/README.md`):
a **PowerShell launch wrapper** that the Start Menu shortcut invokes instead
of `Hermes.exe` directly.

## The pattern

1. `scripts/arcanum-launch-hermes.ps1` fetches the SA key from Secret Manager
   into `%TEMP%\hermes-gcp-keys\` (process-scoped — narrower ACL than a
   permanent `%LOCALAPPDATA%` file), sets `GOOGLE_APPLICATION_CREDENTIALS`
   for its own process, launches `Hermes.exe` as a child, waits on it, then
   deletes the key in a `finally` block. The `finally` block runs on normal
   exit **and** on a forced kill of the child (verified — see below).
2. `scripts/arcanum-repoint-shortcut.ps1` edits the existing Start Menu
   `.lnk` (backing up the original first) so its target is
   `powershell.exe -ExecutionPolicy Bypass -File <wrapper>` instead of
   `Hermes.exe` directly. The shortcut's icon is preserved.
3. `scripts/arcanum-install-hermes-memory-into-uv-python.ps1` — see
   **Gotcha #3** below; needed once per Hermes install/update on Arcanum.

## Deploying to a new Windows machine

```powershell
# 1. Ship the wrapper (edit the hardcoded paths inside it first — see below)
scp scripts/arcanum-launch-hermes.ps1 <host>:C:/Users/<user>/hermes-memory-layer/scripts/

# 2. Repoint the shortcut (back up first; the script does this automatically)
scp scripts/arcanum-repoint-shortcut.ps1 <host>:C:/Users/<user>/hermes-memory-layer/scripts/
ssh <host> "powershell -ExecutionPolicy Bypass -File C:\Users\<user>\hermes-memory-layer\scripts\arcanum-repoint-shortcut.ps1"

# 3. Remove the old persistent credential state (only after testing the new
#    path works — see Gotcha #4)
[Environment]::SetEnvironmentVariable("GOOGLE_APPLICATION_CREDENTIALS", $null, "User")
Remove-Item "<old on-disk key path>" -Force
```

**Edit `arcanum-launch-hermes.ps1` before shipping** — three paths are
hardcoded and machine-specific:
- `$hermesExe` — get the real path with `Get-Process Hermes | Select Path`
  while Hermes is running. **Do not guess this path** (see Gotcha #1).
- `$project` / `$secret` — the Secret Manager project + secret name.
- The `.pth`/site-packages paths in the uv-python install script (Gotcha #3).

## Gotchas found during live testing (A4) — read before touching this again

**#1 — `Hermes.exe`'s real path is not what you'd guess.** The obvious guess
(`%LOCALAPPDATA%\Programs\hermes-agent-desktop\Hermes.exe`) was wrong on this
machine; the real path was
`...\AppData\Local\hermes\hermes-agent\apps\desktop\release\win-unpacked\Hermes.exe`.
Always confirm live via `Get-Process Hermes | Select Path` before hardcoding
it into the wrapper — the original shortcut's `.TargetPath` (captured by
`arcanum-repoint-shortcut.ps1` before it overwrites it) is also a reliable
source of truth.

**#2 — Windows PowerShell 5.1's `Out-File -Encoding utf8` prepends a UTF-8
BOM.** This silently breaks `google.auth`'s credential loader
(`google.auth.exceptions.DefaultCredentialsError: ... not a valid json file`,
underlying `json.decoder.JSONDecodeError: Expecting value: line 1 column 0`).
Write the fetched key with `[System.IO.File]::WriteAllText($path, $json, (New-Object System.Text.UTF8Encoding($false)))`
instead — explicit BOM-less UTF-8. The wrapper already does this; if you see
this error again after an edit, check whether that line got reverted to
`Out-File`.

**#3 — Hermes's own backend runs in a *different* Python interpreter than
you'd expect, and it may not have your package.** Hermes's venv
(`hermes-agent\venv\Scripts\python.exe`) is a thin launcher shim; the actual
serving interpreter it re-execs into is a separate uv-managed bare CPython
at `%APPDATA%\uv\python\cpython-3.11-...\python.exe`. That interpreter is
`EXTERNALLY-MANAGED` (PEP 668) and does **not** automatically inherit
whatever you `pip install -e`'d into the venv. Symptom:
`ModuleNotFoundError: No module named 'hermes_memory'` (or `.config`) even
though the same import works fine from the venv's own python.

Fix (one-time per install/update — re-run `arcanum-install-hermes-memory-into-uv-python.ps1`
whenever Hermes updates and creates a fresh uv-managed interpreter):

```powershell
$uvPython = 'C:\Users\<user>\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe'
& $uvPython -m pip install --break-system-packages -e 'C:\Users\<user>\hermes-memory-layer'
```

`--break-system-packages` is safe here specifically because we're installing
our *own* already-vetted editable package, not an arbitrary third-party one.

Electron's own backend spawn (`buildDesktopBackendEnv` in
`apps/desktop/electron/backend-env.ts`) already puts the venv's
`Lib\site-packages` on the child's `PYTHONPATH`, so this only bites tooling
that spawns the uv interpreter directly (SSH diagnostics, `hermes memory
status` run manually) — but it's cheap to fix once and forget it.

**#4 — There are (at least) three places a stale credential path can hide,
and Hermes's own `.env` loader `override=True`s two of them on every
startup.** All three had the OLD on-disk key path and had to be found and
cleared independently before the new tmpfs-scoped path actually took effect
end-to-end:
1. `~/.hermes/.env` → `GOOGLE_APPLICATION_CREDENTIALS=...` (loaded with
   `override=True` at every Hermes startup — this **overwrites** whatever
   the launch wrapper set, so leaving this line in place silently defeats
   the entire migration even though the wrapper "worked").
2. A persistent Windows **User**-scope environment variable
   (`[Environment]::GetEnvironmentVariable('GOOGLE_APPLICATION_CREDENTIALS','User')`)
   — separate from `.env`, inherited by every new process on the machine.
3. The actual on-disk key file itself (`%LOCALAPPDATA%\hermes\gcp-keys\...json`).

**Verification order that actually proves the migration works** (don't trust
a single check — each of the above can mask the others):
1. `Get-Process Hermes.exe` shows it running, `%TEMP%\hermes-gcp-keys\...`
   exists while it runs.
2. Run `hermes memory status` (via the venv's own python, with the *live
   process's actual captured environment* — a static/offline check of
   `is_available()` can pass even when the real launch path is broken by
   #4, because it doesn't reproduce the real env). Confirm `Status:
   available ✓`, not just "no error".
3. Close Hermes normally → confirm the key file is gone within a few
   seconds.
4. Force-kill `Hermes.exe` (`Stop-Process -Force`) while it's running →
   confirm the key file is **still** cleaned up (proves the wrapper's
   `finally` fires on a killed child, not just a graceful exit) and no
   orphaned child processes are left behind.

## Rollback

The original shortcut is backed up automatically by
`arcanum-repoint-shortcut.ps1` as `Hermes.lnk.bak-<timestamp>` next to the
live one. To roll back: delete the current `Hermes.lnk` and rename the
backup back to `Hermes.lnk` — Hermes goes back to launching directly with
whatever credential state was live before the migration.
