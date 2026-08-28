# Arcanum launch wrapper for Hermes Agent Desktop.
#
# Replaces a direct double-click launch of Hermes.exe. Fetches the GCP
# service-account key from Secret Manager into a process-scoped temp file
# instead of relying on a permanent on-disk key + persistent User env var,
# then deletes the key when Hermes exits (normal exit OR forced-kill).
#
# Spec ref: docs/sietch-arcanum-implementation-spec.md §2.3 Step A1
# Exact Hermes.exe path confirmed live via `Get-Process Hermes | Select Path`
# on 2026-08-27 (edge case A-1) — do NOT revert to the guessed
# %LOCALAPPDATA%\Programs\hermes-agent-desktop\Hermes.exe path from the spec draft.

$ErrorActionPreference = "Stop"

$project   = "gen-lang-client-0810135629"
$secret    = "hermes-memory-agent-sa-key"
$tempDir   = Join-Path $env:TEMP "hermes-gcp-keys"
$keyPath   = Join-Path $tempDir "hermes-memory-agent.json"
$hermesExe = "C:\Users\rpgmo\AppData\Local\hermes\hermes-agent\apps\desktop\release\win-unpacked\Hermes.exe"

if (-not (Test-Path $hermesExe)) {
    Write-Error "Hermes.exe not found at expected path: $hermesExe"
    exit 1
}

# Arcanum's Hermes install re-execs its backend into a separate uv-managed
# bare CPython interpreter (AppData\Roaming\uv\python\cpython-3.11-...)
# that does NOT reliably inherit the venv's site-packages (found during A4
# live testing: `import yaml` / `import hermes_memory` failed there without
# this). Setting PYTHONPATH here guarantees the venv's full dependency set
# (including hermes_memory, installed via `pip install -e` for this exact
# reason) is importable no matter which interpreter actually serves.
$env:PYTHONPATH = "C:\Users\rpgmo\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages"

New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

try {
    $keyJson = & gcloud secrets versions access latest --secret=$secret --project=$project
    $keyJson = $keyJson -join "`n"

    if ([string]::IsNullOrWhiteSpace($keyJson)) {
        Write-Error "Failed to fetch SA key from Secret Manager (empty response)."
        exit 1
    }

    # IMPORTANT: Out-File -Encoding utf8 (Windows PowerShell 5.1) always
    # prepends a UTF-8 BOM, which breaks Python's json.load() /
    # google.auth's credential loader ("not a valid json file" /
    # JSONDecodeError: Expecting value: line 1 column 0). Write explicitly
    # BOM-less instead. (Found + fixed during A4 live testing on Arcanum.)
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($keyPath, $keyJson, $utf8NoBom)

    if (-not (Test-Path $keyPath) -or (Get-Item $keyPath).Length -eq 0) {
        Write-Error "Failed to fetch SA key from Secret Manager (empty or missing file)."
        exit 1
    }

    $env:GOOGLE_APPLICATION_CREDENTIALS = $keyPath

    $proc = Start-Process -FilePath $hermesExe -PassThru
    Wait-Process -Id $proc.Id
}
finally {
    # Runs on normal exit, Ctrl+C, and terminating errors (PowerShell
    # try/finally semantics) — but NOT if this wrapper process itself is
    # forcibly killed (e.g. Stop-Process on the wrapper's own PID) before
    # reaching this block. Verified separately in A4 for the Hermes.exe
    # child being force-killed while the wrapper is still alive.
    if (Test-Path $keyPath) {
        Remove-Item -Path $keyPath -Force -ErrorAction SilentlyContinue
    }
}
