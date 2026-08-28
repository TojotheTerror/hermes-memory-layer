# Standalone script: remove the stale GOOGLE_APPLICATION_CREDENTIALS line
# from ~/.hermes/.env on Arcanum. Hermes loads this file with override=True
# at startup (hermes_cli/env_loader.py), which was clobbering our launch
# wrapper's tmpfs-scoped credential path with the old persistent on-disk
# path — root cause of A4's "provider reports unavailable" failure.
#
# Spec ref: this pulls forward part of A5 (remove persistent env var) since
# it's a hard blocker for A4 passing at all.

$ErrorActionPreference = "Stop"
$envPath = "C:\Users\rpgmo\AppData\Local\hermes\.env"

if (-not (Test-Path $envPath)) {
    Write-Error ".env not found at: $envPath"
    exit 1
}

$backupPath = "$envPath.bak-$(Get-Date -Format yyyyMMddHHmmss)"
Copy-Item -Path $envPath -Destination $backupPath -Force
Write-Output "Backed up .env to: $backupPath"

$lines = Get-Content -Path $envPath
$filtered = $lines | Where-Object { $_ -notmatch '^\s*GOOGLE_APPLICATION_CREDENTIALS\s*=' }

$removedCount = $lines.Count - $filtered.Count
Write-Output "Removed $removedCount line(s) matching GOOGLE_APPLICATION_CREDENTIALS="

# Write back BOM-less UTF-8 (same lesson as the launch wrapper fix).
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($envPath, ($filtered -join "`r`n") + "`r`n", $utf8NoBom)

Write-Output "Done. Remaining GOOGLE_ lines:"
Get-Content -Path $envPath | Select-String -Pattern '^GOOGLE_'
