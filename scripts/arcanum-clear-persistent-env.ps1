# A5 (pulled forward): remove the persistent Windows User environment
# variable GOOGLE_APPLICATION_CREDENTIALS that points at the old on-disk
# key. This is a separate mechanism from ~/.hermes/.env (already cleaned)
# and from our wrapper's process-scoped $env: assignment — Electron/Hermes
# inherits this at process creation time, so a stale persistent value here
# can still leak into Hermes's environment via any code path that doesn't
# strictly go through our wrapper (interactive shells, ssh sessions, or any
# spawn that reads process.env before the wrapper's override lands).

$ErrorActionPreference = "Stop"

$before = [Environment]::GetEnvironmentVariable('GOOGLE_APPLICATION_CREDENTIALS', 'User')
Write-Output "Before (User scope): $before"

[Environment]::SetEnvironmentVariable('GOOGLE_APPLICATION_CREDENTIALS', $null, 'User')

$after = [Environment]::GetEnvironmentVariable('GOOGLE_APPLICATION_CREDENTIALS', 'User')
Write-Output "After (User scope): $after"

if ($after) {
    Write-Error "Failed to clear the User-scope env var."
    exit 1
}
Write-Output "Cleared successfully."
