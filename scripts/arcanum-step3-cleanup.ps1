$plugdir = "C:\Users\rpgmo\AppData\Local\hermes\plugins"
$active = Join-Path $plugdir 'gcp_memory_bank'

Write-Output '=== remove .bak-* debug leftovers in active plugin ==='
Get-ChildItem $active -Filter '*.bak-*' -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Output ("removing " + $_.Name)
    Remove-Item $_.FullName -Force
}

Write-Output '=== clear stale __pycache__ so hardened __init__ recompiles ==='
$pyc = Join-Path $active '__pycache__'
if (Test-Path $pyc) { Remove-Item $pyc -Recurse -Force; Write-Output 'pycache cleared' }

Write-Output '=== remove stale hardcoded debug trace file ==='
$trace = "C:\Users\rpgmo\hermes-memory-layer\scripts\a4_debug_trace.txt"
if (Test-Path $trace) { Remove-Item $trace -Force; Write-Output 'trace removed' } else { Write-Output 'no trace file' }

Write-Output '=== remove stale OLD-unfinished plugin dir (backed up already) ==='
$old = Join-Path $plugdir 'gcp-memory-bank.OLD-unfinished'
if (Test-Path $old) { Remove-Item $old -Recurse -Force; Write-Output 'OLD dir removed' } else { Write-Output 'no OLD dir' }

Write-Output ''
Write-Output '=== FINAL plugin dir state ==='
Get-ChildItem $active | Select-Object Name, Length | Format-Table -AutoSize | Out-String
Write-Output '=== plugins/ top-level ==='
Get-ChildItem $plugdir -Directory | Select-Object Name | Format-Table -AutoSize | Out-String
