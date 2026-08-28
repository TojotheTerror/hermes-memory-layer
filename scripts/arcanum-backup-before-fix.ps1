$ts = Get-Date -Format 'yyyyMMddHHmmss'
$bk = "C:\Users\rpgmo\hermes-memory-layer\backups\arcanum-fix-$ts"
New-Item -ItemType Directory -Force -Path $bk | Out-Null
Write-Output "backup dir: $bk"

# 1) Record the exact venv install (so we can restore the duplicate if ever needed)
$venvpy = "C:\Users\rpgmo\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe"
& $venvpy -m pip show hermes-memory-layer 2>&1 | Out-File "$bk\venv-pip-show-hermes-memory-layer.txt" -Encoding utf8
Copy-Item "C:\Users\rpgmo\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages\hermes_memory_layer-0.1.0.dist-info\RECORD" "$bk\venv-RECORD.txt" -ErrorAction SilentlyContinue

# 2) Back up the active plugin dir (init + baks + yaml)
$plugin = "C:\Users\rpgmo\AppData\Local\hermes\plugins\gcp_memory_bank"
Copy-Item $plugin "$bk\plugin-gcp_memory_bank" -Recurse -Force -ErrorAction SilentlyContinue

# 3) Back up the launch wrapper
Copy-Item "C:\Users\rpgmo\hermes-memory-layer\scripts\arcanum-launch-hermes.ps1" "$bk\arcanum-launch-hermes.ps1.bak" -ErrorAction SilentlyContinue

# 4) List the stale OLD plugin dir contents (record before deletion in step 3)
$old = "C:\Users\rpgmo\AppData\Local\hermes\plugins\gcp-memory-bank.OLD-unfinished"
if (Test-Path $old) {
    Get-ChildItem $old -Recurse | Select-Object FullName, Length | Out-File "$bk\OLD-plugin-manifest.txt" -Encoding utf8
    Copy-Item $old "$bk\gcp-memory-bank.OLD-unfinished" -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Output '=== backup contents ==='
Get-ChildItem $bk -Recurse | Select-Object FullName, Length | Format-Table -AutoSize | Out-String
