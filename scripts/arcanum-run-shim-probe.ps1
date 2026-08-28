$venvpy = "C:\Users\rpgmo\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe"
$env:PYTHONPATH = "C:\Users\rpgmo\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages"
Write-Output '=== venv shim: import hermes_cli.main only, check hermes_memory side-effect ==='
& $venvpy 'C:\Users\rpgmo\hermes-memory-layer\scripts\arcanum_shim_import_probe.py' 2>&1
Write-Output "exit=$LASTEXITCODE"
