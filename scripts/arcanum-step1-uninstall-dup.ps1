$venvpy = "C:\Users\rpgmo\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe"

Write-Output '=== BEFORE: venv hermes_memory location ==='
$env:PYTHONPATH = ''
& $venvpy -c "import importlib.util as u; s=u.find_spec('hermes_memory'); print(s.origin if s else 'NOT FOUND')" 2>&1

Write-Output ''
Write-Output '=== pip uninstall hermes-memory-layer (venv only) ==='
& $venvpy -m pip uninstall -y hermes-memory-layer 2>&1

Write-Output ''
Write-Output '=== AFTER: venv site-packages hermes_memory present? ==='
$venvhm = "C:\Users\rpgmo\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages\hermes_memory"
Write-Output ("venv hermes_memory dir exists: " + (Test-Path $venvhm))

Write-Output ''
Write-Output '=== AFTER: with wrapper PYTHONPATH, does uv interp now resolve to SRC? (collision gone?) ==='
$uv = (Get-ChildItem 'C:\Users\rpgmo\AppData\Roaming\uv\python' -Directory | Where-Object { $_.Name -like 'cpython-3.11*' } | Select-Object -First 1).FullName
$uvpy = Join-Path $uv 'python.exe'
$env:PYTHONPATH = "C:\Users\rpgmo\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages"
& $uvpy -c "import hermes_memory as h; print('bound:', h.__file__); import hermes_memory.config as c; print('config OK ->', c.__file__)" 2>&1

Write-Output ''
Write-Output '=== AFTER: uv editable install intact? (pip show) ==='
$env:PYTHONPATH = ''
& $uvpy -m pip show hermes-memory-layer 2>&1 | Select-String 'Name|Version|Editable'
