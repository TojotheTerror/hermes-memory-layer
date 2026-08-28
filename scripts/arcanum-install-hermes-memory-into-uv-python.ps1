# The uv-managed bare CPython interpreter that Hermes actually re-execs its
# serve subprocess into (separate from the project's venv\Scripts\python.exe
# launcher shim -- confirmed via process tree inspection during A4 testing)
# does not have hermes-memory-layer installed, causing gcp_memory_bank's
# is_available() to fail its `import hermes_memory` check even though the
# main venv has it. uv marks this interpreter EXTERNALLY-MANAGED; using
# --break-system-packages is the documented escape hatch for installing our
# own already-vetted editable package into it.
$uvPython = 'C:\Users\rpgmo\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe'
& $uvPython -m pip install --break-system-packages -e 'C:\Users\rpgmo\hermes-memory-layer' 2>&1
Write-Output '---VERIFY---'
& $uvPython -c "import hermes_memory; print('hermes_memory OK:', hermes_memory.__file__)" 2>&1
Write-Output '---VERIFY-CONFIG---'
& $uvPython -c "from hermes_memory.config import load_config; c = load_config(); print('project:', c.project)" 2>&1
