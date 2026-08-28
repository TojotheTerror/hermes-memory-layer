import sys, importlib
# Import the CLI entry module WITHOUT running main(); report side-effect imports.
m = importlib.import_module("hermes_cli.main")
print("imported hermes_cli.main OK")
print("hermes_memory in sys.modules:", "hermes_memory" in sys.modules)
loaded = [k for k in sys.modules if k.startswith("hermes_memory")]
print("hermes_memory.* loaded:", loaded[:12])
