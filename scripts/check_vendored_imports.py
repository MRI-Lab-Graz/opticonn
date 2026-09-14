#!/usr/bin/env python3
"""Check imports for vendored scripts in this directory.

Usage: python scripts/check_vendored_imports.py

This script will attempt to import each .py file in the same folder using
importlib to avoid affecting sys.path. It reports success/failure for each
file and exits with code 0 if all imports succeeded, else 1.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
SKIP = {Path(__file__).name, "__init__.py"}

files = sorted([p for p in HERE.glob("*.py") if p.name not in SKIP])

fails = []
print(f"Checking {len(files)} vendored script(s) in: {HERE}\n")
for p in files:
    name = p.name
    print(f"-> {name}: ", end="", flush=True)
    try:
        spec = importlib.util.spec_from_file_location(name.rstrip(".py"), str(p))
        if spec is None or spec.loader is None:
            raise ImportError("Could not create spec/loader")
        mod = importlib.util.module_from_spec(spec)
        # Register module in sys.modules before executing to support
        # constructs (like dataclass) that access the module during class creation.
        sys.modules[spec.name] = mod
        try:
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        except SystemExit as se:
            # Some modules may call sys.exit() on import; treat as failure but capture
            raise RuntimeError(f"module exited during import with code {se.code}")
        print("OK")
    except Exception as e:
        print("FAILED")
        print("-" * 72)
        print(f"Error importing {name}:")
        traceback.print_exc()
        print("-" * 72)
        fails.append((name, e))

print("\nSummary:")
if not fails:
    print("All vendored scripts imported successfully ✅")
    sys.exit(0)
else:
    print(f"{len(fails)} script(s) failed to import:")
    for name, err in fails:
        print(f" - {name}: {err}")
    sys.exit(1)
