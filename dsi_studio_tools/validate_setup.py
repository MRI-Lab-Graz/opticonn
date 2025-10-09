#!/usr/bin/env python3
"""
Deprecated wrapper: forwards to scripts/validate_setup.py

Keep this for backward compatibility.
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "scripts" / "validate_setup.py"

    if not target.exists():
        print("❌ scripts/validate_setup.py not found", file=sys.stderr)
        return 1

    python_cmd = sys.executable or "python"
    cmd = [python_cmd, str(target)] + sys.argv[1:]
    print("ℹ️  Deprecated wrapper: using scripts/validate_setup.py", file=sys.stderr)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
