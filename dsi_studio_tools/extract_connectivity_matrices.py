#!/usr/bin/env python3
"""
Deprecated wrapper: forwards to scripts/extract_connectivity_matrices.py

This file remains for backward compatibility. All logic has moved to
scripts/extract_connectivity_matrices.py. Please update any references.
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "scripts" / "extract_connectivity_matrices.py"

    if not target.exists():
        print(
            "❌ Canonical extractor not found at 'scripts/extract_connectivity_matrices.py'",
            file=sys.stderr,
        )
        return 1

    # Respect current Python environment
    python_cmd = sys.executable or "python"
    cmd = [python_cmd, str(target)] + sys.argv[1:]

    # Friendly notice once
    print(
        "ℹ️  Deprecated wrapper: using scripts/extract_connectivity_matrices.py",
        file=sys.stderr,
    )

    # Unbuffered output passthrough
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
