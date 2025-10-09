#!/usr/bin/env python3
"""OptiConn CLI entrypoint.

This launcher ensures the repository's local virtual environment is used by
default. If a venv is found (e.g., ``braingraph_pipeline/`` or ``.venv``),
the script re-execs itself with that interpreter before handing off to the
central CLI hub in ``scripts/opticonn_hub.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _candidate_venvs(root: Path) -> list[Path]:
    return [
        root / "braingraph_pipeline",  # project-named venv (observed in this repo)
        root / ".venv",
        root / "venv",
    ]


def _venv_python_path(venv_dir: Path) -> Path:
    # POSIX vs Windows
    posix = venv_dir / "bin" / "python"
    win = venv_dir / "Scripts" / "python.exe"
    return posix if posix.exists() else win


def _in_that_venv(target_venv: Path) -> bool:
    # Heuristic: VIRTUAL_ENV matches or sys.prefix located under target venv
    ve = os.environ.get("VIRTUAL_ENV")
    if ve and Path(ve).resolve() == target_venv.resolve():
        return True
    try:
        return target_venv.resolve() in Path(sys.prefix).resolve().parents
    except Exception:
        return False


def _bootstrap_venv_if_available() -> None:
    if os.environ.get("OPTICONN_SKIP_VENV", "0") in ("1", "true", "yes"):
        return
    root = _repo_root()
    for venv in _candidate_venvs(root):
        py = _venv_python_path(venv)
        if py.exists():
            # Avoid infinite loop: if we already run with this interpreter, skip
            if Path(sys.executable).resolve() == py.resolve() or _in_that_venv(venv):
                return
            # Re-exec current script with the venv interpreter
            os.execv(str(py), [str(py), __file__, *sys.argv[1:]])
    # No venv found: continue with current interpreter


def main() -> int:
    from scripts.opticonn_hub import main as hub_main

    return hub_main()


if __name__ == "__main__":
    _bootstrap_venv_if_available()
    raise SystemExit(main())
