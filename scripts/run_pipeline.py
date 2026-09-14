#!/usr/bin/env python3
"""
OptiConn extraction pipeline.

--step 01   run tractography + connectivity extraction for every subject
--step all  Step 01, then aggregate DSI Studio network measures into one CSV
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from scripts.utils.runtime import (
    configure_stdio,
    no_emoji_enabled,
    prepare_path_for_subprocess,
    propagate_no_emoji,
)

DRY_RUN = False


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _abs(p: str | os.PathLike | None) -> str | None:
    return None if p is None else prepare_path_for_subprocess(p)


def _run(cmd: list[str], live_prefix: str | None = None, env: dict[str, str] | None = None) -> int:
    """Run a subprocess, streaming its output, and return the exit code."""
    print(f"🚀 Running: {' '.join(cmd)}")
    env = {**os.environ, **(env or {})}
    repo = str(repo_root())
    existing = env.get("PYTHONPATH", "")
    if repo not in existing.split(os.pathsep):
        env["PYTHONPATH"] = repo + (os.pathsep + existing if existing else "")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, encoding="utf-8", errors="replace"
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{live_prefix}] {line.rstrip()}" if live_prefix else line.rstrip())
    return proc.wait()


@dataclass
class Paths:
    output: Path
    step01_dir: Path
    agg_csv: Path


def build_paths(output_dir: str) -> Paths:
    base = Path(output_dir)
    step01 = base / "01_connectivity"
    return Paths(base, step01, step01 / "aggregated_network_measures.csv")


def run_step01(data_dir: str, extraction_config: str, paths: Paths, quiet: bool) -> None:
    """Run connectivity extraction for all subjects in data_dir."""
    cmd = [
        sys.executable, "-m", "scripts.extract_connectivity_matrices",
        "--batch", "-i", data_dir, "-o", str(paths.step01_dir), "--config", extraction_config,
    ]
    pilot = os.environ.get("OPTICONN_PILOT_COUNT", "")
    if pilot.isdigit() and int(pilot) > 0:
        cmd += ["--pilot", "--pilot-count", pilot]
    if no_emoji_enabled():
        cmd.append("--no-emoji")
    if quiet:
        cmd.append("--quiet")
    if DRY_RUN:
        print(f"[DRY-RUN] Would run: {' '.join(cmd)}")
        return
    rc = _run(cmd, live_prefix="step01", env=propagate_no_emoji())
    if rc != 0:
        raise SystemExit(f"Step 01 failed with exit code {rc}")


def run_aggregate(paths: Paths) -> None:
    """Aggregate per-subject network_measures CSVs into one table."""
    cmd = [sys.executable, "-m", "scripts.aggregate_network_measures", str(paths.step01_dir), str(paths.agg_csv)]
    if DRY_RUN:
        print(f"[DRY-RUN] Would run: {' '.join(cmd)}")
        return
    rc = _run(cmd, live_prefix="aggregate", env=propagate_no_emoji())
    if rc != 0 or not paths.agg_csv.exists():
        raise SystemExit(f"Aggregation failed (code {rc}); expected {paths.agg_csv}")


def maybe_build_extraction_config_from_cv(cv_config_path: str, out_dir: Path) -> str:
    try:
        data = json.loads(Path(cv_config_path).read_text())
        if isinstance(data, dict):
            atlases = data.get("atlases") or data.get("atlas")
            metrics = data.get("connectivity_values") or data.get("connectivity_metric")
            cfg = {
                "description": "Auto-generated from cross-validated config",
                "atlases": [atlases] if isinstance(atlases, str) else (atlases or []),
                "connectivity_values": [metrics] if isinstance(metrics, str) else (metrics or []),
            }
            out_cfg = out_dir / "extraction_from_cv.json"
            out_cfg.write_text(json.dumps(cfg, indent=2))
            return str(out_cfg)
    except Exception:
        pass
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="OptiConn extraction pipeline")
    ap.add_argument("--step", default="all", choices=["01", "all"],
                    help="01 = extraction only; all = extraction + network-measure aggregation")
    ap.add_argument("-i", "--input", help="Alias for --data-dir")
    ap.add_argument("--data-dir", help="Directory with subject inputs")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--extraction-config", help="Extraction JSON (default: configs/default_sweep.json)")
    ap.add_argument("--cross-validated-config", help="Dict config with atlases/connectivity_values to extract")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-emoji", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    if len(sys.argv) == 1:
        ap.print_help()
        return 0
    args = ap.parse_args()
    configure_stdio(args.no_emoji)

    global DRY_RUN
    DRY_RUN = args.dry_run

    paths = build_paths(args.output)
    paths.step01_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir or args.input
    extraction_cfg = args.extraction_config or str(repo_root() / "configs" / "default_sweep.json")
    if args.cross_validated_config:
        extraction_cfg = maybe_build_extraction_config_from_cv(args.cross_validated_config, paths.output) or extraction_cfg

    t0 = time.time()
    print(f"🧠 OptiConn Pipeline | step={args.step} | output={paths.output}")
    try:
        if not data_dir:
            raise SystemExit("--data-dir (or -i) is required")
        run_step01(_abs(data_dir), _abs(extraction_cfg), paths, args.quiet)
        if args.step == "all":
            run_aggregate(paths)
        print(f"✅ Pipeline completed in {time.time() - t0:.1f}s")
        return 0
    except SystemExit as e:
        if isinstance(e.code, int):
            return e.code
        print(str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
