#!/usr/bin/env python3
"""
Run Pipeline (Steps 01–03)
==========================

Lightweight orchestrator for the OptiConn pipeline. It runs:

  Step 01: Connectivity extraction (DSI Studio) in batch mode
           → outputs live under <output>/01_connectivity/

  Step 02: Metric optimization on aggregated network measures
           → outputs under <output>/02_optimization/

  Step 03: Optimal selection (prepare datasets for scientific analysis)
           → outputs under <output>/03_selection/

Usage examples:
  # Full run (01→03)
  python scripts/run_pipeline.py --step all --data-dir /path/to/fz --output studies/run1 \
         --extraction-config configs/braingraph_default_config.json

  # Analysis only (re-run Step 02→03 using existing Step 01 results in <output>)
  python scripts/run_pipeline.py --step analysis --output studies/run1

This script resolves all helper scripts via absolute paths relative to the
repository root, so it is robust to the current working directory and venv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import subprocess

from scripts.utils.runtime import (
    configure_stdio,
    no_emoji_enabled,
    prepare_path_for_subprocess,
    propagate_no_emoji,
)

# Global dry-run flag (set in main from CLI)
DRY_RUN = False


def repo_root() -> Path:
    """Return the repository root directory (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def scripts_dir() -> Path:
    return repo_root() / "scripts"


def _abs(p: str | os.PathLike | None) -> str | None:
    if p is None:
        return None
    return prepare_path_for_subprocess(p)


def _run(
    cmd: list[str],
    cwd: str | None = None,
    live_prefix: str | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Run a subprocess with live stdout folding and return code."""
    print(f"🚀 Running: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
        env=env,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if live_prefix:
            print(f"[{live_prefix}] {line.rstrip()}")
        else:
            print(line.rstrip())
    return proc.wait()


@dataclass
class Paths:
    output: Path
    step01_dir: Path
    step02_dir: Path
    step03_dir: Path
    agg_csv: Path
    optimized_csv: Path


def build_paths(output_dir: str) -> Paths:
    base = Path(output_dir)
    step01 = base / "01_connectivity"
    step02 = base / "02_optimization"
    step03 = base / "03_selection"
    agg_csv = step01 / "aggregated_network_measures.csv"
    optimized_csv = step02 / "optimized_metrics.csv"
    return Paths(base, step01, step02, step03, agg_csv, optimized_csv)


def ensure_dirs(paths: Paths):
    paths.output.mkdir(parents=True, exist_ok=True)
    paths.step01_dir.mkdir(parents=True, exist_ok=True)
    paths.step02_dir.mkdir(parents=True, exist_ok=True)
    paths.step03_dir.mkdir(parents=True, exist_ok=True)


def run_step01(
    data_dir: str, extraction_config: str, paths: Paths, quiet: bool
) -> None:
    """Run batch connectivity extraction (Step 01)."""
    exe = sys.executable
    # Prefer scripts/extract_connectivity_matrices.py; fallback to repo-root extractor if missing
    primary = scripts_dir() / "extract_connectivity_matrices.py"
    root_extractor = repo_root() / "extract_connectivity_matrices.py"
    script_path = (
        primary
        if primary.exists()
        else (root_extractor if root_extractor.exists() else primary)
    )
    script = str(script_path)
    # Announce the extraction configuration being used (absolute path)
    try:
        cfg_path = Path(extraction_config)
        if cfg_path.exists():
            print(f"⚙️  Extraction config: {cfg_path.resolve()}")
            # Print a concise parameter snapshot if possible
            import json as _json

            with cfg_path.open() as _f:
                _cfg = _json.load(_f)
            _tp = _cfg.get("tracking_parameters") or {}
            _co = _cfg.get("connectivity_options") or {}
            _tc = _cfg.get("tract_count", None)
            snapshot = {
                "tract_count": _tc,
                "connectivity_threshold": _co.get("connectivity_threshold"),
                "otsu_threshold": _tp.get("otsu_threshold"),
                "fa_threshold": _tp.get("fa_threshold"),
                "min_length": _tp.get("min_length"),
                "max_length": _tp.get("max_length"),
                "track_voxel_ratio": _tp.get("track_voxel_ratio"),
            }
            # Compact one-liner (skip None values)
            items = [f"{k}={v}" for k, v in snapshot.items() if v is not None]
            if items:
                print("   ↳ " + ", ".join(items))
    except Exception:
        # Non-fatal; continue
        pass
    cmd = [
        exe,
        script,
        "--batch",
        "-i",
        data_dir,
        "-o",
        str(paths.step01_dir),
        "--config",
        extraction_config,
    ]
    if no_emoji_enabled():
        cmd.append("--no-emoji")
    if quiet:
        cmd.append("--quiet")
    # If we are using fallback, make it visible once
    if script_path == root_extractor and not primary.exists():
        print(f"ℹ️  Using fallback extractor at: {root_extractor}")
    if DRY_RUN:
        print(f"[DRY-RUN] Would run: {' '.join(cmd)}")
        return
    rc = _run(cmd, live_prefix="step01", env=propagate_no_emoji())
    if rc != 0:
        raise SystemExit(f"Step 01 failed with exit code {rc}")


def run_aggregate(paths: Paths) -> None:
    """Aggregate network_measures into a single CSV for optimization."""
    exe = sys.executable
    script = str(scripts_dir() / "aggregate_network_measures.py")
    cmd = [exe, script, str(paths.step01_dir), str(paths.agg_csv)]
    if DRY_RUN:
        print(f"[DRY-RUN] Would run: {' '.join(cmd)}")
        return
    rc = _run(cmd, live_prefix="aggregate", env=propagate_no_emoji())
    if rc != 0 or not paths.agg_csv.exists():
        raise SystemExit(f"Aggregation failed (code {rc}); expected {paths.agg_csv}")


def run_step02(paths: Paths, quiet: bool) -> None:
    """Run metric optimization (Step 02)."""
    exe = sys.executable
    script = str(scripts_dir() / "metric_optimizer.py")
    cmd = [exe, script, "-i", str(paths.agg_csv), "-o", str(paths.step02_dir)]
    if no_emoji_enabled():
        cmd.append("--no-emoji")
    if DRY_RUN:
        print(f"[DRY-RUN] Would run: {' '.join(cmd)}")
        return
    rc = _run(cmd, live_prefix="step02", env=propagate_no_emoji())
    if rc != 0 or not (paths.step02_dir / "optimized_metrics.csv").exists():
        raise SystemExit(
            f"Step 02 failed (code {rc}); expected optimized_metrics.csv in {paths.step02_dir}"
        )


def run_step03(paths: Paths, quiet: bool) -> None:
    """Run optimal selection (Step 03)."""
    exe = sys.executable
    script = str(scripts_dir() / "optimal_selection.py")
    cmd = [exe, script, "-i", str(paths.optimized_csv), "-o", str(paths.step03_dir)]
    if no_emoji_enabled():
        cmd.append("--no-emoji")
    if DRY_RUN:
        print(f"[DRY-RUN] Would run: {' '.join(cmd)}")
        return
    rc = _run(cmd, live_prefix="step03", env=propagate_no_emoji())
    if rc != 0:
        raise SystemExit(f"Step 03 failed with code {rc}")


def maybe_build_extraction_config_from_cv(cv_config_path: str, out_dir: Path) -> str:
    """If a cross-validated dict is provided, derive a minimal extraction config.

    This allows: python scripts/run_pipeline.py --cross-validated-config cv.json --step all
    """
    try:
        data = json.loads(Path(cv_config_path).read_text())
        # Expect a dict with atlases/connectivity_values, otherwise no-op
        if isinstance(data, dict):
            atlases = data.get("atlases") or data.get("atlas")
            metrics = data.get("connectivity_values") or data.get("connectivity_metric")
            if isinstance(atlases, str):
                atlases = [atlases]
            if isinstance(metrics, str):
                metrics = [metrics]
            cfg = {
                "description": "Auto-generated from cross-validated config",
                "atlases": atlases or [],
                "connectivity_values": metrics or [],
            }
            out_cfg = out_dir / "extraction_from_cv.json"
            out_cfg.write_text(json.dumps(cfg, indent=2))
            return str(out_cfg)
    except Exception:
        pass
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description="OptiConn Pipeline Orchestrator (Steps 01–03)"
    )
    ap.add_argument(
        "--step",
        default="all",
        choices=["01", "02", "03", "all", "analysis"],
        help="Which step(s) to run",
    )
    ap.add_argument(
        "-i",
        "--input",
        help="Alias for --data-dir (Step 01 input directory) or explicit input for single steps",
    )
    ap.add_argument(
        "--data-dir", help="Input data directory with .fz/.fib.gz files (Step 01)"
    )
    ap.add_argument(
        "--output", required=True, help="Base output directory for pipeline results"
    )
    ap.add_argument(
        "--extraction-config",
        help="JSON extraction config for Step 01 (default: configs/braingraph_default_config.json)",
    )
    ap.add_argument(
        "--cross-validated-config",
        help="Optional cross-validated config; will be converted to extraction-config if step includes 01",
    )
    ap.add_argument(
        "--quiet", action="store_true", help="Reduce console output where supported"
    )
    ap.add_argument(
        "--no-emoji", action="store_true", help="Disable emoji in console output"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run: print actions without executing external commands",
    )
    args = ap.parse_args()

    # When invoked with no args, print help per project conventions
    if len(sys.argv) == 1:
        ap.print_help()
        sys.exit(0)

    configure_stdio(args.no_emoji)

    # Propagate dry-run setting to module-level flag
    global DRY_RUN
    DRY_RUN = bool(getattr(args, "dry_run", False))

    root = repo_root()
    paths = build_paths(args.output)
    ensure_dirs(paths)

    # Normalize data-dir from -i if provided
    if not args.data_dir and args.input:
        args.data_dir = args.input

    # Default extraction config if missing
    extraction_cfg = args.extraction_config
    if not extraction_cfg:
        extraction_cfg = str(root / "configs" / "braingraph_default_config.json")

    # If we got a cross-validated config and step includes 01, create a minimal extraction config from it
    if args.cross_validated_config and args.step in ("01", "all"):
        derived = maybe_build_extraction_config_from_cv(
            args.cross_validated_config, paths.output
        )
        if derived:
            extraction_cfg = derived

    t0 = time.time()
    print("==================================================")
    print(f"🧠 OptiConn Pipeline | step={args.step} | output={paths.output}")
    print("==================================================")

    try:
        if args.step in ("01", "all"):
            if not args.data_dir:
                raise SystemExit("--data-dir (or -i) is required for Step 01")
            run_step01(_abs(args.data_dir), _abs(extraction_cfg), paths, args.quiet)

        if args.step in ("01", "all", "analysis", "02", "03"):
            # Ensure aggregated CSV exists for downstream steps
            if (
                args.step in ("all", "analysis", "02", "03")
                and not paths.agg_csv.exists()
            ):
                run_aggregate(paths)

        if args.step in ("02", "all", "analysis"):
            run_step02(paths, args.quiet)

        if args.step in ("03", "all", "analysis"):
            # Ensure optimized CSV exists
            if not paths.optimized_csv.exists():
                # If user ran only 03 and provided explicit input file via -i, respect it
                if args.step == "03" and args.input and Path(args.input).exists():
                    paths.optimized_csv = Path(args.input)
                else:
                    raise SystemExit(
                        f"optimized_metrics.csv not found at {paths.optimized_csv}"
                    )
            run_step03(paths, args.quiet)

        print("✅ Pipeline completed successfully!")
        print(f"⏱️  Elapsed: {time.time() - t0:.1f}s")
        return 0

    except SystemExit as e:
        # Re-raise explicit exit with message shown above
        if isinstance(e.code, int):
            return e.code
        print(str(e))
        return 1
    except Exception as e:
        print(f"❌ Pipeline crashed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
