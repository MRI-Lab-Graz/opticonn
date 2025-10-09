#!/usr/bin/env python3
"""
Run Parameter Sweep
===================

Erzeugt Parameterkombinationen basierend auf configs/braingraph_default_config.json (oder custom)
unter Verwendung von sweep_parameters und sampling. Schreibt pro Kombination eine abgeleitete
Config-Datei und optional eine CSV-Übersicht. Die eigentliche Pipeline-Ausführung kann danach
sequenziell oder parallel erfolgen (hier nur Erzeugung). Optional: sofortige Ausführung für N Beispiele.
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from typing import Any, Dict, List

from sweep_utils import (
    build_param_grid_from_config,
    grid_product,
    random_sampling,
    lhs_sampling,
    apply_param_choice_to_config,
)


def choose_sampler(method: str):
    method = (method or "grid").lower()
    if method == "grid":
        return "grid"
    if method == "random":
        return "random"
    if method == "lhs":
        return "lhs"
    return "grid"


def main():
    ap = argparse.ArgumentParser(description="Generate parameter sweep configurations")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform a safe dry-run: show what would be generated without writing files",
    )
    # Show help when called without arguments to satisfy global instructions
    if len(sys.argv) == 1:
        ap.print_help()
        return 0
    ap.add_argument(
        "--config",
        default="configs/braingraph_default_config.json",
        help="Base config with sweep_parameters",
    )
    ap.add_argument(
        "-i",
        "--data-dir",
        required=False,
        help="Data directory with .fib.gz/.fz files (preferred)",
    )
    ap.add_argument(
        "-o",
        "--output-dir",
        default="sweep_runs",
        help="Output directory for generated configs",
    )
    ap.add_argument(
        "--quick", action="store_true", help="Use quick_sweep ranges if present"
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Run pipeline for generated configs (sequential)",
    )
    ap.add_argument(
        "--max-executions",
        type=int,
        default=0,
        help="Max number of runs to execute immediately (0=all)",
    )
    ap.add_argument(
        "--quiet", action="store_true", help="Reduce console output during execution"
    )
    args = ap.parse_args()

    base_path = Path(args.config)
    if not base_path.exists():
        print(f"❌ Base config not found: {base_path}")
        return 1

    with base_path.open() as f:
        cfg = json.load(f)

    sp = cfg.get("sweep_parameters", {}) or {}
    if args.quick and isinstance(sp.get("quick_sweep"), dict):
        # overlay quick ranges on sweep_parameters
        quick = sp["quick_sweep"]
        merged = dict(sp)
        merged.update(
            {
                k: v
                for k, v in quick.items()
                if k.endswith("_range") or k in ("sweep_tract_count",)
            }
        )
        # preserve sampling from parent if not overridden
        if "sampling" not in merged and "sampling" in sp:
            merged["sampling"] = sp["sampling"]
        sp = merged

    param_values, mapping = build_param_grid_from_config({"sweep_parameters": sp})
    if not param_values:
        print("⚠️  No sweep parameters found. Nothing to do.")
        return 0

    # Decide sampler
    sampling = sp.get("sampling", {}) or {}
    method = choose_sampler(sampling.get("method", "grid"))
    n_samples = int(sampling.get("n_samples", 0) or 0)
    seed = int(sampling.get("random_seed", 42))

    if method == "grid":
        combos = grid_product(param_values)
    elif method == "random":
        if n_samples <= 0:
            n_samples = 24
        combos = random_sampling(param_values, n_samples, seed)
    else:  # lhs
        if n_samples <= 0:
            n_samples = 24
        combos = lhs_sampling(param_values, n_samples, seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save summary CSV
    import csv

    summary_file = out_dir / "sweep_combinations.csv"
    keys = sorted(set(k for c in combos for k in c.keys()))
    with summary_file.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index"] + keys)
        w.writeheader()
        for i, c in enumerate(combos, 1):
            row = {"index": i}
            row.update(c)
            w.writerow(row)
    print(f"📄 Combinations: {summary_file} ({len(combos)} entries)")

    # Generate derived configs
    config_dir = out_dir / "configs"
    config_dir.mkdir(exist_ok=True)

    cfg_files: List[Path] = []
    for i, choice in enumerate(combos, 1):
        derived = apply_param_choice_to_config(cfg, choice, mapping)
        # Attach lightweight sweep metadata for traceability
        try:
            import datetime as _dt

            derived["sweep_meta"] = {
                "index": i,
                "choice": choice,
                "sampler": method,
                "total_combinations": len(combos),
                "source_config": str(base_path),
                "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            }
        except Exception:
            # Best-effort only
            pass
        path = config_dir / f"sweep_{i:04d}.json"
        with path.open("w") as f:
            json.dump(derived, f, indent=2)
        cfg_files.append(path)
    print(f"🧩 Generated {len(cfg_files)} derived config files in {config_dir}")

    if not args.execute:
        return 0

    # Optional: execute sequentially
    max_exec = args.max_executions or len(cfg_files)
    import subprocess

    runs = 0
    total = min(max_exec, len(cfg_files))
    # Preferred key order for compact parameter echo
    preferred_order = [
        "tract_count",
        "connectivity_threshold",
        "otsu_threshold",
        "fa_threshold",
        "min_length",
        "max_length",
        "track_voxel_ratio",
        "turning_angle",
        "step_size",
        "smoothing",
        "dt_threshold",
    ]

    def fmt_choice(c: Dict[str, Any]) -> str:
        # Stable ordering: preferred first, then remaining sorted
        shown = []
        used = set()
        for k in preferred_order:
            if k in c:
                shown.append(f"{k}={c[k]}")
                used.add(k)
        remaining = sorted([k for k in c.keys() if k not in used])
        for k in remaining:
            shown.append(f"{k}={c[k]}")
        return ", ".join(shown)

    for idx, path in enumerate(cfg_files, 1):
        if runs >= max_exec:
            break
        # Read choice back from file meta if available; fallback to summary CSV row
        choice_display = ""
        try:
            with path.open() as f:
                d = json.load(f)
            meta = d.get("sweep_meta") or {}
            choice_display = fmt_choice(meta.get("choice", {}))
        except Exception:
            choice_display = ""
        cmd = [
            "python",
            "run_pipeline.py",
            "--step",
            "all",
            "--data-dir",
            args.data_dir or cfg.get("data_dir", cfg.get("input_dir", ".")),
            "--extraction-config",
            str(path),
            "--output",
            str(out_dir / f"run_{path.stem}"),
        ]
        if args.quiet:
            cmd.append("--quiet")
        # Echo the exact parameter combination for this run (one concise line)
        if choice_display:
            print(f"🔎 Parameters [{runs+1}/{total}]: {choice_display}")
        print(f"🚀 Running [{runs+1}/{total}]: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Run failed ({path}): {e.returncode}")
        runs += 1

    print(f"✅ Executed {runs} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
