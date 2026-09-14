#!/usr/bin/env python3
"""
Comprehensive single-run optimizer (vendored)
=============================================

This simplified optimizer runs a single comprehensive optimization
wave (no cross-validation waves). It is a cleaned and self-contained
version used by the opticonn vendored scripts.

It largely follows the upstream flow: generate a single comprehensive
wave config (if not provided), run step01 -> aggregate -> step02 for
each sweep combination, pick the best, and produce selection results
under <output>/optimize.
"""

import json
import math
import os
import sys
import subprocess
import argparse
import logging
import time
from pathlib import Path
import pandas as pd
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from scripts.utils.runtime import configure_stdio
from scripts.reliability import collect_matrices, loo_top1_frequency, rank, score_combo
from scripts.sweep_utils import (
    build_param_grid_from_config,
    grid_product,
    random_sampling as sweep_random_sampling,
    apply_param_choice_to_config,
)


def setup_logging(output_dir: Optional[str] = None):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    handlers = [console_handler]
    if output_dir:
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_handler = logging.FileHandler(
                str(Path(output_dir) / f"cross_validation_{timestamp}.log"),
                encoding="utf-8",
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            handlers.append(file_handler)
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, handlers=handlers)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def generate_single_wave_config(
    data_dir, output_dir, n_subjects: int = 5, extraction_cfg: Optional[str] = None
):
    configs_dir = Path(output_dir) / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    if not extraction_cfg:
        extraction_cfg = "configs/default_sweep.json"

    wave_config = {
        "test_config": {
            "name": "comprehensive_optimization",
            "description": "Single comprehensive wave for parameter optimization",
        },
        "data_selection": {
            "source_dir": str(data_dir),
            "selection_method": "random",
            "n_subjects": int(n_subjects),
            "random_seed": 42,
            "file_pattern": "*.fz",
        },
        "pipeline_config": {
            "steps_to_run": ["01", "02", "03"],
            "extraction_config": extraction_cfg,
        },
        "bootstrap": {"n_iterations": 5, "sample_ratio": 0.8},
    }

    wave_path = configs_dir / "comprehensive_wave.json"
    with open(wave_path, "w") as f:
        json.dump(wave_config, f, indent=2)

    logging.info(
        f"📝 Generated single comprehensive wave configuration in {configs_dir}"
    )
    return str(wave_path)


def load_wave_config(config_file):
    with open(config_file, "r") as f:
        return json.load(f)


def to_phase2_candidate(row: dict) -> dict:
    """Shape a ranked row the way `opticonn_hub analyze` reads candidates."""
    cfg = json.loads(Path(row["config_path"]).read_text())
    return {
        "atlas": row["atlas"],
        "connectivity_metric": row["connectivity_metric"],
        "average_score": row["discriminability"],
        "repeatability": row["repeatability"],
        "loo_top1_frequency": row.get("loo_top1_frequency"),
        "backend": cfg.get("backend", "dsi_studio"),
        "parameters": {
            "tract_count": cfg.get("tract_count"),
            "tracking_parameters": cfg.get("tracking_parameters", {}),
            "connectivity_threshold": (cfg.get("connectivity_options") or {}).get("connectivity_threshold"),
            "connectivity_options": cfg.get("connectivity_options") or {},
        },
    }


def json_safe(obj):
    """Recursively replace non-finite floats (NaN/inf) with None so `json.dumps` never emits bare NaN."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def filter_requested_metrics(rows: list[dict], cfg: dict) -> list[dict]:
    """Keep only rows scoring a metric in `cfg["connectivity_values"]`; keep all if unset."""
    requested = cfg.get("connectivity_values")
    if not requested:
        return rows
    return [r for r in rows if r["connectivity_metric"] in requested]


def run_wave_pipeline(
    wave_config_file,
    output_base_dir,
    max_parallel: int = 1,
    prune_nonbest: bool = False,
):
    logging.info(f"🚀 Running pipeline for {wave_config_file}")
    wave_config = load_wave_config(wave_config_file)
    wave_name = wave_config["test_config"]["name"]

    logging.info(f"📋 Wave configuration loaded: {wave_name}")

    wave_output_dir = Path(output_base_dir) / wave_name
    wave_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"📁 Created wave output directory: {wave_output_dir}")

    # list available files
    try:
        src_dir = Path(wave_config["data_selection"]["source_dir"])
        patterns = [wave_config["data_selection"].get("file_pattern", "*.fz")]
        files = []
        for pat in patterns:
            files.extend(sorted([p for p in src_dir.rglob(pat)]))
        files.extend(sorted([p for p in src_dir.rglob("*.fib.gz")]))
        seen = set()
        uniq = []
        for p in files:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        available_manifest = wave_output_dir / "available_files.txt"
        with available_manifest.open("w") as mf:
            for p in uniq:
                mf.write(str(p) + "\n")
        logging.info(f"📄 Available files listed: {available_manifest} ({len(uniq)})")
    except Exception as e:
        logging.warning(f"⚠️  Could not list available files: {e}")

    n_subjects = int(wave_config["data_selection"].get("n_subjects") or 3)
    seed = int(wave_config["data_selection"].get("random_seed") or 42)
    fz_files = [p for p in uniq if str(p).endswith(".fz")]
    fib_files = [p for p in uniq if str(p).endswith(".fib.gz")]
    pool = fz_files + fib_files
    if not pool:
        logging.error("❌ No candidate files found for selection")
        return False
    if n_subjects >= len(pool):
        selected = pool
    else:
        selected = random.Random(seed).sample(pool, n_subjects)

    selected_manifest = wave_output_dir / "selected_files.txt"
    with selected_manifest.open("w") as sf:
        for p in selected:
            sf.write(str(p) + "\n")
    logging.info(f"📄 Selected files listed: {selected_manifest} ({len(selected)})")

    staging_dir = wave_output_dir / "selected_data"
    staging_dir.mkdir(exist_ok=True)
    for p in selected:
        dest = staging_dir / p.name
        try:
            if not dest.exists():
                dest.symlink_to(p)
        except OSError:
            shutil.copy2(p, dest)
    logging.info(f"📂 Staging data directory: {staging_dir}")

    root = repo_root()
    extraction_cfg_rel = wave_config.get("pipeline_config", {}).get(
        "extraction_config", "configs/default_sweep.json"
    )
    extraction_cfg = (
        str((root / extraction_cfg_rel).resolve())
        if not Path(extraction_cfg_rel).is_absolute()
        else extraction_cfg_rel
    )
    logging.info(f"🔧 Wave '{wave_name}' using extraction config: {extraction_cfg}")

    try:
        with open(extraction_cfg, "r") as _f:
            base_cfg = json.load(_f)
    except Exception as e:
        logging.error(f"❌ Failed to load extraction config {extraction_cfg}: {e}")
        return False

    sp = base_cfg.get("sweep_parameters") or {}
    param_values, mapping = build_param_grid_from_config({"sweep_parameters": sp})
    sampling = (sp.get("sampling") or {}) if isinstance(sp, dict) else {}
    method = (sampling.get("method") or "grid").lower()
    n_samples = int(sampling.get("n_samples") or 0)
    seed = int(sampling.get("random_seed") or 42)
    if method == "grid" or not param_values:
        combos = grid_product(param_values) if param_values else [{}]
    elif method == "random":
        if n_samples <= 0:
            n_samples = 24
        combos = sweep_random_sampling(param_values, n_samples, seed)
    else:
        raise ValueError(f"Unknown sampling method '{method}' (use 'grid' or 'random')")

    sweep_cfg_dir = wave_output_dir / "configs" / "sweep"
    sweep_cfg_dir.mkdir(parents=True, exist_ok=True)
    combos_dir = wave_output_dir / "combos"
    combos_dir.mkdir(parents=True, exist_ok=True)

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

    def fmt_choice(c: dict) -> str:
        items = []
        used = set()
        for k in preferred_order:
            if k in c:
                items.append(f"{k}={c[k]}")
                used.add(k)
        for k in sorted([k for k in c.keys() if k not in used]):
            items.append(f"{k}={c[k]}")
        return ", ".join(items)

    logging.info(
        f"⏳ Starting parameter sweep for {wave_name}: {len(combos)} combination(s) [method={method}, max_parallel={max_parallel}]"
    )

    reliability_cfg = base_cfg.get("reliability") or {}
    repeats = int(reliability_cfg.get("repeats", 2))
    base_threads = int(base_cfg.get("thread_count") or 8)
    adj_threads = max(1, base_threads // max(1, int(max_parallel)))

    tasks = []
    for i, choice in enumerate(combos, 1):
        derived = apply_param_choice_to_config(base_cfg, choice, mapping)
        derived["sweep_meta"] = {
            "index": i,
            "choice": choice,
            "sampler": method,
            "total_combinations": len(combos),
            "source_config": extraction_cfg,
        }
        derived["thread_count"] = adj_threads
        cfg_path = sweep_cfg_dir / f"sweep_{i:04d}.json"
        cfg_path.write_text(json.dumps(derived, indent=2))
        combo_out = combos_dir / f"sweep_{i:04d}"
        combo_out.mkdir(parents=True, exist_ok=True)
        logging.info(f"🔎 Parameters [{i}/{len(combos)}]: {fmt_choice(choice)} | repeats={repeats}")
        tasks.append((cfg_path, combo_out))

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = str(repo_root()) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    def run_combo(cfg_path: Path, combo_out: Path):
        base = json.loads(cfg_path.read_text())
        for k in range(1, repeats + 1):
            rep_dir = combo_out / f"rep_{k}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            rep_cfg = json.loads(json.dumps(base))
            rep_cfg.setdefault("tracking_parameters", {})["random_seed"] = k
            rep_cfg_path = rep_dir / "config.json"
            rep_cfg_path.write_text(json.dumps(rep_cfg, indent=2))
            cmd01 = [
                sys.executable, "-m", "scripts.run_pipeline",
                "--data-dir", str(staging_dir),
                "--step", "01",
                "--output", str(rep_dir),
                "--extraction-config", str(rep_cfg_path),
            ]
            p1 = subprocess.run(cmd01, capture_output=True, text=True, env=env)
            if p1.returncode != 0:
                (combo_out / "diagnostics.json").write_text(
                    json.dumps(
                        {
                            "status": "failed",
                            "stage": f"step01_repeat_{k}",
                            "config_path": str(cfg_path),
                            "return_code": p1.returncode,
                            "stdout_tail": (p1.stdout or "")[-4000:],
                            "stderr_tail": (p1.stderr or "")[-4000:],
                        },
                        indent=2,
                    )
                )
                return cfg_path, None, f"step01 repeat {k} failed (rc={p1.returncode}); see {combo_out / 'diagnostics.json'}"
        rows = score_combo(combo_out, reliability_cfg)
        rows = filter_requested_metrics(rows, base_cfg)
        for r in rows:
            r.update(
                sweep_id=combo_out.name,
                config_path=str(cfg_path),
                tract_count=base.get("tract_count"),
                parameters=base["sweep_meta"]["choice"],
            )
        (combo_out / "diagnostics.json").write_text(json.dumps({"status": "ok", "candidates": rows}, indent=2))
        return cfg_path, rows, "ok"

    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, int(max_parallel))) as ex:
        future_cfg = {ex.submit(run_combo, c, o): c for c, o in tasks}
        for fut in as_completed(future_cfg):
            cfg_path = future_cfg[fut]
            try:
                cfg_path, rows, status = fut.result()
            except Exception as e:
                logging.error(f"❌ [{cfg_path.stem}] crashed: {e}")
                continue
            if rows is None:
                logging.error(f"❌ [{cfg_path.stem}] {status}")
                continue
            if not rows:
                logging.error(f"❌ [{cfg_path.stem}] no connectivity matrices found")
            for r in rows:
                verdict = f"REJECTED ({r['rejected']})" if r["rejected"] else "ok"
                logging.info(
                    f"📊 [{cfg_path.stem}] {r['atlas']}/{r['connectivity_metric']}: "
                    f"discriminability={r['discriminability']:.3f} repeatability={r['repeatability']:.3f} "
                    f"density={r['density']:.3f} → {verdict}"
                )
            all_rows.extend(rows)

    if not all_rows:
        logging.error("❌ No combination produced connectivity matrices")
        return False

    diagnostics_csv = wave_output_dir / "combo_diagnostics.csv"
    pd.DataFrame(
        [{k: json.dumps(v) if isinstance(v, dict) else v for k, v in r.items()} for r in all_rows]
    ).to_csv(diagnostics_csv, index=False)
    logging.info(f"📝 Wrote {diagnostics_csv}")

    ranked = rank(all_rows)
    if not ranked:
        logging.error(
            "❌ Every candidate failed the quality gates; adjust `reliability` gates or sweep ranges (see combo_diagnostics.csv)"
        )
        return False

    def candidate_key(r: dict) -> str:
        return f"{r['sweep_id']}|{r['atlas']}|{r['connectivity_metric']}"

    top = ranked[:10]
    freq = loo_top1_frequency(
        {candidate_key(r): collect_matrices(combos_dir / r["sweep_id"])[(r["atlas"], r["connectivity_metric"])] for r in top},
        {candidate_key(r): r.get("tract_count") for r in top},
    )
    for r in top:
        r["loo_top1_frequency"] = freq[candidate_key(r)]

    results_dir = Path(output_base_dir) / "optimization_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "ranked_candidates.json").write_text(json.dumps(json_safe(ranked), indent=2))
    (results_dir / "top3_candidates.json").write_text(
        json.dumps(json_safe([to_phase2_candidate(r) for r in top[:3]]), indent=2)
    )
    best_cfg = json.loads(Path(top[0]["config_path"]).read_text())
    best_cfg["atlases"] = [top[0]["atlas"]]
    best_cfg["connectivity_values"] = [top[0]["connectivity_metric"]]
    (results_dir / "selected_parameters.json").write_text(json.dumps({"selected_config": best_cfg}, indent=2))
    best_freq = top[0]["loo_top1_frequency"]
    freq_str = "n/a" if best_freq is None or not math.isfinite(best_freq) else f"{best_freq:.0%}"
    logging.info(
        f"🏆 Best: {candidate_key(top[0])} discriminability={top[0]['discriminability']:.3f} "
        f"(top-1 in {freq_str} of leave-one-subject-out rankings)"
    )

    if prune_nonbest:
        keep = {r["sweep_id"] for r in top[:3]}
        for child in combos_dir.iterdir():
            if child.is_dir() and child.name not in keep:
                shutil.rmtree(child, ignore_errors=True)
                logging.info(f"🧹 Pruned {child.name}")

    logging.info(f"✅ Wave {wave_name} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Comprehensive single-run optimizer")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform a dry-run: generate configs and summarize actions without executing the pipeline",
    )
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    parser.add_argument("-i", "--data-dir", required=True, help="Data directory")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory")
    parser.add_argument("--config", help="Master configuration file (optional)")
    parser.add_argument(
        "--extraction-config",
        help="Override extraction config used in auto-generated wave",
    )
    parser.add_argument(
        "--subjects", type=int, default=3, help="Subjects to sample (default: 3)"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Max combinations to run in parallel (default: 1)",
    )
    parser.add_argument(
        "--prune-nonbest",
        action="store_true",
        help="After selection, delete non-best combo outputs to save space",
    )
    parser.add_argument(
        "--no-emoji",
        action="store_true",
        help="Disable emoji in console output (Windows-safe)",
    )

    args = parser.parse_args()

    configure_stdio(args.no_emoji)
    base_output = Path(args.output_dir) / "optimize"
    setup_logging(str(base_output))
    logging.info("🎯 COMPREHENSIVE OPTIMIZER (single-run)")
    logging.info("=" * 50)
    logging.info(f"📂 Input data directory: {args.data_dir}")
    logging.info(f"📁 Output directory: {args.output_dir}")

    output_dir = base_output
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"📁 Created output directory: {output_dir}")

    # Determine config: prefer explicit --config, otherwise auto-generate a single comprehensive wave
    if args.config:
        logging.info("📋 Loading master configuration file")
        with open(args.config, "r") as f:
            master_config = json.load(f)
        wave_config = master_config.get("wave1_config") or master_config.get(
            "comprehensive_wave"
        )
        if not wave_config:
            logging.info(
                "📝 Master config did not contain wave config; generating single comprehensive wave"
            )
            wave_cfg_path = generate_single_wave_config(
                args.data_dir,
                output_dir,
                n_subjects=args.subjects,
                extraction_cfg=args.extraction_config,
            )
        else:
            # Save provided config to file for consistency
            cfg_path = Path(output_dir) / "configs" / "provided_wave.json"
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            with cfg_path.open("w") as wf:
                json.dump(wave_config, wf, indent=2)
            wave_cfg_path = str(cfg_path)
    else:
        wave_cfg_path = generate_single_wave_config(
            args.data_dir,
            output_dir,
            n_subjects=args.subjects,
            extraction_cfg=args.extraction_config,
        )

    logging.info(f"📄 Wave config: {wave_cfg_path}")

    if args.dry_run:
        logging.info(
            "--dry-run requested; config generated and actions summarized above."
        )
        return 0

    start_time = time.time()
    success = run_wave_pipeline(
        wave_cfg_path,
        output_dir,
        max_parallel=args.max_parallel,
        prune_nonbest=args.prune_nonbest,
    )
    total_duration = time.time() - start_time

    if success:
        logging.info("✅ COMPREHENSIVE OPTIMIZATION COMPLETED SUCCESSFULLY")
        logging.info(f"📊 Results saved in: {output_dir}")
        logging.info(f"⏱️  Total runtime: {total_duration:.1f} seconds")
        return 0
    else:
        logging.error("❌ OPTIMIZATION FAILED")
        logging.error(f"⏱️  Runtime before failure: {total_duration:.1f} seconds")
        sys.exit(1)


if __name__ == "__main__":
    main()
