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
import os
import sys
import subprocess
import argparse
import logging
import time
from pathlib import Path
import pandas as pd
import numpy as np
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from scripts.utils.runtime import configure_stdio
from scripts.sweep_utils import (
    build_param_grid_from_config,
    grid_product,
    random_sampling as sweep_random_sampling,
    lhs_sampling,
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
        extraction_cfg = "configs/braingraph_default_config.json"

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
    random.seed(seed)
    fz_files = [p for p in uniq if str(p).endswith(".fz")]
    fib_files = [p for p in uniq if str(p).endswith(".fib.gz")]
    pool = fz_files + fib_files
    if not pool:
        logging.error("❌ No candidate files found for selection")
        return False
    if n_subjects >= len(pool):
        selected = pool
    else:
        selected = random.sample(pool, n_subjects)

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
        "extraction_config", "configs/braingraph_default_config.json"
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
        if n_samples <= 0:
            n_samples = 24
        combos = lhs_sampling(param_values, n_samples, seed)

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

    optimized_csvs = []
    logging.info(
        f"⏳ Starting parameter sweep for {wave_name}: {len(combos)} combination(s) [method={method}, max_parallel={max_parallel}]"
    )

    base_threads = int(base_cfg.get("thread_count") or 8)
    adj_threads = max(1, base_threads // max(1, int(max_parallel)))

    tasks = []
    for i, choice in enumerate(combos, 1):
        derived = apply_param_choice_to_config(base_cfg, choice, mapping)
        try:
            import datetime as _dt

            derived["sweep_meta"] = {
                "index": i,
                "choice": choice,
                "sampler": method,
                "total_combinations": len(combos),
                "source_config": extraction_cfg,
                "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            }
        except Exception:
            pass
        derived["thread_count"] = adj_threads

        cfg_path = sweep_cfg_dir / f"sweep_{i:04d}.json"
        with cfg_path.open("w") as _out:
            json.dump(derived, _out, indent=2)

        combo_out = combos_dir / f"sweep_{i:04d}"
        combo_out.mkdir(parents=True, exist_ok=True)

        logging.info(
            f"🔎 Parameters [{i}/{len(combos)}]: {fmt_choice(choice)} | thread_count={adj_threads}"
        )
        tasks.append((i, cfg_path, combo_out))

    def run_combo(
        i: int, cfg_path: Path, combo_out: Path
    ) -> tuple[Path, Path, float, int, str, str]:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        # Ensure subprocesses can import the vendored `scripts` package
        repo = str(repo_root())
        existing_pp = env.get("PYTHONPATH", "")
        if repo not in existing_pp.split(os.pathsep):
            env["PYTHONPATH"] = repo + (os.pathsep + existing_pp if existing_pp else "")
        # Step 01: run pipeline extraction/tracking for this combo
        # Run pipeline step01 via module invocation so vendored imports resolve
        cmd01 = [
            sys.executable,
            "-m",
            "scripts.run_pipeline",
            "--data-dir",
            str(staging_dir),
            "--step",
            "01",
            "--output",
            str(combo_out),
            "--extraction-config",
            str(cfg_path),
        ]
        p1 = subprocess.run(cmd01, capture_output=True, text=True, env=env)
        if p1.returncode != 0:
            try:
                fail_diag = {
                    "status": "failed",
                    "stage": "step01",
                    "wave": wave_name,
                    "combo_dir": str(combo_out),
                    "config_path": str(cfg_path),
                    "return_code": p1.returncode,
                    "stdout_tail": p1.stdout[-4000:] if p1.stdout else "",
                    "stderr_tail": p1.stderr[-4000:] if p1.stderr else "",
                }
                (combo_out / "diagnostics.json").write_text(
                    json.dumps(fail_diag, indent=2)
                )
            except Exception:
                pass
            return (
                cfg_path,
                Path(""),
                -1.0,
                -1,
                f"step01_failed: rc={p1.returncode}\n{p1.stdout[-4000:]}\n{p1.stderr[-4000:] if p1.stderr else ''}",
                "",
            )

        # Aggregate measures
        agg_csv = combo_out / "01_connectivity" / "aggregated_network_measures.csv"
        if not agg_csv.exists():
            cmdAgg = [
                sys.executable,
                "-m",
                "scripts.aggregate_network_measures",
                str(combo_out / "01_connectivity"),
                str(agg_csv),
            ]
            pAgg = subprocess.run(
                cmdAgg, capture_output=True, text=True, env=env, cwd=str(repo_root())
            )
            if pAgg.returncode != 0 or not agg_csv.exists():
                try:
                    fail_diag = {
                        "status": "failed",
                        "stage": "aggregate",
                        "wave": wave_name,
                        "combo_dir": str(combo_out),
                        "config_path": str(cfg_path),
                        "return_code": pAgg.returncode,
                        "stdout_tail": pAgg.stdout[-4000:] if pAgg.stdout else "",
                        "stderr_tail": pAgg.stderr[-4000:] if pAgg.stderr else "",
                    }
                    (combo_out / "diagnostics.json").write_text(
                        json.dumps(fail_diag, indent=2)
                    )
                except Exception:
                    pass
                return (
                    cfg_path,
                    Path(""),
                    -1.0,
                    -1,
                    f"aggregate_failed: rc={pAgg.returncode}\n{pAgg.stdout[-4000:]}\n{pAgg.stderr[-4000:] if pAgg.stderr else ''}",
                    "",
                )

        # Step 02: metric optimizer
        step02_dir = combo_out / "02_optimization"
        step02_dir.mkdir(exist_ok=True)
        cmd02 = [
            sys.executable,
            "-m",
            "scripts.metric_optimizer",
            str(agg_csv),
            str(step02_dir),
        ]
        p2 = subprocess.run(
            cmd02, capture_output=True, text=True, env=env, cwd=str(repo_root())
        )
        opt_csv = step02_dir / "optimized_metrics.csv"
        if p2.returncode != 0 or not opt_csv.exists():
            try:
                fail_diag = {
                    "status": "failed",
                    "stage": "step02",
                    "wave": wave_name,
                    "combo_dir": str(combo_out),
                    "config_path": str(cfg_path),
                    "return_code": p2.returncode,
                    "stdout_tail": p2.stdout[-4000:] if p2.stdout else "",
                    "stderr_tail": p2.stderr[-4000:] if p2.stderr else "",
                }
                (combo_out / "diagnostics.json").write_text(
                    json.dumps(fail_diag, indent=2)
                )
            except Exception:
                pass
            return (
                cfg_path,
                Path(""),
                -1.0,
                -1,
                f"step02_failed: rc={p2.returncode}\n{p2.stdout[-4000:]}\n{p2.stderr[-4000:] if p2.stderr else ''}",
                "",
            )

        # Score selection
        try:
            df = pd.read_csv(opt_csv)
            raw_mean = (
                float(df["quality_score_raw"].mean())
                if "quality_score_raw" in df.columns
                else float("nan")
            )
            norm_max = (
                float(df["quality_score"].max())
                if "quality_score" in df.columns
                else float("nan")
            )
            score = 0.0
            score_components: list[dict[str, float]] = []
            if not np.isnan(raw_mean):
                contrib = raw_mean * 1.0
                score += contrib
                score_components.append(
                    {
                        "metric": "quality_score_raw_mean",
                        "value": raw_mean,
                        "weight": 1.0,
                        "contribution": contrib,
                    }
                )
            if not np.isnan(norm_max):
                weight = 0.1
                contrib = norm_max * weight
                score += contrib
                score_components.append(
                    {
                        "metric": "quality_score_norm_max",
                        "value": norm_max,
                        "weight": weight,
                        "contribution": contrib,
                    }
                )
            try:
                with open(cfg_path, "r") as _cf:
                    _cfg_json = json.load(_cf)
                tract_count = int(
                    _cfg_json.get("sweep_parameters", {}).get(
                        "tract_count", _cfg_json.get("tract_count", -1)
                    )
                )
                thread_count = int(_cfg_json.get("thread_count") or -1)
                sweep_meta = _cfg_json.get("sweep_meta") or {}
            except Exception:
                tract_count = -1
                thread_count = -1
                sweep_meta = {}

            dens = float("nan")
            geff = float("nan")
            try:
                agg_csv = (
                    combo_out / "01_connectivity" / "aggregated_network_measures.csv"
                )
                diag_df = pd.read_csv(agg_csv)
                dens = (
                    float(diag_df["density"].mean())
                    if "density" in diag_df.columns
                    else float("nan")
                )
                geff = (
                    float(diag_df["global_efficiency(weighted)"].mean())
                    if "global_efficiency(weighted)" in diag_df.columns
                    else float("nan")
                )
            except Exception:
                pass

            if not np.isnan(dens):
                weight = 0.05
                contrib = dens * weight
                score += contrib
                score_components.append(
                    {
                        "metric": "density_mean",
                        "value": dens,
                        "weight": weight,
                        "contribution": contrib,
                    }
                )
            if not np.isnan(geff):
                weight = 0.05
                contrib = geff * weight
                score += contrib
                score_components.append(
                    {
                        "metric": "global_efficiency_weighted_mean",
                        "value": geff,
                        "weight": weight,
                        "contribution": contrib,
                    }
                )

            tract_penalty = 0.0
            if tract_count and tract_count > 0:
                tract_penalty = tract_count * 1e-10
                score -= tract_penalty

            if not score_components:
                score = -1.0

            try:
                diag_json = {
                    "status": "ok",
                    "wave": wave_name,
                    "combo_dir": str(combo_out),
                    "config_path": str(cfg_path),
                    "combo_index": int(sweep_meta.get("index") or i),
                    "total_combinations": int(
                        sweep_meta.get("total_combinations") or -1
                    ),
                    "sampler": sweep_meta.get("sampler"),
                    "parameters": sweep_meta.get("choice"),
                    "thread_count": thread_count,
                    "tract_count": tract_count,
                    "selection_score": float(score),
                    "selection_score_components": score_components,
                    "tract_count_penalty": tract_penalty if tract_penalty else 0.0,
                    "quality_score_raw_mean": (
                        float(raw_mean) if not np.isnan(raw_mean) else None
                    ),
                    "quality_score_norm_max": (
                        float(norm_max) if not np.isnan(norm_max) else None
                    ),
                    "aggregates": {
                        "density_mean": None if np.isnan(dens) else float(dens),
                        "global_efficiency_weighted_mean": (
                            None if np.isnan(geff) else float(geff)
                        ),
                    },
                    "files": {
                        "optimized_metrics_csv": str(opt_csv),
                        "aggregated_measures_csv": str(agg_csv),
                    },
                }
                (combo_out / "diagnostics.json").write_text(
                    json.dumps(diag_json, indent=2)
                )
            except Exception:
                pass

            extra_bits = []
            if not np.isnan(dens):
                extra_bits.append(f"density_mean={dens:.4f}")
            if tract_penalty:
                extra_bits.append(f"tract_penalty={tract_penalty:.6f}")
            diag = " ".join(extra_bits)
        except Exception as e:
            try:
                fail_diag = {
                    "status": "failed",
                    "stage": "score",
                    "wave": wave_name,
                    "combo_dir": str(combo_out),
                    "config_path": str(cfg_path),
                    "error": str(e),
                }
                (combo_out / "diagnostics.json").write_text(
                    json.dumps(fail_diag, indent=2)
                )
            except Exception:
                pass
            return (cfg_path, opt_csv, -1.0, -1, f"score_error: {e}", "")
        return (cfg_path, opt_csv, score, tract_count, "ok", diag)

    if max_parallel <= 1:
        for i, cfg_path, combo_out in tasks:
            cfg, opt_csv, score, tc, status, diag = run_combo(i, cfg_path, combo_out)
            if status == "ok":
                try:
                    df = pd.read_csv(opt_csv)
                    raw_mean = (
                        float(df["quality_score_raw"].mean())
                        if "quality_score_raw" in df.columns
                        else float("nan")
                    )
                    norm_max = (
                        float(df["quality_score"].max())
                        if "quality_score" in df.columns
                        else float("nan")
                    )
                    extra = f" | {diag}" if diag else ""
                    logging.info(
                        f"✅ [{cfg_path.stem}] raw_mean={raw_mean:.3f} | max quality_score(norm)={norm_max:.3f} | tract_count={tc}{extra}"
                    )
                except Exception:
                    logging.info(
                        f"✅ [{cfg_path.stem}] score={score:.3f} | tract_count={tc}"
                    )
                optimized_csvs.append((cfg, opt_csv, score, tc))
            else:
                logging.error(f"❌ [{cfg_path.stem}] {status}")
    else:
        with ThreadPoolExecutor(max_workers=max_parallel) as ex:
            futs = {
                ex.submit(run_combo, i, cfg_path, combo_out): (i, cfg_path)
                for i, cfg_path, combo_out in tasks
            }
            for fut in as_completed(futs):
                i, cfg_path = futs[fut]
                try:
                    cfg, opt_csv, score, tc, status, diag = fut.result()
                except Exception as e:
                    logging.error(f"❌ [{cfg_path.stem}] exception: {e}")
                    continue
                if status == "ok":
                    try:
                        df = pd.read_csv(opt_csv)
                        raw_mean = (
                            float(df["quality_score_raw"].mean())
                            if "quality_score_raw" in df.columns
                            else float("nan")
                        )
                        norm_max = (
                            float(df["quality_score"].max())
                            if "quality_score" in df.columns
                            else float("nan")
                        )
                        extra = f" | {diag}" if diag else ""
                        logging.info(
                            f"✅ [{cfg_path.stem}] raw_mean={raw_mean:.3f} | max quality_score(norm)={norm_max:.3f} | tract_count={tc}{extra}"
                        )
                    except Exception:
                        logging.info(
                            f"✅ [{cfg_path.stem}] score={score:.3f} | tract_count={tc}"
                        )
                    optimized_csvs.append((cfg, opt_csv, score, tc))
                else:
                    logging.error(f"❌ [{cfg_path.stem}] {status}")

    # aggregate diagnostics
    try:
        import csv as _csv

        diag_rows = []
        param_keys: set[str] = set()
        for child in combos_dir.iterdir():
            if child.is_dir() and child.name.startswith("sweep_"):
                j = child / "diagnostics.json"
                if j.exists():
                    try:
                        rec = json.loads(j.read_text())
                        row = {
                            "wave": rec.get("wave"),
                            "sweep_id": child.name,
                            "status": rec.get("status"),
                            "combo_index": rec.get("combo_index"),
                            "total_combinations": rec.get("total_combinations"),
                            "sampler": rec.get("sampler"),
                            "thread_count": rec.get("thread_count"),
                            "tract_count": rec.get("tract_count"),
                            "selection_score": rec.get("selection_score"),
                            "tract_count_penalty": rec.get("tract_count_penalty"),
                            "selection_score_components": json.dumps(
                                rec.get("selection_score_components", [])
                            ),
                            "quality_score_raw_mean": rec.get("quality_score_raw_mean"),
                            "quality_score_norm_max": rec.get("quality_score_norm_max"),
                            "density_mean": (rec.get("aggregates") or {}).get(
                                "density_mean"
                            ),
                            "global_efficiency_weighted_mean": (
                                rec.get("aggregates") or {}
                            ).get("global_efficiency_weighted_mean"),
                        }
                        params = rec.get("parameters") or {}
                        if isinstance(params, dict):
                            for pk, pv in params.items():
                                col_name = f"param_{pk}"
                                if isinstance(pv, (dict, list)):
                                    pv = json.dumps(pv)
                                row[col_name] = pv
                                param_keys.add(col_name)
                        diag_rows.append(row)
                    except Exception:
                        pass
        if diag_rows:
            out_csv = wave_output_dir / "combo_diagnostics.csv"
            cols = [
                "wave",
                "sweep_id",
                "status",
                "combo_index",
                "total_combinations",
                "sampler",
                "thread_count",
                "tract_count",
                "selection_score",
                "tract_count_penalty",
                "selection_score_components",
                "quality_score_raw_mean",
                "quality_score_norm_max",
                "density_mean",
                "global_efficiency_weighted_mean",
            ]
            cols.extend(sorted(param_keys))
            with out_csv.open("w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in sorted(diag_rows, key=lambda r: (r.get("combo_index") or 0)):
                    w.writerow({k: r.get(k) for k in cols})
            logging.info(f"📝 Wrote wave-level combo diagnostics: {out_csv}")
    except Exception as e:
        logging.warning(f"⚠️  Could not write wave-level diagnostics CSV: {e}")

    if not optimized_csvs:
        logging.error("❌ No successful combinations completed Step 02")
        return False

    # choose best
    best = None
    best_score = -1.0
    best_tc = None
    eps = 1e-4
    for cfg_path, opt_csv, sc, tc in optimized_csvs:
        logging.info(f"📊 {cfg_path.stem}: selection_score={sc:.3f} | tract_count={tc}")
        if (sc > best_score + eps) or (
            abs(sc - best_score) <= eps
            and (best_tc is None or (tc != -1 and tc < best_tc))
        ):
            best_score = sc
            best_tc = tc
            best = (cfg_path, opt_csv)

    if not best:
        logging.error("❌ Could not determine best combination (no scores)")
        return False

    best_cfg, best_opt_csv = best
    logging.info(
        f"🏆 Selected best parameters: {best_cfg.name} (selection_score={best_score:.3f}, tract_count={best_tc})"
    )
    step03_dir = wave_output_dir / "03_selection"
    step03_dir.mkdir(exist_ok=True)
    # Run Step 03 (optimal selection) via module invocation so vendored imports resolve
    cmd03 = [
        sys.executable,
        "-m",
        "scripts.optimal_selection",
        "-i",
        str(best_opt_csv),
        "-o",
        str(step03_dir),
    ]
    logging.debug(f"🔧 Step03 cmd: {' '.join(cmd03)}")
    env_final = os.environ.copy()
    repo = str(repo_root())
    existing_pp = env_final.get("PYTHONPATH", "")
    if repo not in existing_pp.split(os.pathsep):
        env_final["PYTHONPATH"] = repo + (
            os.pathsep + existing_pp if existing_pp else ""
        )
    p3 = subprocess.run(cmd03, capture_output=True, text=True, env=env_final, cwd=repo)
    if p3.returncode != 0:
        logging.error("❌ Step 03 failed for best combination")
        logging.debug(f"Step03 stdout: {p3.stdout[-4000:] if p3.stdout else ''}")
        logging.debug(f"Step03 stderr: {p3.stderr[-4000:] if p3.stderr else ''}")
        return False

    try:
        meta_out = wave_output_dir / "selected_parameters.json"
        with open(best_cfg, "r") as _in, meta_out.open("w") as _out:
            data = json.load(_in)
            json.dump({"selected_config": data}, _out, indent=2)
        logging.info(f"📝 Selected parameters saved to {meta_out}")
    except Exception:
        pass

    if prune_nonbest:
        try:
            for child in combos_dir.iterdir():
                if (
                    child.is_dir()
                    and child.name.startswith("sweep_")
                    and (child / "02_optimization").exists()
                ):
                    if child != best_opt_csv.parent.parent:
                        shutil.rmtree(child, ignore_errors=True)
                        logging.info(f"🧹 Pruned {child.name}")
        except Exception as e:
            logging.warning(f"⚠️  Pruning non-best combos failed: {e}")

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
        # Prepare optimization_results like upstream
        try:
            root = repo_root()
            optimization_results_dir = Path(output_dir) / "optimization_results"
            optimization_results_dir.mkdir(parents=True, exist_ok=True)
            wave1_dir = Path(output_dir) / "comprehensive_optimization"
            selected_params = wave1_dir / "selected_parameters.json"
            if selected_params.exists():
                import shutil as _sh

                _sh.copy2(
                    selected_params,
                    optimization_results_dir / "selected_parameters.json",
                )
            candidates_file = optimization_results_dir / "top3_candidates.json"
            if selected_params.exists():
                with open(selected_params, "r") as f:
                    data = json.load(f)
                selected_config = data.get("selected_config", {})
                candidates = [
                    {
                        "atlas": (
                            selected_config.get("atlases", ["Unknown"])[0]
                            if isinstance(selected_config.get("atlases"), list)
                            else selected_config.get("atlases", "Unknown")
                        ),
                        "connectivity_metric": (
                            selected_config.get("connectivity_values", ["Unknown"])[0]
                            if isinstance(
                                selected_config.get("connectivity_values"), list
                            )
                            else selected_config.get("connectivity_values", "Unknown")
                        ),
                        "average_score": 1.0,
                        "parameters": {
                            "fa_threshold": selected_config.get(
                                "fa_threshold",
                                selected_config.get("tracking_parameters", {}).get(
                                    "fa_threshold", "Unknown"
                                ),
                            ),
                            "min_length": selected_config.get(
                                "min_length",
                                selected_config.get("tracking_parameters", {}).get(
                                    "min_length", "Unknown"
                                ),
                            ),
                            "tract_count": selected_config.get(
                                "tract_count", "Unknown"
                            ),
                            "connectivity_threshold": selected_config.get(
                                "connectivity_threshold",
                                selected_config.get("connectivity_options", {}).get(
                                    "connectivity_threshold", "Unknown"
                                ),
                            ),
                        },
                    }
                ]
                with open(candidates_file, "w") as f:
                    json.dump(candidates, f, indent=2)
                logging.info(
                    f"📄 Generated top3_candidates.json from comprehensive optimization"
                )
        except Exception as e:
            logging.warning(f"⚠️  Failed to prepare optimization results: {e}")
        return 0
    else:
        logging.error("❌ OPTIMIZATION FAILED")
        logging.error(f"⏱️  Runtime before failure: {total_duration:.1f} seconds")
        sys.exit(1)


if __name__ == "__main__":
    main()
