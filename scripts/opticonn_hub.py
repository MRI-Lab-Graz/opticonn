#!/usr/bin/env python3
"""
Vendored OptiConn Hub CLI for local use inside the opticonn/ workspace.

This is a near-copy of the upstream hub but lives inside opticonn/scripts so
all Phase 2 work can be executed without depending on the parent repository.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure repository root (parent of this scripts/ directory) is on sys.path so
# `import scripts.*` resolves when this file is executed as a script.
_this_dir = Path(__file__).resolve().parent
_repo_root = str(_this_dir.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.utils.runtime import configure_stdio, propagate_no_emoji


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _abs(path_like: str | os.PathLike | None) -> str | None:
    if not path_like:
        return None
    return str(Path(path_like).resolve())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OptiConn - Unbiased, modality-agnostic connectomics optimization & analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--version", action="version", version="OptiConn v2.0.0")
    parser.add_argument(
        "--no-emoji",
        action="store_true",
        help="Disable emoji in console output (useful on limited terminals)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform a dry-run: print the command(s) that would be executed without running them",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_opt = subparsers.add_parser(
        "optimize", help="Find optimal parameters using cross-validation"
    )
    p_opt.add_argument("-i", "--data-dir", required=True)
    p_opt.add_argument("-o", "--output-dir", required=True)
    p_opt.add_argument("--config", help="Optional master optimizer config (rare).")
    p_opt.add_argument(
        "--quick", action="store_true", help="Run a tiny demonstration sweep"
    )
    p_opt.add_argument("--subjects", type=int, default=3)
    p_opt.add_argument(
        "--max-parallel", type=int, help="Max combinations to run in parallel per wave"
    )
    p_opt.add_argument(
        "--prune-nonbest",
        action="store_true",
        help="Prune non-best combos to save disk space",
    )
    p_opt.add_argument(
        "--extraction-config",
        help="Override extraction config for auto-generated waves",
    )
    p_opt.add_argument(
        "--pareto-report",
        action="store_true",
        help="After optimization, generate a Pareto report from wave diagnostics",
    )
    p_opt.add_argument(
        "--no-emoji",
        action="store_true",
        help="Disable emoji in console output (Windows-safe)",
    )

    for name in ("analyze", "apply"):
        p = subparsers.add_parser(
            name, help="Run full analysis with optimal parameters"
        )
        p.add_argument("-i", "--data-dir", required=True)
        p.add_argument("--optimal-config", required=True)
        p.add_argument("-o", "--output-dir", default="analysis_results")
        p.add_argument("--outlier-detection", action="store_true")
        p.add_argument("--skip-extraction", action="store_true")
        p.add_argument("--interactive", action="store_true")
        p.add_argument("--candidate-index", type=int, default=1)
        p.add_argument("--quiet", action="store_true")
        p.add_argument(
            "--no-emoji",
            action="store_true",
            help="Disable emoji in console output (Windows-safe)",
        )

    p_pipe = subparsers.add_parser(
        "pipeline", help="Advanced pipeline execution (steps 01–03)"
    )
    p_pipe.add_argument(
        "--step", default="all", choices=["01", "02", "03", "all", "analysis"]
    )
    p_pipe.add_argument("-i", "--input")
    p_pipe.add_argument("-o", "--output")
    p_pipe.add_argument("--config")
    p_pipe.add_argument("--data-dir")
    p_pipe.add_argument("--cross-validated-config")
    p_pipe.add_argument("--quiet", action="store_true")
    p_pipe.add_argument(
        "--no-emoji",
        action="store_true",
        help="Disable emoji in console output (Windows-safe)",
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    root = repo_root()
    scripts_dir = root / "scripts"
    no_emoji = configure_stdio(args.no_emoji)

    if args.command == "optimize":
        cmd = [
            sys.executable,
            str(scripts_dir / "cross_validation_bootstrap_optimizer.py"),
            "--data-dir",
            _abs(args.data_dir),
            "--output-dir",
            _abs(args.output_dir),
        ]
        chosen_extraction_cfg: str | None = None
        chosen_master_cfg: str | None = None
        if args.quick:
            chosen_extraction_cfg = str(root / "configs" / "sweep_micro.json")
        if args.extraction_config:
            chosen_extraction_cfg = _abs(args.extraction_config)
        if args.config:
            try:
                import json as _json

                _cfg_path = Path(args.config)
                cfg_txt = _cfg_path.read_text()
                cfg_json = _json.loads(cfg_txt)
                is_master = any(
                    k in cfg_json
                    for k in ("wave1_config", "wave2_config", "bootstrap_optimization")
                )
                is_extraction_like = any(
                    k in cfg_json
                    for k in ("atlases", "connectivity_values", "sweep_parameters")
                )
                if is_master and not is_extraction_like:
                    chosen_master_cfg = _abs(args.config)
                else:
                    chosen_extraction_cfg = _abs(args.config)
            except Exception:
                chosen_extraction_cfg = _abs(args.config)

        if chosen_master_cfg:
            cmd += ["--config", chosen_master_cfg]
        if chosen_extraction_cfg:
            cmd += ["--extraction-config", chosen_extraction_cfg]

        if args.subjects:
            cmd += ["--subjects", str(int(args.subjects))]
        if args.max_parallel and int(args.max_parallel) > 1:
            cmd += ["--max-parallel", str(int(args.max_parallel))]
        if args.prune_nonbest:
            cmd += ["--prune-nonbest"]

        if no_emoji:
            cmd.append("--no-emoji")

        if chosen_extraction_cfg:
            print(f"🧪 Using extraction config: {chosen_extraction_cfg}")
        if chosen_master_cfg:
            print(f"📋 Using master optimizer config: {chosen_master_cfg}")
        print(f"🚀 Running: {' '.join(cmd)}")
        import subprocess

        env = propagate_no_emoji()
        try:
            subprocess.run(cmd, check=True, env=env)
            print("✅ Parameter optimization completed successfully!")
            print(f"📋 Results saved to: {Path(args.output_dir) / 'optimize'}")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"❌ Optimization failed with error code {e.returncode}")
            return e.returncode

    if args.command in ("analyze", "apply"):
        import json

        cfg_path = Path(args.optimal_config)
        try:
            cfg_json = json.loads(Path(cfg_path).read_text())
        except Exception:
            cfg_json = None

        out_selected = Path(args.output_dir) / "selected"
        if isinstance(cfg_json, list):

            def score(obj):
                for k in ("average_score", "score", "pure_qa_score", "quality_score"):
                    v = obj.get(k)
                    if isinstance(v, (int, float)):
                        return float(v)
                pw = obj.get("per_wave")
                if isinstance(pw, list):
                    vals = [
                        w.get("score")
                        for w in pw
                        if isinstance(w, dict)
                        and isinstance(w.get("score"), (int, float))
                    ]
                    if vals:
                        return float(sum(vals) / len(vals))
                return 0.0

            ranked = sorted(cfg_json, key=score, reverse=True)
            idx = max(1, min(args.candidate_index, len(ranked))) - 1
            chosen = ranked[idx]

            dsi_cmd = os.environ.get("DSI_STUDIO_CMD")
            if (
                not dsi_cmd
                and (root / "configs" / "braingraph_default_config.json").exists()
            ):
                try:
                    default_cfg = json.loads(
                        (
                            root / "configs" / "braingraph_default_config.json"
                        ).read_text()
                    )
                    dsi_cmd = default_cfg.get("dsi_studio_cmd")
                except Exception:
                    dsi_cmd = None
            if not dsi_cmd:
                dsi_cmd = (
                    "/Applications/dsi_studio.app/Contents/MacOS/dsi_studio"
                    if sys.platform == "darwin"
                    else "dsi_studio"
                )

            chosen_params = (
                chosen.get("parameters") if isinstance(chosen, dict) else None
            )
            extraction_cfg = {
                "description": "Extraction from selection (optimal_combinations.json)",
                "atlases": [chosen["atlas"]],
                "connectivity_values": [chosen["connectivity_metric"]],
                "dsi_studio_cmd": dsi_cmd,
            }
            try:
                if isinstance(chosen_params, dict):
                    if "tract_count" in chosen_params:
                        extraction_cfg["tract_count"] = chosen_params["tract_count"]
                    tp = chosen_params.get("tracking_parameters") or {}
                    if tp:
                        extraction_cfg.setdefault("tracking_parameters", {})
                        for k in (
                            "fa_threshold",
                            "turning_angle",
                            "step_size",
                            "smoothing",
                            "min_length",
                            "max_length",
                            "track_voxel_ratio",
                            "dt_threshold",
                        ):
                            if tp.get(k) is not None:
                                extraction_cfg["tracking_parameters"][k] = tp.get(k)
                    ct = chosen_params.get("connectivity_threshold")
                    if ct is not None:
                        extraction_cfg.setdefault("connectivity_options", {})
                        extraction_cfg["connectivity_options"][
                            "connectivity_threshold"
                        ] = ct
            except Exception:
                pass
            out_selected.mkdir(parents=True, exist_ok=True)
            extraction_cfg_path = out_selected / "extraction_from_selection.json"
            extraction_cfg_path.write_text(json.dumps(extraction_cfg, indent=2))
            try:
                (out_selected / "selected_parameters.json").write_text(
                    json.dumps({"selected_config": extraction_cfg}, indent=2)
                )
            except Exception:
                pass

            cmd = [
                sys.executable,
                "-m",
                "scripts.run_pipeline",
                "--data-dir",
                _abs(args.data_dir),
                "--output",
                str(out_selected),
                "--extraction-config",
                str(extraction_cfg_path),
                "--step",
                "analysis" if args.skip_extraction else "all",
            ]
            if args.quiet:
                cmd.append("--quiet")
        else:
            cmd = [
                sys.executable,
                "-m",
                "scripts.run_pipeline",
                "--cross-validated-config",
                _abs(args.optimal_config),
                "--data-dir",
                _abs(args.data_dir),
                "--output",
                str(out_selected),
                "--step",
                "analysis" if args.skip_extraction else "all",
            ]
            if args.quiet:
                cmd.append("--quiet")

        if no_emoji:
            cmd.append("--no-emoji")

        print(f"🚀 Running: {' '.join(cmd)}")
        import subprocess

        env = propagate_no_emoji()
        try:
            subprocess.run(cmd, check=True, env=env)
            print("✅ Complete analysis finished successfully!")
            print(f"📋 Results available in: {out_selected}")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"❌ Analysis failed with error code {e.returncode}")
            return e.returncode

    if args.command == "pipeline":
        cmd = [sys.executable, "-m", "scripts.run_pipeline"]
        if args.step:
            cmd += ["--step", args.step]
        if args.input:
            cmd += ["--input", _abs(args.input)]
        if args.output:
            cmd += ["--output", _abs(args.output)]
        if args.config:
            cmd += ["--extraction-config", _abs(args.config)]
        else:
            cmd += [
                "--extraction-config",
                str(root / "configs" / "braingraph_default_config.json"),
            ]
        if args.data_dir:
            cmd += ["--data-dir", _abs(args.data_dir)]
        if args.cross_validated_config:
            cmd += ["--cross-validated-config", _abs(args.cross_validated_config)]
        if args.quiet:
            cmd.append("--quiet")

        if no_emoji:
            cmd.append("--no-emoji")

        print(f"🚀 Running: {' '.join(cmd)}")
        import subprocess

        env = propagate_no_emoji()
        try:
            subprocess.run(cmd, check=True, env=env)
            print("✅ Pipeline execution completed!")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"❌ Pipeline failed with error code {e.returncode}")
            return e.returncode

    print("Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
