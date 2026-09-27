"""Preflight: prove, before a battery runs, that every specification executes as sent.

For each distinct specification of a sweep config (grid and reference) and each
repeat, one tracking run on one subject through the verified extractor path (echo
and execution report, scripts/dsi_verify.py). For repeat 1 additionally:
  - a direct .trk export and a conversion of the verified tract file; the two must
    be the same streamlines, and the direct export must obey the executed step,
    turning angle, length bounds and count (streamline geometry);
  - the differential check: specifications differing in exactly one parameter must
    produce different streamlines -- a parameter that changes nothing was not
    executed. Tracking is deterministic at one thread, so this is exact.

Writes <out>/preflight.json with the expected parameter_id per (specification,
repeat) -- the fingerprints every production run is checked against -- and exits 1
unless everything passed. Run once per pinned DSI Studio build.

    python -m scripts.dsi_preflight --config configs/battery.json \\
        --subject staged/ds000221/sub-X.qsdr.fz --out preflight/ --max-parallel 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scripts import dsi_verify
from scripts.cross_validation_bootstrap_optimizer import (
    apply_param_choice_to_config, apply_unmapped_params, build_combos)
from scripts.extract_connectivity_matrices import ConnectivityExtractor, build_track_command
from scripts.sweep_utils import build_param_grid_from_config

ATLAS = "AAL3"


def enumerate_specs(config: dict) -> list[dict]:
    """Every distinct specification x repeat of a sweep config, as extractor configs."""
    if config.get("thread_count") != 1:
        raise ValueError("preflight needs thread_count 1: multithreaded DSI Studio tracking "
                         "is not deterministic, so neither fingerprints nor the "
                         "differential check would be reproducible")
    sp = config.get("sweep_parameters") or {}
    _, mapping = build_param_grid_from_config({"sweep_parameters": sp})
    combos, _, _ = build_combos(sp, None)
    repeats = int((config.get("reliability") or {}).get("repeats", 2))
    specs = []
    for spec, choice in enumerate(combos):
        base = apply_unmapped_params(apply_param_choice_to_config(config, choice, mapping),
                                     choice, mapping)
        for repeat in range(1, repeats + 1):
            cfg = json.loads(json.dumps(base))
            cfg.pop("sweep_parameters", None)
            cfg.setdefault("tracking_parameters", {})["random_seed"] = repeat
            specs.append({"spec": spec, "repeat": repeat, "choice": choice, "config": cfg})
    return specs


def single_axis_pairs(choices: list[dict]) -> list[tuple[int, int, str]]:
    """(i, j, parameter) for specifications differing in exactly one parameter."""
    pairs = []
    for i, j in itertools.combinations(range(len(choices)), 2):
        a, b = choices[i], choices[j]
        diff = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
        if len(diff) == 1:
            pairs.append((i, j, diff[0]))
    return pairs


def summarize(runs: list[dict], pairs, digests: dict) -> dict:
    """Pass/fail and the expected fingerprints from all runs of a preflight."""
    failures = [f"spec {r['spec']} repeat {r['repeat']}: {e}" for r in runs for e in r["errors"]]
    by_pid: dict = {}
    for r in runs:
        by_pid.setdefault(r["parameter_id"], set()).add(r["fingerprint_key"])
    failures += [f"{len(keys)} different specifications produced the same parameter_id {pid}"
                 for pid, keys in by_pid.items() if pid is not None and len(keys) > 1]
    failures += [f"{axis} did not change the streamlines (specs {i} and {j})"
                 for i, j, axis in pairs if digests.get(i) == digests.get(j)]
    return {
        "passed": not failures,
        "failures": failures,
        "expected_fingerprints": {r["fingerprint_key"]: r["parameter_id"] for r in runs},
    }


def _dsi_cmd() -> str:
    return os.environ.get("DSI_STUDIO_PATH", "dsi_studio")


def _atlas_arg(cfg: dict) -> str:
    """The atlas argument the extractor sends: a full path when the atlas is external."""
    atlas_dir = cfg.get("atlas_dir")
    if atlas_dir and (Path(atlas_dir) / f"{ATLAS}.nii.gz").exists():
        return str(Path(atlas_dir) / f"{ATLAS}.nii.gz")
    return ATLAS


def _run_spec(item: dict, subject: str, out: Path) -> dict:
    """One verified tracking run; for repeat 1 also the geometry proof."""
    import nibabel as nib  # preflight-only dependency

    run_dir = out / f"spec{item['spec']:03d}_rep{item['repeat']}"
    # A rerun into the same --out must not let a previous run's execution
    # record, tract file or .trk exports be read as if this run produced them.
    shutil.rmtree(run_dir, ignore_errors=True)
    cfg = {**item["config"], "atlases": [ATLAS],
           "verification": {"keep_tract": True}, "dsi_studio_cmd": _dsi_cmd()}
    res = ConnectivityExtractor(cfg).extract_connectivity_matrix(subject, run_dir, ATLAS, "pf")
    atlas_dir = run_dir / "results" / ATLAS
    info = json.loads((atlas_dir / f"pf_{ATLAS}.dsi_execution.json").read_text())
    errors = list(res.get("verification_errors") or [])
    if not res.get("success") and not errors:
        errors.append("tracking failed without a verification error; see the run log")

    if item["repeat"] == 1 and not errors:
        tract = atlas_dir / f"pf_{ATLAS}.tt.gz"
        direct = run_dir / "direct.trk.gz"
        converted = run_dir / "converted.trk.gz"
        # A stale export from a previous run into this same directory must never
        # be mistaken for this run's proof, even if the fresh export fails.
        direct.unlink(missing_ok=True)
        converted.unlink(missing_ok=True)
        # Same command as the verified run except the output file, so the direct
        # export is the same execution (deterministic at one thread).
        cmd = build_track_command(cfg, _dsi_cmd(), subject, str(direct), _atlas_arg(cfg))
        direct_proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        convert_proc = subprocess.run(
            [_dsi_cmd(), "--action=ana", f"--source={subject}", f"--tract={tract}",
             f"--output={converted}"], capture_output=True, text=True, check=False)
        if direct_proc.returncode != 0:
            errors.append(f"direct .trk export failed (returncode {direct_proc.returncode})")
        if convert_proc.returncode != 0:
            errors.append(f"tract conversion failed (returncode {convert_proc.returncode})")
        if not (direct.exists() and converted.exists()):
            errors.append("geometry export failed (direct or converted .trk missing)")
        elif not errors:
            # lazy_load=True: the non-lazy path calls seek(0, SEEK_END) purely to
            # guess a read buffer size, and indexed_gzip's IndexedGzipFile (which
            # nibabel picks for .gz files when the package is installed) refuses
            # that seek unless its full index is already built -- it raises
            # NotCoveredError on every real DSI Studio .trk.gz. Lazy loading skips
            # that call; the streamlines and header below are still fully read.
            a = nib.streamlines.load(str(direct), lazy_load=True)
            b = nib.streamlines.load(str(converted), lazy_load=True)
            voxel = float(a.header["voxel_sizes"][0])
            errors += dsi_verify.same_streamlines(list(a.streamlines), list(b.streamlines))
            errors += dsi_verify.check_geometry(list(a.streamlines), info["executed"], voxel)
        tract.unlink(missing_ok=True)

    return {"spec": item["spec"], "repeat": item["repeat"],
            "fingerprint_key": dsi_verify.fingerprint_key(info["sent"]),
            "parameter_id": info["parameter_id"], "track_sha256": info["track_sha256"],
            "executed": info["executed"], "errors": errors}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--subject", required=True, help="one QSDR .fz file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-parallel", type=int, default=8)
    args = ap.parse_args(argv)

    config = json.loads(Path(args.config).read_text())
    specs = enumerate_specs(config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # A rerun into the same --out must not leave a previous "passed": true
    # verdict on disk if this run crashes before writing its own.
    (out / "preflight.json").unlink(missing_ok=True)
    with ProcessPoolExecutor(max_workers=args.max_parallel) as pool:
        runs = list(pool.map(_run_spec, specs, [args.subject] * len(specs),
                             [out] * len(specs)))

    first = {r["spec"]: r for r in runs if r["repeat"] == 1}
    choices = [next(s["choice"] for s in specs if s["spec"] == i) for i in sorted(first)]
    summary = summarize(runs, single_axis_pairs(choices),
                        {i: first[i]["track_sha256"] for i in first})
    summary.update({
        "dsi_apptainer_image": os.environ.get("DSI_APPTAINER_IMAGE"),
        "subject": args.subject, "config": args.config, "runs": runs,
        "determinism": {i: len({r["track_sha256"] for r in runs if r["spec"] == i}) == 1
                        for i in first},
    })
    (out / "preflight.json").write_text(json.dumps(summary, indent=2))
    print(f"preflight {'PASSED' if summary['passed'] else 'FAILED'}: "
          f"{len(specs)} runs, {len(summary['failures'])} failure(s) -> {out / 'preflight.json'}")
    for f in summary["failures"]:
        print(f"  - {f}", file=sys.stderr)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
