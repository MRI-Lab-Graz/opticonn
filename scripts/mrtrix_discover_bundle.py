#!/usr/bin/env python3
# Supports: --dry-run (prints intended actions without running)
# When run without arguments the script prints help: parser.print_help()
"""MRtrix bundle discovery helper.

Author: Karl Koschutnig (MRI-Lab Graz)
Contact: karl.koschutnig@uni-graz.at
Date:

Purpose
-------
Create an MRtrix tuning config (compatible with `scripts/mrtrix_tune.py`) by discovering
tracking-ready QSIRecon outputs.

Two discovery modes
-------------------
1) Explicit paths:
   - provide `--qsirecon-dir` (and optionally `--qsiprep-dir`)
2) Single derivatives root (common lab layout):
   - provide `--derivatives-dir` and the script will locate qsirecon/qsiprep under it.

The script locates:
- WM FOD image (wm_fod)
- ACT tissue image (act_5tt_or_hsvs): tries hsvs probseg first
- Parcellation dseg + label lookup for an atlas

It then writes a JSON config usable by:
  python scripts/mrtrix_tune.py sweep|bayes --config <json> ...

Notes
-----
- This is intentionally conservative: if multiple candidates are found it will ask you
  to disambiguate (e.g., specify session, workflow tag).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DiscoveredPaths:
    qsirecon_dir: Path
    qsiprep_dir: Optional[Path]
    wm_fod: Path
    act_5tt_or_hsvs: Optional[Path]
    dseg: Path
    labels: Path


def _print_candidates(label: str, candidates: Sequence[Path]) -> str:
    lines = [f"{label} ({len(candidates)} candidates):"]
    for p in candidates:
        lines.append(f"  - {p}")
    return "\n".join(lines)


def _ensure_dir(p: Path) -> Path:
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(str(p))
    return p


def _first_existing_dir(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists() and p.is_dir():
            return p
    return None


def _glob_one_or_error(globs: Sequence[Path], label: str, allow_many: bool = False) -> Path:
    candidates: List[Path] = []
    for g in globs:
        candidates.extend(sorted(g.parent.glob(g.name)))

    # Deduplicate while preserving order
    seen: set[Path] = set()
    uniq: List[Path] = []
    for c in candidates:
        rc = c.resolve()
        if rc in seen:
            continue
        seen.add(rc)
        uniq.append(c)

    if not uniq:
        raise FileNotFoundError(f"No matches for {label}")

    if len(uniq) == 1 or allow_many:
        return uniq[0]

    # Heuristics: prefer msmtcsd fod if present
    preferred = [p for p in uniq if "model-msmtcsd" in p.name or "msmtcsd" in p.name]
    if len(preferred) == 1:
        return preferred[0]

    # Heuristic: prefer MRtrix-native .mif/.mif.gz when both .mif and .nii exist
    mif = [p for p in uniq if p.name.endswith(".mif") or p.name.endswith(".mif.gz")]
    if len(mif) == 1:
        return mif[0]

    raise ValueError(_print_candidates(f"Ambiguous {label}", uniq))


def _prefer_dirs_by_hint(dirs: Sequence[Path], hint: Optional[str]) -> List[Path]:
    if not dirs:
        return []
    if hint:
        return [d for d in dirs if hint in d.name]

    # Default heuristic: prefer MRtrix workflows if present.
    mrtrix = [d for d in dirs if "mrtrix" in d.name.lower()]
    return mrtrix or list(dirs)


def _find_first_matches(
    workflow_dirs: Sequence[Path],
    subject: str,
    session: Optional[str],
    filename_glob: str,
) -> List[Path]:
    matches: List[Path] = []
    for wf in workflow_dirs:
        candidates_dirs: List[Path] = []
        if session:
            candidates_dirs.append(wf / subject / session / "dwi")
        candidates_dirs.append(wf / subject / "dwi")

        for d in candidates_dirs:
            if not d.exists() or not d.is_dir():
                continue
            matches.extend(sorted(d.glob(filename_glob)))
        if matches:
            break
    return matches


def _find_qsi_roots(derivatives_dir: Path) -> Tuple[Path, Optional[Path]]:
    qsirecon = _first_existing_dir(
        [
            derivatives_dir / "qsirecon",
            derivatives_dir / "qsirecon" / "derivatives",
        ]
    )

    # Some installations nest an additional derivatives/qsirecon/derivatives/... layer.
    if qsirecon is None:
        # fallback: search one level deep
        for p in sorted(derivatives_dir.glob("*")):
            if p.is_dir() and "qsirecon" in p.name.lower():
                qsirecon = p
                break

    if qsirecon is None:
        raise FileNotFoundError(
            f"Could not find qsirecon under derivatives root: {derivatives_dir}"
        )

    qsiprep = _first_existing_dir(
        [
            derivatives_dir / "qsiprep",
            derivatives_dir / "fmriprep",  # sometimes present; not used here but helps
        ]
    )

    return _ensure_dir(qsirecon), qsiprep


def _infer_sessions(qsirecon_dir: Path, subject: str) -> List[str]:
    subj = qsirecon_dir / subject
    if not subj.exists():
        return []
    ses = [p.name for p in subj.glob("ses-*") if p.is_dir()]
    return sorted(ses)


def discover_bundle(
    *,
    derivatives_dir: Optional[Path],
    qsirecon_dir: Optional[Path],
    qsiprep_dir: Optional[Path],
    subject: str,
    session: Optional[str],
    atlas: str,
    workflow_hint: Optional[str],
    allow_missing_act: bool,
) -> DiscoveredPaths:
    if derivatives_dir is not None:
        qsirecon_root, qsiprep_root = _find_qsi_roots(_ensure_dir(derivatives_dir))
        if qsirecon_dir is None:
            qsirecon_dir = qsirecon_root
        if qsiprep_dir is None:
            qsiprep_dir = qsiprep_root

    if qsirecon_dir is None:
        raise ValueError("Must provide --qsirecon-dir or --derivatives-dir")

    qsirecon_dir = _ensure_dir(qsirecon_dir)
    if qsiprep_dir is not None:
        qsiprep_dir = _ensure_dir(qsiprep_dir)

    if session is None:
        sessions = _infer_sessions(qsirecon_dir, subject)
        if len(sessions) == 1:
            session = sessions[0]
        elif len(sessions) > 1:
            raise ValueError(
                "Multiple sessions found; please specify --session.\n"
                + "\n".join(f"  - {s}" for s in sessions)
            )
        else:
            session = None

    # ---- WM FOD ----
    fod_globs: List[Path] = []

    # Typical qsirecon MRtrix workflow locations:
    # <qsirecon>/derivatives/<workflow>/sub-*/ses-*/dwi/*_label-WM*_dwimap.mif.gz
    # Also allow without ses- for single-session datasets.
    workflow_dirs: List[Path] = []
    derivatives_layer = qsirecon_dir / "derivatives"
    if derivatives_layer.exists() and derivatives_layer.is_dir():
        workflow_dirs = [p for p in derivatives_layer.glob("*") if p.is_dir()]

    if workflow_hint:
        hinted = [p for p in workflow_dirs if workflow_hint in p.name]
        if not hinted:
            raise FileNotFoundError(
                f"No qsirecon workflow directories matched --workflow-hint '{workflow_hint}' under {derivatives_layer}"
            )
        workflow_dirs = hinted

    if workflow_dirs:
        preferred = _prefer_dirs_by_hint(workflow_dirs, workflow_hint)
        fod_matches = _find_first_matches(
            preferred,
            subject=subject,
            session=session,
            filename_glob="*_label-WM*_dwimap.mif*",
        )
        if fod_matches:
            # Convert to globs for the existing selection logic
            fod_globs = fod_matches
        elif workflow_hint:
            raise FileNotFoundError(
                f"No WM FOD found under hinted workflow '{workflow_hint}'. Try without --workflow-hint or specify the correct hint."
            )

    # fallback: search within qsirecon root too (only if workflow search did not find anything)
    if not fod_globs:
        if session:
            fod_globs.append(
                qsirecon_dir
                / subject
                / session
                / "dwi"
                / "*_label-WM*_dwimap.mif*"
            )
        fod_globs.append(qsirecon_dir / subject / "dwi" / "*_label-WM*_dwimap.mif*")

    wm_fod = _glob_one_or_error(fod_globs, "WM FOD")

    # ---- ACT/HSVS ----
    act_candidates: List[Path] = []
    for root in [qsirecon_dir, qsiprep_dir]:
        if root is None:
            continue
        
        search_dirs = [root / subject / "anat"]
        if session:
            search_dirs.append(root / subject / session / "anat")

        for d in search_dirs:
            if not d.exists():
                continue
            # In your PK01 layout: <qsirecon>/sub-*/anat/*_space-ACPC_seg-hsvs_probseg.nii.gz
            act_candidates.extend(
                sorted(d.glob("*_space-ACPC*_seg-hsvs_probseg.nii*") )
            )
            # Allow alternate naming
            act_candidates.extend(
                sorted(d.glob("*_seg-hsvs_probseg.nii*") )
            )
            act_candidates.extend(
                sorted(d.glob("*_5tt*.mif*") )
            )

    act_5tt_or_hsvs: Optional[Path] = None
    if act_candidates:
        # pick the first (sorted) candidate
        act_5tt_or_hsvs = sorted({p.resolve(): p for p in act_candidates}.values(), key=lambda p: str(p))[0]
    elif not allow_missing_act:
        raise FileNotFoundError(
            "Could not find ACT tissue image (hsvs probseg or 5TT). "
            "Provide --qsiprep-dir/--qsirecon-dir, or set --allow-missing-act to emit a config without ACT."
        )

    # ---- Parcellation dseg + labels ----
    dseg_globs: List[Path] = []
    qsirecon_subj = qsirecon_dir / subject
    if session:
        dseg_globs.append(
            qsirecon_subj
            / session
            / "dwi"
            / f"*_space-ACPC*_seg-{atlas}_dseg.mif*"
        )
        dseg_globs.append(
            qsirecon_subj
            / session
            / "dwi"
            / f"*_space-ACPC*_seg-{atlas}_dseg.nii*"
        )
        # More flexible fallback
        dseg_globs.append(
            qsirecon_subj
            / session
            / "dwi"
            / f"*_seg-{atlas}_dseg.mif*"
        )
        dseg_globs.append(
            qsirecon_subj
            / session
            / "dwi"
            / f"*_seg-{atlas}_dseg.nii*"
        )
    dseg_globs.append(qsirecon_subj / "dwi" / f"*_seg-{atlas}_dseg.mif*")
    dseg_globs.append(qsirecon_subj / "dwi" / f"*_seg-{atlas}_dseg.nii*")

    dseg = _glob_one_or_error(dseg_globs, f"parcellation dseg for atlas={atlas}")

    labels_candidates: List[Path] = []
    # labels are usually adjacent (txt/tsv)
    labels_candidates.extend(sorted(dseg.parent.glob(f"*_seg-{atlas}_dseg.txt")))
    labels_candidates.extend(sorted(dseg.parent.glob(f"*_seg-{atlas}_dseg.tsv")))
    labels_candidates.extend(sorted(dseg.parent.glob(f"*_seg-{atlas}_dseg*.txt")))
    labels_candidates.extend(sorted(dseg.parent.glob(f"*_seg-{atlas}_dseg*.tsv")))

    # Deduplicate (multiple glob patterns can return the same path)
    labels_candidates = list({p.resolve(): p for p in labels_candidates}.values())
    labels_candidates = sorted(labels_candidates, key=lambda p: str(p))

    if not labels_candidates:
        raise FileNotFoundError(
            f"Could not find labels lookup for atlas '{atlas}' next to {dseg}. "
            "Expected something like *_seg-<atlas>_dseg.txt or .tsv"
        )

    if len(labels_candidates) > 1:
        # If multiple, prefer exact *_dseg.txt
        exact = [p for p in labels_candidates if p.name.endswith(f"_seg-{atlas}_dseg.txt")]
        if len(exact) == 1:
            labels = exact[0]
        else:
            raise ValueError(_print_candidates(f"Ambiguous labels lookup for atlas={atlas}", labels_candidates))
    else:
        labels = labels_candidates[0]

    return DiscoveredPaths(
        qsirecon_dir=qsirecon_dir,
        qsiprep_dir=qsiprep_dir,
        wm_fod=wm_fod,
        act_5tt_or_hsvs=act_5tt_or_hsvs,
        dseg=dseg,
        labels=labels,
    )


def _default_out_config(derivatives_dir: Optional[Path], subject: str, session: Optional[str], atlas: str) -> Path:
    if derivatives_dir is None:
        # fall back to repo-local studies
        base = Path.cwd() / "studies" / "mrtrix_discovered"
    else:
        base = derivatives_dir / "opticonn" / "mrtrix_tune_configs"

    tag = subject
    if session:
        tag += f"_{session}"
    tag += f"_{atlas}"
    return base / tag / "mrtrix_tune_config.json"


def build_config_json(paths: DiscoveredPaths, atlas: str) -> dict:
    cfg = {
        "description": f"MRtrix tuning config discovered for {paths.wm_fod.name}",
        "backend": "mrtrix",
        "inputs": {
            "bundle": {
                "wm_fod": str(paths.wm_fod),
                "act_5tt_or_hsvs": str(paths.act_5tt_or_hsvs) if paths.act_5tt_or_hsvs else None,
                "parcellations": [
                    {
                        "name": atlas,
                        "dseg": str(paths.dseg),
                        "labels_tsv": str(paths.labels),
                    }
                ],
            }
        },
        "mrtrix": {
            "tckgen": {
                "algorithm": "iFOD2",
                "select": 1000000,
                "maxlength": 250,
                "minlength": 30,
                "power": 0.33,
                "backtrack": False,
                "crop_at_gmwmi": False,
                "seed": {"type": "dynamic", "image": None},
            },
            "tcksift2": {"enabled": False},
            "tck2connectome": {"assignment_radial_search_mm": 2},
        },
        "search_space": {
            "type": "random",
            "parameters": {
                "tckgen.cutoff": [0.06, 0.20],
                "tckgen.angle": [30, 60],
                "tckgen.step": [0.5, 1.5],
            },
        },
    }

    # Remove act field if missing to avoid confusion; user can add later.
    if cfg["inputs"]["bundle"]["act_5tt_or_hsvs"] is None:
        del cfg["inputs"]["bundle"]["act_5tt_or_hsvs"]

    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover MRtrix tracking bundle from QSIRecon outputs and write mrtrix_tune config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--derivatives-dir", type=Path, default=None, help="BIDS derivatives root (auto-detect qsiprep/qsirecon underneath)")
    parser.add_argument("--qsirecon-dir", type=Path, default=None, help="Explicit qsirecon derivatives directory")
    parser.add_argument("--qsiprep-dir", type=Path, default=None, help="Explicit qsiprep derivatives directory (optional)")

    parser.add_argument("--subject", required=True, help="Subject label, e.g., sub-1293171")
    parser.add_argument("--session", default=None, help="Session label, e.g., ses-3 (optional if only one exists)")
    parser.add_argument("--atlas", required=True, help="Atlas name, e.g., Brainnetome246Ext")

    parser.add_argument(
        "--workflow-hint",
        default=None,
        help="Substring to disambiguate qsirecon/derivatives/<workflow>/ selection (e.g., qsirecon-MRtrix3_act-HSVS)",
    )

    parser.add_argument(
        "--out-config",
        type=Path,
        default=None,
        help="Where to write the mrtrix_tune_config.json (default: derivatives/opticonn/mrtrix_tune_configs/<subject>_<session>_<atlas>/mrtrix_tune_config.json)",
    )

    parser.add_argument(
        "--allow-missing-act",
        action="store_true",
        help="Allow emitting config even if ACT tissue image cannot be found (you must avoid --enable-act later)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered paths and output location, but do not write files",
    )

    # Print help when called with no args (repo convention)
    if len(os.sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    found = discover_bundle(
        derivatives_dir=args.derivatives_dir,
        qsirecon_dir=args.qsirecon_dir,
        qsiprep_dir=args.qsiprep_dir,
        subject=args.subject,
        session=args.session,
        atlas=args.atlas,
        workflow_hint=args.workflow_hint,
        allow_missing_act=args.allow_missing_act,
    )

    out_config = args.out_config
    if out_config is None:
        out_config = _default_out_config(args.derivatives_dir, args.subject, args.session, args.atlas)

    cfg = build_config_json(found, args.atlas)

    print("Discovered MRtrix bundle:")
    print(f"  qsirecon_dir: {found.qsirecon_dir}")
    if found.qsiprep_dir:
        print(f"  qsiprep_dir:  {found.qsiprep_dir}")
    print(f"  wm_fod:       {found.wm_fod}")
    print(f"  act:          {found.act_5tt_or_hsvs if found.act_5tt_or_hsvs else '(missing)'}")
    print(f"  dseg:         {found.dseg}")
    print(f"  labels:       {found.labels}")
    print("")
    print(f"Config output:  {out_config}")

    if args.dry_run:
        print("DRY-RUN: no files written")
        return 0

    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(json.dumps(cfg, indent=2))
    print(f"Wrote config:   {out_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
