# Subject-Ordering Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit one self-contained HTML file per sweep in which a reader drags a slider across candidate specifications and watches subject ranks reshuffle against DSI Studio's default parameters, with the tracking-noise floor shown alongside.

**Architecture:** A new `scripts/ordering_viewer.py` reuses the matrix-collection and graph-measure machinery that `scripts/graph_icc.py` already walks, precomputes two Spearman correlations per candidate in Python, and inlines the result as JSON into a single HTML template with no server, bundler or JavaScript dependency. A prior change makes the sweep runner able to run DSI Studio's defaults as a non-ranked reference combo, since the grid contains no defaults cell to measure displacement from.

**Tech Stack:** Python 3.10+, numpy, scipy (`scipy.stats.spearmanr`), networkx — all already dependencies. Vanilla JavaScript and inline SVG on the page. pytest for tests.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-25-ordering-viewer-design.md`. Read it before Task 4.
- **No new runtime dependency.** Not for Python, and none at all for the page: no CDN, no bundler, no JavaScript library. The HTML must open over `file://`.
- **Refinement of the spec's payload:** the spec says the payload carries per-subject per-repeat measure values. The page only needs subject *ranks* under repeat 1 plus the two precomputed correlations, so the payload carries those and not the raw values. Smaller file, same page. Everything else in the spec is binding as written.
- Subject labels are emitted as `S01`, `S02`, ... unconditionally — never the source scan identifiers, even for a viewer built from a private cohort.
- Combo ids are wave-scoped (`wave1/sweep_0001`), as `collect_sweep_matrices()` produces them. The reference combo is resolved **within each wave**; a wave's candidates are never compared against another wave's reference.
- Commit after every task. Every commit message ends with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

---

### Task 1: Exclude reference combos from ranking

DSI Studio's defaults will be run as an extra combo. It is not a screened candidate and must never win a sweep. `scripts/reliability.py`'s `rank()` and `rank_with_fallback()` are the single chokepoint every selection path routes through — both backends, and both the per-atlas/metric level and the cross-combo level (`select_best_combo()` in `cross_validation_bootstrap_optimizer.py:342` is a one-line delegation to `rank_with_fallback`). Guarding there covers every caller at once.

**Files:**
- Modify: `scripts/reliability.py:220-234` (`rank`), `scripts/reliability.py:244-262` (`rank_with_fallback`)
- Test: `tests/test_reliability.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `rank(rows)` and `rank_with_fallback(rows)` drop any row where `row.get("reference")` is truthy. Rows without the key are unaffected. Task 2 relies on this.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reliability.py`:

```python
def test_rank_excludes_reference_rows():
    from scripts.reliability import rank, rank_with_fallback

    candidate = {
        "rejected": "", "discriminability": 0.8, "discriminability_margin": 0.1,
        "repeatability": 0.5, "tract_count": 5000,
    }
    reference = {
        "rejected": "", "discriminability": 1.0, "discriminability_margin": 0.9,
        "repeatability": 0.9, "tract_count": 5000, "reference": True,
    }

    ranked = rank([reference, candidate])
    assert [r["discriminability"] for r in ranked] == [0.8]
    assert rank_with_fallback([reference, candidate])["discriminability"] == 0.8


def test_rank_with_fallback_excludes_reference_when_discriminability_is_nan():
    from scripts.reliability import rank_with_fallback

    candidate = {
        "rejected": "", "discriminability": float("nan"),
        "repeatability": 0.4, "quality_score_raw": 0.1,
    }
    reference = {
        "rejected": "", "discriminability": float("nan"),
        "repeatability": 0.99, "quality_score_raw": 0.9, "reference": True,
    }

    assert rank_with_fallback([reference, candidate])["repeatability"] == 0.4


def test_rank_with_fallback_returns_none_when_only_reference_is_usable():
    from scripts.reliability import rank_with_fallback

    reference = {
        "rejected": "", "discriminability": 1.0, "discriminability_margin": 0.5,
        "repeatability": 0.9, "tract_count": 5000, "reference": True,
    }
    assert rank_with_fallback([reference]) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_reliability.py -k reference -v`
Expected: FAIL — the reference row currently ranks first.

- [ ] **Step 3: Add the guard**

In `scripts/reliability.py`, change `rank()`'s filter from:

```python
    usable = [r for r in rows if not r["rejected"] and not np.isnan(r["discriminability"])]
```

to:

```python
    usable = [
        r for r in rows
        if not r["rejected"] and not r.get("reference") and not np.isnan(r["discriminability"])
    ]
```

and in `rank_with_fallback()`, change:

```python
    passable = [r for r in rows if not r.get("rejected")]
```

to:

```python
    passable = [r for r in rows if not r.get("rejected") and not r.get("reference")]
```

Extend `rank()`'s docstring with one line:

```python
    A row marked `reference: True` (DSI Studio's defaults, run as a displacement
    origin rather than as a candidate) is never ranked or selected.
```

- [ ] **Step 4: Run the full reliability suite**

Run: `python -m pytest tests/test_reliability.py -v`
Expected: PASS, including the pre-existing tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/reliability.py tests/test_reliability.py
git commit -m "$(cat <<'MSG'
fix(reliability): a reference combo is never ranked or selected

DSI Studio's defaults will be run alongside the grid as a displacement origin
for the ordering viewer. Guarding in rank()/rank_with_fallback() covers both
backends and both selection levels at once, since select_best_combo() delegates
to it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 2: Sweep runs a configured reference combo

**Files:**
- Modify: `scripts/cross_validation_bootstrap_optimizer.py` — combo enumeration (around line 595-613) and the task loop's `sweep_meta` (around line 664)
- Modify: `scripts/cross_validation_bootstrap_optimizer.py` — the `diag_json` dict (around line 922) and the `combo_diagnostics.csv` row assembly (around line 925)
- Test: `tests/test_cross_validation_repeats.py`

**Interfaces:**
- Consumes: Task 1's `reference: True` convention.
- Produces: a config key `sweep_parameters.reference_candidate` (a dict of tracking parameters). When present, the sweep appends exactly one extra combo with those parameters, last, with `sweep_meta["reference"] = True`; its `diagnostics.json` and its `combo_diagnostics.csv` row both carry `reference: true`. Task 4 reads that flag.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cross_validation_repeats.py`:

```python
def test_reference_candidate_is_appended_last_and_flagged():
    from scripts.cross_validation_bootstrap_optimizer import build_combos

    sp = {
        "fa_threshold_range": [0.05, 0.10],
        "reference_candidate": {"fa_threshold": 0.0, "turning_angle": 0.0, "step_size": 0.0},
    }
    combos, method, reference_index = build_combos(sp, candidate_combos=None)

    assert method == "grid"
    assert len(combos) == 3
    assert reference_index == 3
    assert combos[-1] == {"fa_threshold": 0.0, "turning_angle": 0.0, "step_size": 0.0}


def test_no_reference_candidate_means_no_reference_index():
    from scripts.cross_validation_bootstrap_optimizer import build_combos

    combos, method, reference_index = build_combos(
        {"fa_threshold_range": [0.05, 0.10]}, candidate_combos=None
    )

    assert len(combos) == 2
    assert reference_index is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_cross_validation_repeats.py -k reference -v`
Expected: FAIL with `ImportError: cannot import name 'build_combos'`.

- [ ] **Step 3: Extract combo enumeration into `build_combos` and append the reference**

The combo enumeration currently sits inline in `run_wave_sweep`. Extract it so it is testable without running a sweep. Add this function above `select_best_combo`:

```python
def build_combos(
    sp: dict, candidate_combos: list[dict] | None
) -> tuple[list[dict], str, int | None]:
    """Candidate combos, the sampler name, and the 1-based index of the reference combo.

    `sp["reference_candidate"]`, when set, is appended last as a fixed reference
    (DSI Studio's untouched defaults) rather than enumerated as a grid cell. It is
    a displacement origin for the ordering viewer, never a candidate: it is excluded
    from ranking by `scripts.reliability.rank`. Returns reference_index None when no
    reference is configured.
    """
    param_values, _ = build_param_grid_from_config({"sweep_parameters": sp})

    if candidate_combos is not None:
        combos, method = list(candidate_combos), "candidates"
    else:
        sampling = (sp.get("sampling") or {}) if isinstance(sp, dict) else {}
        method = (sampling.get("method") or "grid").lower()
        n_samples = int(sampling.get("n_samples") or 0)
        seed = int(sampling.get("random_seed") or 42)
        if method == "grid" or not param_values:
            combos = grid_product(param_values) if param_values else [{}]
        elif method == "random":
            combos = sweep_random_sampling(param_values, n_samples or 24, seed)
        else:
            combos = lhs_sampling(param_values, n_samples or 24, seed)

    reference = sp.get("reference_candidate") if isinstance(sp, dict) else None
    if not reference:
        return combos, method, None
    combos = [*combos, dict(reference)]
    return combos, method, len(combos)
```

In `run_wave_sweep`, replace the whole block from `param_values, mapping = build_param_grid_from_config(...)` through the end of the `else:` sampling branch with:

```python
    param_values, mapping = build_param_grid_from_config({"sweep_parameters": sp})
    combos, method, reference_index = build_combos(sp, candidate_combos)
```

Keep the existing `reliability_cfg` / `repeats` lines where they are.

- [ ] **Step 4: Flag the reference combo through to the artifacts**

In the task loop, the `derived["sweep_meta"]` dict gains one key:

```python
            derived["sweep_meta"] = {
                "index": i,
                "choice": choice,
                "sampler": method,
                "reference": i == reference_index,
                "total_combinations": len(combos),
                "source_config": extraction_cfg,
                "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            }
```

In the `diag_json` dict, immediately after the `"sampler"` line:

```python
                "reference": bool(sweep_meta.get("reference")),
```

In the `combo_diagnostics.csv` row dict, immediately after its `"sampler"` entry:

```python
                "reference": bool(sweep_meta.get("reference")),
```

Then pass the flag into ranking. In `run_combo`'s returned dict (the one containing `"sweep_meta": sweep_meta`), add:

```python
        "reference": bool(sweep_meta.get("reference")),
```

so that `select_best_combo` — which forwards these records to `rank_with_fallback` — sees Task 1's guard.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_cross_validation_repeats.py tests/test_reliability.py -v`
Expected: PASS.

- [ ] **Step 6: Verify the CSV header picks up the new column**

Run: `python -m pytest tests/ -v -k "diagnostics or csv or repeats"`
Expected: PASS. If a test asserts an exact `combo_diagnostics.csv` header list, add `reference` to it in the position matching the row dict.

- [ ] **Step 7: Commit**

```bash
git add scripts/cross_validation_bootstrap_optimizer.py tests/test_cross_validation_repeats.py
git commit -m "$(cat <<'MSG'
feat(sweep): run a configured reference candidate alongside the grid

sweep_parameters.reference_candidate appends one fixed combo (DSI Studio's
untouched defaults) as a displacement origin, flagged through sweep_meta to
diagnostics.json and combo_diagnostics.csv so ranking skips it. Combo
enumeration moves into build_combos() to be testable without a sweep.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 3: Amend the OpenNeuro battery to carry the reference

The battery spec and plan currently describe a 12-candidate grid with nothing to measure displacement from. The viewer's committed demo is generated from battery output, so the battery must run the reference.

Note: `docs/superpowers/plans/2026-09-25-openneuro-battery.md` may have uncommitted changes in the working tree. Check `git status` first and do not discard them.

**Files:**
- Modify: `docs/superpowers/specs/2026-09-25-openneuro-multiverse-battery-design.md:203` (the "12-candidate grid" bullet)
- Modify: `docs/superpowers/plans/2026-09-25-openneuro-battery.md` (wherever it states the candidate count or writes `configs/battery.json`)
- Modify: `docs/papers/2026-09-25-multiverse-methods-paper.md` (claim (iii) in the abstract)

**Interfaces:**
- Consumes: Task 2's `sweep_parameters.reference_candidate` config key.
- Produces: the battery's extraction config carries the reference block below. Task 6 uses it.

- [ ] **Step 1: Amend the battery spec**

In `docs/superpowers/specs/2026-09-25-openneuro-multiverse-battery-design.md`, replace the bullet:

```
- the same 12-candidate grid (FA threshold x turning angle x track/voxel ratio)
```

with:

```
- the same 12-candidate grid (FA threshold x turning angle x track/voxel ratio), plus a
  13th **reference** combo running DSI Studio's untouched defaults (`fa_threshold=0`,
  `turning_angle=0`, `step_size=0` -- all three meaning "automatic" to DSI Studio).
  The reference is not a candidate: it is excluded from ranking
  (`scripts.reliability.rank` skips rows flagged `reference`) and exists as the origin
  the ordering viewer measures displacement from. Without it there is nothing on disk
  representing the out-of-the-box setting most published connectomes silently used.
```

- [ ] **Step 2: Record the config block in the battery plan**

In `docs/superpowers/plans/2026-09-25-openneuro-battery.md`, wherever `configs/battery.json` is specified, add to its `sweep_parameters`:

```json
    "reference_candidate": {
      "fa_threshold": 0.0,
      "turning_angle": 0.0,
      "step_size": 0.0
    }
```

and update any stated candidate count from 12 to "12 candidates + 1 reference".

- [ ] **Step 3: Sharpen the paper's claim (iii)**

In `docs/papers/2026-09-25-multiverse-methods-paper.md`, claim (iii) of the abstract currently reads that moving *within* the multiverse displaces a connectome. Add the anchor, keeping the existing numbers untouched since they are pilot results:

```
`[NOTE]` With the battery's reference combo in place, (iii) can additionally be stated
as displacement from DSI Studio's untouched defaults -- the out-of-the-box setting a
large share of published connectomes used without reporting it. Fill the magnitude in
once the battery has run; do not state it from the pilot cohort.
```

- [ ] **Step 4: Verify no other document still says 12**

Run: `grep -rn "12-candidate\|12 candidate" docs/ paper.md`
Expected: every remaining hit either refers to the historical 129/134 sweeps (correct, leave alone) or has been updated above.

- [ ] **Step 5: Commit**

```bash
git add docs/
git commit -m "$(cat <<'MSG'
docs(battery): add a DSI Studio defaults reference combo

The grid has no defaults cell, so nothing on disk represents the out-of-the-box
setting to measure displacement from. The reference runs per dataset and is
excluded from ranking.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 4: Build the viewer payload

**Files:**
- Create: `scripts/ordering_viewer.py`
- Test: `tests/test_ordering_viewer.py`

**Interfaces:**
- Consumes: Task 2's `reference` flag in each combo's `diagnostics.json`.
- Produces:
  - `MIN_SUBJECTS_FOR_ORDERING = 3`
  - `class ReferenceMissing(RuntimeError)`
  - `collect(sweep_optimize_dir: Path) -> dict` — the payload described below.

The payload:

```python
{
  "sweep_dir": str,
  "generated_at": str,             # ISO 8601, seconds
  "pairs": ["AAL3/count", ...],    # atlas/metric, the page's first dropdown
  "measures": ["density", ...],    # measures present for every combo
  "order_params": ["fa_threshold", ...],   # parameters that vary across candidates
  "waves": ["wave1", ...],
  "subjects": {"wave1": ["S01", "S02", ...]},   # fixed order; ranks index into this
  "reference": {"wave1": "wave1/sweep_0013", ...},
  "combos": [
    {
      "id": "wave1/sweep_0001",
      "wave": "wave1",
      "reference": False,
      "params": {"fa_threshold": 0.05, ...},
      # ranks of each subject under repeat 1, in `subjects[wave]` order.
      # Floats, because tied subjects get averaged ranks.
      "ranks": {"AAL3/count": {"density": [3.0, 1.0, 2.0, ...]}},
      "rho_noise": {"AAL3/count": {"density": 0.93}},
      "rho_vs_reference": {"AAL3/count": {"density": 0.66}},
    },
  ],
}
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ordering_viewer.py`:

```python
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from scripts.ordering_viewer import ReferenceMissing, collect

N = 16


def _matrix(rng, scale):
    """A symmetric matrix whose overall weight is set by `scale`, so subjects
    ordered by scale are ordered the same way on density-like measures."""
    mask = np.triu(rng.random((N, N)) < 0.4, 1)
    upper = np.where(mask, rng.random((N, N)) * scale, 0.0)
    return upper + upper.T


def _write_combo(root, wave, index, params, reference, scales, seed):
    combo = root / wave / "combos" / f"sweep_{index:04d}"
    for rep, rep_name in enumerate(("rep_1", "rep_2"), start=1):
        for s_i, (subject, scale) in enumerate(sorted(scales.items())):
            d = combo / rep_name / "01_connectivity"
            d.mkdir(parents=True, exist_ok=True)
            # Deterministic across processes: str hashing is salted, s_i is not.
            rng = np.random.default_rng(seed + rep * 100 + s_i)
            scipy.io.savemat(
                d / f"{subject}.AAL3..pass.connectivity.mat",
                {"connectivity": _matrix(rng, scale)},
            )
    (combo / "diagnostics.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "wave": wave,
                "combo_index": index,
                "sampler": "grid",
                "reference": reference,
                "parameters": params,
            }
        )
    )
    return combo


def _sweep(tmp_path, seed=0):
    scales = {f"sub-{i:02d}": 10.0 * i for i in range(1, 6)}
    _write_combo(tmp_path, "wave1", 1, {"fa_threshold": 0.05}, False, scales, seed)
    _write_combo(tmp_path, "wave1", 2, {"fa_threshold": 0.10}, False, scales, seed + 1)
    _write_combo(tmp_path, "wave1", 3, {"fa_threshold": 0.0}, True, scales, seed)
    return tmp_path


def test_collect_anonymizes_subjects(tmp_path):
    payload = collect(_sweep(tmp_path))
    assert payload["subjects"]["wave1"] == ["S01", "S02", "S03", "S04", "S05"]
    assert "sub-01" not in json.dumps(payload)


def test_collect_identifies_the_reference_per_wave(tmp_path):
    payload = collect(_sweep(tmp_path))
    assert payload["reference"]["wave1"] == "wave1/sweep_0003"
    flagged = [c["id"] for c in payload["combos"] if c["reference"]]
    assert flagged == ["wave1/sweep_0003"]


def test_combo_identical_to_reference_correlates_perfectly(tmp_path):
    """sweep_0001 and the reference are built from the same seed, so their
    subject ordering is identical and rho must be 1."""
    payload = collect(_sweep(tmp_path))
    combo = next(c for c in payload["combos"] if c["id"] == "wave1/sweep_0001")
    assert combo["rho_vs_reference"]["AAL3/count"]["density"] == pytest.approx(1.0)


def test_ranks_have_one_entry_per_subject(tmp_path):
    payload = collect(_sweep(tmp_path))
    combo = payload["combos"][0]
    ranks = combo["ranks"]["AAL3/count"]["density"]
    assert sorted(ranks) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_order_params_lists_only_parameters_that_vary(tmp_path):
    payload = collect(_sweep(tmp_path))
    assert payload["order_params"] == ["fa_threshold"]


def test_missing_reference_raises_naming_the_wave(tmp_path):
    scales = {f"sub-{i:02d}": 10.0 * i for i in range(1, 6)}
    _write_combo(tmp_path, "wave1", 1, {"fa_threshold": 0.05}, False, scales, 0)
    with pytest.raises(ReferenceMissing, match="wave1"):
        collect(tmp_path)


def test_too_few_subjects_raises(tmp_path):
    scales = {f"sub-{i:02d}": 10.0 * i for i in range(1, 3)}
    _write_combo(tmp_path, "wave1", 1, {"fa_threshold": 0.05}, False, scales, 0)
    _write_combo(tmp_path, "wave1", 2, {"fa_threshold": 0.0}, True, scales, 0)
    with pytest.raises(ValueError, match="3"):
        collect(tmp_path)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_ordering_viewer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.ordering_viewer'`.

- [ ] **Step 3: Write `collect()`**

Create `scripts/ordering_viewer.py`:

```python
"""Subject-ordering viewer: how far a parameter choice moves your subjects.

Emits one self-contained HTML file per sweep. A reader drags a slider across
candidate specifications and watches subject ranks on a graph measure reshuffle
against DSI Studio's untouched defaults, with the tracking-noise floor -- what
re-running the identical specification already costs -- shown alongside.

This is reporting, not selection: nothing here feeds scripts.reliability.rank().
See docs/superpowers/specs/2026-09-25-ordering-viewer-design.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from scripts.compute_network_measures_from_connectivity import measures_from_matrix
from scripts.variance_decomposition import collect_sweep_matrices

MIN_SUBJECTS_FOR_ORDERING = 3


class ReferenceMissing(RuntimeError):
    """No combo in a wave was flagged as the DSI Studio defaults reference."""


def _combo_meta(sweep_optimize_dir: Path) -> dict[str, dict]:
    """{combo_id: diagnostics.json}, keyed exactly as collect_sweep_matrices keys."""
    meta = {}
    for path in sorted(Path(sweep_optimize_dir).glob("*/combos/sweep_*/diagnostics.json")):
        combo_dir = path.parent
        meta[f"{combo_dir.parent.parent.name}/{combo_dir.name}"] = json.loads(path.read_text())
    return meta


def _references(meta: dict[str, dict], waves: list[str]) -> dict[str, str]:
    refs = {}
    for wave in waves:
        flagged = [cid for cid, m in meta.items()
                   if m.get("wave", cid.split("/")[0]) == wave and m.get("reference")]
        if not flagged:
            raise ReferenceMissing(
                f"{wave} has no combo flagged reference. Re-run the sweep with "
                f"sweep_parameters.reference_candidate set to DSI Studio's defaults; "
                f"displacement is meaningless without that origin."
            )
        refs[wave] = sorted(flagged)[0]
    return refs


def _varying(meta: dict[str, dict]) -> list[str]:
    """Parameters that take more than one value across non-reference combos."""
    seen: dict[str, set] = {}
    for m in meta.values():
        if m.get("reference"):
            continue
        for key, value in (m.get("parameters") or {}).items():
            seen.setdefault(key, set()).add(value)
    return sorted(k for k, v in seen.items() if len(v) > 1)


def collect(sweep_optimize_dir: Path) -> dict:
    """Viewer payload for a completed sweep. See the module docstring."""
    sweep_optimize_dir = Path(sweep_optimize_dir)
    grouped = collect_sweep_matrices(sweep_optimize_dir)
    if not grouped:
        raise ValueError(
            f"no */combos/sweep_* directories with connectivity matrices under {sweep_optimize_dir}"
        )
    meta = _combo_meta(sweep_optimize_dir)
    waves = sorted({cid.split("/")[0] for cid in meta})
    references = _references(meta, waves)

    # {wave: [anonymised label]}, and the scan key order the labels stand for.
    scan_keys: dict[str, list[str]] = {}
    for (atlas, metric), combos in grouped.items():
        for cid, scans in combos.items():
            wave = cid.split("/")[0]
            scan_keys.setdefault(wave, sorted(scans))
    for wave, keys in scan_keys.items():
        if len(keys) < MIN_SUBJECTS_FOR_ORDERING:
            raise ValueError(
                f"{wave} has {len(keys)} subject(s); rank correlation needs at least "
                f"{MIN_SUBJECTS_FOR_ORDERING}"
            )
    subjects = {w: [f"S{i:02d}" for i in range(1, len(k) + 1)] for w, k in scan_keys.items()}

    # measures[combo_id][pair][measure][repeat] -> [value per subject]
    values: dict[str, dict[str, dict[str, list[list[float]]]]] = {}
    measure_names: set[str] | None = None
    for (atlas, metric), combos in sorted(grouped.items()):
        pair = f"{atlas}/{metric}"
        for cid, scans in combos.items():
            keys = scan_keys[cid.split("/")[0]]
            if any(len(scans.get(k, [])) < 2 for k in keys):
                logging.warning("ordering viewer: %s lacks 2 repeats for some subject; skipped", cid)
                continue
            per_repeat = [[measures_from_matrix(scans[k][rep]) for k in keys] for rep in (0, 1)]
            names = set(per_repeat[0][0])
            measure_names = names if measure_names is None else measure_names & names
            slot = values.setdefault(cid, {}).setdefault(pair, {})
            for name in names:
                slot[name] = [[tbl[name] for tbl in rep] for rep in per_repeat]

    measures = sorted(measure_names or ())
    pairs = sorted({p for c in values.values() for p in c})

    combos_out = []
    for cid in sorted(values):
        wave = cid.split("/")[0]
        ref_id = references[wave]
        entry = {
            "id": cid,
            "wave": wave,
            "reference": bool(meta[cid].get("reference")),
            "params": meta[cid].get("parameters") or {},
            "ranks": {}, "rho_noise": {}, "rho_vs_reference": {},
        }
        for pair in pairs:
            if pair not in values[cid]:
                continue
            entry["ranks"][pair] = {}
            entry["rho_noise"][pair] = {}
            entry["rho_vs_reference"][pair] = {}
            for name in measures:
                rep1, rep2 = values[cid][pair][name]
                entry["ranks"][pair][name] = [float(r) for r in rankdata(rep1)]
                entry["rho_noise"][pair][name] = _rho(rep1, rep2)
                ref = values.get(ref_id, {}).get(pair, {}).get(name)
                entry["rho_vs_reference"][pair][name] = _rho(rep1, ref[0]) if ref else None
        combos_out.append(entry)

    return {
        "sweep_dir": str(sweep_optimize_dir),
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "pairs": pairs,
        "measures": measures,
        "order_params": _varying(meta),
        "waves": waves,
        "subjects": subjects,
        "reference": references,
        "combos": combos_out,
    }


def _rho(a: list[float], b: list[float]) -> float | None:
    """Spearman of two subject orderings; None when either is constant or too short."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < MIN_SUBJECTS_FOR_ORDERING:
        return None
    a, b = a[finite], b[finite]
    if a.std() == 0 or b.std() == 0:
        return None
    value = float(spearmanr(a, b)[0])
    return None if np.isnan(value) else value
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_ordering_viewer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/ordering_viewer.py tests/test_ordering_viewer.py
git commit -m "$(cat <<'MSG'
feat(viewer): build the subject-ordering payload from a sweep

Reuses collect_sweep_matrices + measures_from_matrix -- the path graph_icc
already walks, since per-subject measures are not stored on disk. Emits subject
ranks plus two precomputed Spearman correlations per candidate: the
tracking-noise floor, and displacement from the DSI Studio defaults reference.
Subject labels are anonymised unconditionally.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 5: Render the page and wire the CLI

**Files:**
- Modify: `scripts/ordering_viewer.py` (add `render`, `main`)
- Modify: `scripts/opticonn_hub.py` (add the `view` subparser beside `select` at line 69, and its dispatch beside `tune-grid` at line 785)
- Test: `tests/test_ordering_viewer.py`

**Interfaces:**
- Consumes: `collect()` from Task 4.
- Produces: `render(payload: dict) -> str`, and `opticonn view -i <optimize dir> [-o viewer.html]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ordering_viewer.py`:

```python
def test_render_inlines_the_payload_and_needs_no_network(tmp_path):
    from scripts.ordering_viewer import render

    html = render(collect(_sweep(tmp_path)))
    assert "<html" in html
    assert "wave1/sweep_0001" in html          # payload is inline
    # No network at load time. The SVG namespace string is an http:// URL and is
    # never fetched, so assert on the things that would actually issue a request.
    assert "fetch(" not in html
    assert "<script src" not in html
    assert "<link" not in html
    assert "cdn" not in html.lower()


def test_render_payload_round_trips(tmp_path):
    from scripts.ordering_viewer import render

    payload = collect(_sweep(tmp_path))
    html = render(payload)
    start = html.index("const PAYLOAD = ") + len("const PAYLOAD = ")
    end = html.index(";\n", start)
    assert json.loads(html[start:end])["reference"]["wave1"] == "wave1/sweep_0003"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_ordering_viewer.py -k render -v`
Expected: FAIL with `ImportError: cannot import name 'render'`.

- [ ] **Step 3: Add `render()` and `main()`**

Append to `scripts/ordering_viewer.py`:

```python
_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Subject ordering under parameter choice</title>
<style>
  :root { --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#bbb; --accent:#c2410c; --ok:#0369a1; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
    --bg:#15171a; --fg:#e8e8e8; --muted:#9aa0a6; --line:#555; --accent:#fb923c; --ok:#38bdf8; } }
  body { background:var(--bg); color:var(--fg); margin:0; padding:16px;
         font:15px/1.5 system-ui, -apple-system, sans-serif; }
  .wrap { max-width:900px; margin:0 auto; }
  .controls { display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin:16px 0; }
  select, input[type=range] { font:inherit; }
  .readout { display:flex; gap:24px; flex-wrap:wrap; margin:12px 0; }
  .readout div { font-variant-numeric:tabular-nums; }
  .big { font-size:1.6em; font-weight:600; }
  .muted { color:var(--muted); font-size:.85em; }
  svg { max-width:100%; height:auto; }
</style></head><body><div class="wrap">
<h1>How far does the parameter choice move your subjects?</h1>
<p class="muted">Left column: subject rank under DSI Studio's untouched defaults.
Right column: rank under the selected specification. Crossing lines are subjects the
choice re-ordered. <span id="src"></span></p>
<div class="controls">
  <label>Wave <select id="wave"></select></label>
  <label>Atlas / metric <select id="pair"></select></label>
  <label>Graph measure <select id="measure"></select></label>
  <label>Order by <select id="orderby"></select></label>
</div>
<div class="controls">
  <label style="flex:1">Specification
    <input type="range" id="slider" min="0" value="0" style="width:100%"></label>
</div>
<p id="params" class="muted"></p>
<div class="readout">
  <div><span class="big" id="rhoRef">-</span><br><span class="muted">vs. DSI Studio defaults</span></div>
  <div><span class="big" id="rhoNoise">-</span><br><span class="muted">same settings, re-run (noise floor)</span></div>
</div>
<svg id="slope" viewBox="0 0 640 420" role="img" aria-label="Subject rank slopegraph"></svg>
<h2>Every specification at once</h2>
<svg id="strip" viewBox="0 0 640 180" role="img" aria-label="Rank correlation per specification"></svg>
</div>
<script>
const PAYLOAD = __PAYLOAD__;
const $ = id => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const el = (n, a) => { const e = document.createElementNS(NS, n);
  for (const k in a) e.setAttribute(k, a[k]); return e; };

$("src").textContent = "Generated " + PAYLOAD.generated_at + ".";
const fill = (sel, vals) => { sel.innerHTML = "";
  vals.forEach(v => { const o = document.createElement("option");
    o.value = v; o.textContent = v; sel.appendChild(o); }); };
fill($("wave"), PAYLOAD.waves);
fill($("pair"), PAYLOAD.pairs);
fill($("measure"), PAYLOAD.measures);
fill($("orderby"), PAYLOAD.order_params.length ? PAYLOAD.order_params : ["(none)"]);

function candidates() {
  const wave = $("wave").value, key = $("orderby").value;
  const list = PAYLOAD.combos.filter(c => c.wave === wave && !c.reference);
  if (PAYLOAD.order_params.includes(key))
    list.sort((a, b) => (a.params[key] - b.params[key]) || a.id.localeCompare(b.id));
  return list;
}

function drawSlope(combo, pair, measure, subjects) {
  const svg = $("slope"); svg.innerHTML = "";
  const refId = PAYLOAD.reference[combo.wave];
  const ref = PAYLOAD.combos.find(c => c.id === refId);
  const a = ref.ranks[pair] && ref.ranks[pair][measure];
  const b = combo.ranks[pair] && combo.ranks[pair][measure];
  if (!a || !b) { svg.appendChild(el("text", {x:20, y:40, fill:"var(--muted)"}))
    .textContent = "measure unavailable for this specification"; return; }
  const n = a.length, top = 30, bottom = 400, xl = 120, xr = 520;
  const y = r => top + (bottom - top) * (r - 1) / Math.max(1, n - 1);
  svg.appendChild(el("text", {x:xl, y:16, fill:"var(--muted)", "text-anchor":"middle",
    "font-size":"13"})).textContent = "DSI Studio defaults";
  svg.appendChild(el("text", {x:xr, y:16, fill:"var(--muted)", "text-anchor":"middle",
    "font-size":"13"})).textContent = "selected specification";
  for (let i = 0; i < n; i++) {
    const moved = a[i] !== b[i];
    svg.appendChild(el("line", {x1:xl, y1:y(a[i]), x2:xr, y2:y(b[i]),
      stroke: moved ? "var(--accent)" : "var(--line)",
      "stroke-width": moved ? 2 : 1, "stroke-opacity": moved ? 0.9 : 0.45}));
    svg.appendChild(el("text", {x:xl-10, y:y(a[i])+4, "text-anchor":"end",
      fill:"var(--muted)", "font-size":"12"})).textContent = subjects[i];
    svg.appendChild(el("text", {x:xr+10, y:y(b[i])+4, fill:"var(--muted)",
      "font-size":"12"})).textContent = subjects[i];
  }
}

function drawStrip(list, current, pair, measure) {
  const svg = $("strip"); svg.innerHTML = "";
  const top = 20, bottom = 140, x0 = 60, x1 = 600;
  const y = rho => bottom - (bottom - top) * Math.max(0, Math.min(1, rho));
  const x = i => list.length < 2 ? (x0+x1)/2 : x0 + (x1-x0) * i / (list.length-1);
  [0, 0.5, 1].forEach(t => {
    svg.appendChild(el("line", {x1:x0, y1:y(t), x2:x1, y2:y(t),
      stroke:"var(--line)", "stroke-opacity":0.3}));
    svg.appendChild(el("text", {x:x0-10, y:y(t)+4, "text-anchor":"end",
      fill:"var(--muted)", "font-size":"12"})).textContent = t.toFixed(1);
  });
  const noise = list.map(c => (c.rho_noise[pair]||{})[measure]).filter(v => v !== null && v !== undefined);
  if (noise.length) {
    const lo = Math.min(...noise), hi = Math.max(...noise);
    svg.appendChild(el("rect", {x:x0, y:y(hi), width:x1-x0, height:Math.max(1, y(lo)-y(hi)),
      fill:"var(--ok)", "fill-opacity":0.15}));
    svg.appendChild(el("text", {x:x1, y:y(hi)-6, "text-anchor":"end", fill:"var(--ok)",
      "font-size":"12"})).textContent = "noise floor (same settings, re-run)";
  }
  let d = "";
  list.forEach((c, i) => {
    const v = (c.rho_vs_reference[pair]||{})[measure];
    if (v === null || v === undefined) return;
    d += (d ? " L" : "M") + x(i) + " " + y(v);
    svg.appendChild(el("circle", {cx:x(i), cy:y(v), r: c.id === current.id ? 6 : 3,
      fill:"var(--accent)"}));
  });
  if (d) svg.appendChild(el("path", {d, fill:"none", stroke:"var(--accent)", "stroke-width":1.5}));
  svg.appendChild(el("text", {x:x0, y:170, fill:"var(--muted)", "font-size":"12"}))
    .textContent = "each point is one specification, ordered by " + $("orderby").value;
}

function draw() {
  const list = candidates();
  $("slider").max = Math.max(0, list.length - 1);
  const combo = list[Math.min($("slider").value, list.length - 1)];
  if (!combo) return;
  const pair = $("pair").value, measure = $("measure").value;
  const fmt = v => (v === null || v === undefined) ? "n/a" : v.toFixed(2);
  $("rhoRef").textContent = fmt((combo.rho_vs_reference[pair]||{})[measure]);
  $("rhoNoise").textContent = fmt((combo.rho_noise[pair]||{})[measure]);
  $("params").textContent = Object.entries(combo.params)
    .map(([k, v]) => k + "=" + v).join(", ");
  drawSlope(combo, pair, measure, PAYLOAD.subjects[combo.wave]);
  drawStrip(list, combo, pair, measure);
}

["wave","pair","measure","orderby","slider"].forEach(id =>
  $(id).addEventListener("input", draw));
draw();
</script></body></html>
"""


def render(payload: dict) -> str:
    """One self-contained HTML page. No server, no CDN, no JavaScript dependency."""
    return _TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Self-contained subject-ordering viewer for a completed OptiConn sweep"
    )
    parser.add_argument("-i", "--input", required=True, help="a sweep's optimize/ directory")
    parser.add_argument("-o", "--output", default=None,
                        help="output HTML (default: <input>/ordering_viewer.html)")
    args = parser.parse_args(argv)

    out = Path(args.output) if args.output else Path(args.input) / "ordering_viewer.html"
    try:
        payload = collect(Path(args.input))
    except (ReferenceMissing, ValueError) as exc:
        logging.error("ordering viewer: %s", exc)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(payload), encoding="utf-8")
    logging.info("Ordering viewer written to %s", out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_ordering_viewer.py -v`
Expected: PASS.

- [ ] **Step 5: Wire the hub subcommand**

In `scripts/opticonn_hub.py`, beside the `p_select` subparser (line 69), add:

```python
    p_view = subparsers.add_parser(
        "view", help="Build a self-contained subject-ordering viewer from a completed sweep"
    )
    p_view.add_argument("-i", "--input", required=True, help="a sweep's optimize/ directory")
    p_view.add_argument("-o", "--output", default=None, help="output HTML path")
```

and beside the `tune-grid` dispatch (line 785), add:

```python
    if args.command == "view":
        from scripts.ordering_viewer import main as view_main

        argv = ["-i", args.input]
        if args.output:
            argv += ["-o", args.output]
        return view_main(argv)
```

- [ ] **Step 6: Verify the CLI surface**

Run: `OPTICONN_SKIP_VENV=1 python opticonn.py view --help`
Expected: the help text for `view`, exit 0.

Run: `python -m pytest tests/test_cli_help.py -v`
Expected: PASS. If that test asserts an exact list of subcommands, add `view` to it.

- [ ] **Step 7: Commit**

```bash
git add scripts/ordering_viewer.py scripts/opticonn_hub.py tests/
git commit -m "$(cat <<'MSG'
feat(viewer): render the ordering page and add `opticonn view`

One self-contained HTML: slopegraph of subject ranks against the DSI Studio
defaults reference, the noise floor beside it, and a strip showing every
specification at once. No server, no CDN, no JavaScript dependency -- it opens
over file://.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 6: Verify on real data and document

The tool must be exercised against a real sweep before it is trusted. `studies/study129/cross_sectional_v1/optimize` has 12 candidates and no reference combo, so it will correctly raise `ReferenceMissing` — that is the first thing to confirm, since it is the failure a user hits most often.

**Files:**
- Modify: `README.md`
- Modify: `docs/` (the page describing sweep outputs, if one exists — find it with the grep in Step 3)

**Interfaces:**
- Consumes: `opticonn view` from Task 5.

- [ ] **Step 1: Confirm the missing-reference error is legible on real data**

Run: `OPTICONN_SKIP_VENV=1 python opticonn.py view -i studies/study129/cross_sectional_v1/optimize`
Expected: exit 1, with an error naming `wave1` and telling the user to set `sweep_parameters.reference_candidate`. No traceback.

- [ ] **Step 2: Confirm the payload builds on real matrices**

Run:

```bash
python - <<'EOF'
import json
from pathlib import Path
from scripts.ordering_viewer import collect, render
import scripts.ordering_viewer as ov

# Treat the lowest-numbered combo as the reference, to exercise the real path
# end to end on a sweep that predates the reference combo.
real = ov._references
ov._references = lambda meta, waves: {
    w: sorted(c for c in meta if c.startswith(w + "/"))[0] for w in waves
}
p = collect(Path("studies/study129/cross_sectional_v1/optimize"))
ov._references = real
print("pairs:", p["pairs"])
print("measures:", p["measures"])
print("order_params:", p["order_params"])
print("combos:", len(p["combos"]), "subjects:", p["subjects"])
Path("/tmp/claude-1002/ordering_check.html").write_text(render(p))
print("html bytes:", Path("/tmp/claude-1002/ordering_check.html").stat().st_size)
EOF
```

Expected: non-empty `pairs` and `measures`, `order_params` listing the parameters the grid varied, 12 combos, and an HTML file well under 1 MB. Open it and confirm the slider moves and lines cross. **Do not commit this HTML** — it is built from the private cohort.

- [ ] **Step 3: Document the subcommand**

Find where the other subcommands are documented:

Run: `grep -rln "opticonn select\|tune-grid" README.md docs/`

In `README.md`, after the three-step workflow block near the top, add:

```markdown
### Seeing how much the choice mattered

```bash
python opticonn.py view -i studies/demo_grid/sweep-*/optimize
```

Writes one self-contained HTML file. Drag the slider across the screened
specifications and watch your subjects re-order on a graph measure, against
DSI Studio's untouched defaults, with the tracking-noise floor beside it —
what re-running the identical settings already costs. Opens in a browser with
no server.

This requires the sweep to have been run with `sweep_parameters.reference_candidate`
set, since displacement is measured from that origin:

```json
"reference_candidate": { "fa_threshold": 0.0, "turning_angle": 0.0, "step_size": 0.0 }
```
```

- [ ] **Step 4: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: all tests pass, including the 174 pre-existing ones.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/
git commit -m "$(cat <<'MSG'
docs: document `opticonn view` and the reference_candidate it needs

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

## Deferred until the battery runs

The committed public demo is **not** part of this plan. Once the OpenNeuro battery has
run with the Task 3 config, generate the demo from one public dataset's output and commit
that single HTML file. Nothing derived from study129 or study134 is committed.

## Deliberately excluded

Per the spec: no PCA view, no server or bundler, no edge-level display, no interactive
re-ranking. Add the PCA view only if the slopegraph turns out not to carry the argument.
