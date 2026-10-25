# Sensitivity report (`opticonn report`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `opticonn report`, a sweep-level sensitivity report (leader test, fragility, effective dimensionality, displacement from the reference, measure-vs-density relation) that reproduces the study repository's committed numbers.

**Architecture:** `measures` gains `--wave all` (wave-scoped specification ids) and a cache-reuse check; the statistics are pure functions in `scripts/report_stats.py`; `scripts/report.py` builds the per-subject tables for repeats 1 and 2 (reusing cached ones), applies the sweep's own `rejected` gating, calls the statistics and writes `report.json` and `report.csv`. Registered in the hub like `measures`/`outcome`/`pca`.

**Tech Stack:** Python 3, numpy, scipy (already OptiConn dependencies), pytest.

Design: `docs/superpowers/specs/2026-10-25-sensitivity-report-design.md`. Work in the worktree `/data/local/software/opticonn-wt-report` on branch `feature/sensitivity-report`; run tests from its root with `python -m pytest`.

## Global Constraints

- `MIN_SUBJECTS = 3` everywhere a statistic needs a subject floor.
- Pooled statistics (fragility, effective dimensionality, displacement, reference rho, density relation) drop specifications whose `rejected` field in their wave's `combo_diagnostics.csv` is non-empty. The leader test does not use that flag: it uses non-missing `discriminability_margin`.
- Fragility and effective dimensionality pool the reference as one more specification; the leader test, displacement and `reference_rho` treat it separately.
- A statistic that cannot be computed is reported as absent with a stated reason, never as an error.
- No NaN in `report.json` (NaN and infinity are written as `null`).
- Specification ids are wave-scoped (`<wave>/sweep_NNNN`) when a sweep has more than one wave, plain otherwise.
- Only `scripts/measures_table.py` and `scripts/opticonn_hub.py` among existing files change (plus `docs/outcome_sensitivity.md`, `README.md`, `tests/sweep_fixture.py`, `tests/test_measures_table.py`).
- Every step that changes behaviour starts with a failing test.
- Commit messages end with the line `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## File Structure

- Modify `scripts/measures_table.py`: `find_waves`, wave-scoped ids, `tables_current`, `--wave all`.
- Create `scripts/report_stats.py`: `wave_of`, `leader_test`, `density_relation`, `reference_percentile`, `fragility`, `effective_dimensionality`, `reference_rho`, `reference_displacement`.
- Create `scripts/report.py`: `build_report`, `flat_row`, `main`.
- Modify `scripts/opticonn_hub.py`: register `report`.
- Modify `tests/sweep_fixture.py`: repeats, diagnostics.
- Create `tests/test_report_stats.py`, `tests/test_report.py`, `tests/test_report_acceptance.py`.
- Modify `docs/outcome_sensitivity.md`, `README.md`.

---

### Task 1: `measures --wave all`, the fixture, and the cache check

**Files:**
- Modify: `tests/sweep_fixture.py`, `scripts/measures_table.py`, `tests/test_measures_table.py`

**Interfaces:**
- Produces: `find_waves(optimize_dir, wave=None) -> list[Path]` (`"all"` returns every wave; anything else returns `[find_wave(...)]`); `read_meta_waves(waves) -> dict` (ids wave-scoped when more than one wave); `tables_current(out_dir, atlas, metric, repeat, wave, n_specifications) -> bool`; `write_measures(...)` accepts `wave="all"`; `measures.json` gains `"waves"` (list of names) and `"wave_setting"` (the requested wave string or `null`), and keeps `"wave"` (the single wave's name, or `"all"` for several waves).
- Produces (fixture): `make_sweep(root, specs, n_subjects=15, wave="wave1", metrics=("count",), base_edges=40, repeats=1, margin=None)`. With `repeats=k`, files `rep_1`..`rep_k` are written (the random edges differ per repeat, the edge counts do not). With `margin` a callable `margin(index) -> float`, a `combo_diagnostics.csv` is written in `<root>/<wave>/` with columns `sweep_id,reference,discriminability,discriminability_margin,rejected`: margin blank and `rejected` = `"density 0.87 outside [0.02, 0.6]"` for specifications whose edge offset is 50 or more, otherwise the margin and an empty `rejected`; `discriminability` 1.0; `reference` `"True"`/`"False"`. `add_session` is kept unchanged.

- [ ] **Step 1: Replace `tests/sweep_fixture.py` with the extended fixture**

```python
# tests/sweep_fixture.py
"""A tiny synthetic sweep in the layout OptiConn's collect_matrices reads (no DSI Studio needed)."""

import csv
import json
from pathlib import Path

import numpy as np
import scipy.io

N = 16


def _matrix(rng, n_edges):
    iu = np.triu_indices(N, k=1)
    chosen = rng.permutation(len(iu[0]))[:n_edges]
    w = rng.random(n_edges) + 0.5
    m = np.zeros((N, N))
    m[iu[0][chosen], iu[1][chosen]] = w
    m[iu[1][chosen], iu[0][chosen]] = w
    return m


def make_sweep(root, specs, n_subjects=15, wave="wave1", metrics=("count",), base_edges=40,
               repeats=1, margin=None):
    """specs: [(parameters, reference, edge_offset)]. Subject i has base_edges + offset + 2*i edges
    in every specification and repeat, so density (and the rank of every subject on it) is known
    exactly; the random edges differ between repeats. Edge counts are capped by the 120 possible
    edges at N=16: keep base_edges + offset + 2*(n_subjects - 1) <= 120. An offset of 50 with <= 15
    subjects gives mean density about 0.87, above the 0.6 gate.

    margin: optional callable index -> discriminability margin; if given, a combo_diagnostics.csv is
    written for the wave (specifications with offset >= 50 are marked rejected, margin blank)."""
    diag = []
    for index, (params, reference, offset) in enumerate(specs, start=1):
        combo = Path(root) / wave / "combos" / f"sweep_{index:04d}"
        for rep in range(1, repeats + 1):
            for i in range(n_subjects):
                rng = np.random.default_rng(1000 * index + i + 100000 * (rep - 1))
                d = combo / f"rep_{rep}" / "AAL3"
                d.mkdir(parents=True, exist_ok=True)
                for metric in metrics:
                    scipy.io.savemat(d / f"sub-{i + 1:02d}_ses-1_dwi.qsdr.{metric}..pass.connectivity.mat",
                                     {"connectivity": _matrix(rng, base_edges + offset + 2 * i)})
        (combo / "diagnostics.json").write_text(json.dumps(
            {"status": "ok", "wave": wave, "combo_index": index, "sampler": "grid",
             "reference": reference, "parameters": params}))
        rejected = offset >= 50
        diag.append({"sweep_id": f"sweep_{index:04d}", "reference": str(bool(reference)),
                     "discriminability": 1.0,
                     "discriminability_margin": "" if rejected or margin is None else margin(index),
                     "rejected": "density 0.87 outside [0.02, 0.6]" if rejected else ""})
    if margin is not None:
        with open(Path(root) / wave / "combo_diagnostics.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(diag[0]))
            writer.writeheader()
            writer.writerows(diag)
    return Path(root)


def add_session(measures_dir):
    """Append a second scan of the first subject of the first specification to measures.csv."""
    path = Path(measures_dir) / "measures.csv"
    lines = path.read_text().splitlines()
    first = lines[1].split(",")
    first[2] = first[2] + "_again"
    path.write_text("\n".join(lines + [",".join(first)]) + "\n")
```

- [ ] **Step 2: Append the failing tests to `tests/test_measures_table.py`**

```python
# append to tests/test_measures_table.py
from scripts.measures_table import find_waves, tables_current


def test_wave_all_includes_every_wave_with_wave_scoped_ids(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=5, wave="wave1")
    make_sweep(tmp_path / "opt", SPECS, n_subjects=5, wave="wave2")
    out = write_measures(tmp_path / "opt", wave="all", workers=1)
    rows = read_measures(out)
    assert {r["combo_id"] for r in rows} == {f"{w}/sweep_000{i}" for w in ("wave1", "wave2") for i in (1, 2, 3, 4)}
    assert len(rows) == 2 * 4 * 5
    params = read_params(out)
    assert params["wave2/sweep_0004"]["reference"] is True and params["wave1/sweep_0001"]["reference"] is False
    side = json.loads((out / "measures.json").read_text())
    assert side["wave"] == "all" and side["waves"] == ["wave1", "wave2"] and side["wave_setting"] == "all"
    assert [w.name for w in find_waves(tmp_path / "opt", "all")] == ["wave1", "wave2"]


def test_wave_all_on_a_single_wave_keeps_the_plain_ids(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=5)
    rows = read_measures(write_measures(tmp_path / "opt", wave="all", workers=1))
    assert {r["combo_id"] for r in rows} == {"sweep_0001", "sweep_0002", "sweep_0003", "sweep_0004"}


def test_tables_are_current_only_for_the_settings_they_were_written_with(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=5)
    out = write_measures(tmp_path / "opt", wave="all", repeat=1, workers=1)
    assert tables_current(out, None, None, 1, "all", 4)
    assert tables_current(out, "AAL3", "count", 1, "all", 4)
    assert not tables_current(out, None, None, 2, "all", 4)          # another repeat
    assert not tables_current(out, None, None, 1, None, 4)           # another wave setting
    assert not tables_current(out, None, None, 1, "all", 5)          # the sweep grew
    assert not tables_current(out, None, "qa", 1, "all", 4)          # another metric
    assert not tables_current(tmp_path / "nowhere", None, None, 1, "all", 4)


def test_the_fixture_writes_repeats_and_a_diagnostics_table(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=5, repeats=2, margin=lambda i: 0.1 * i)
    assert len(read_measures(write_measures(tmp_path / "opt", repeat=2, workers=1))) == 4 * 5
    with open(tmp_path / "opt" / "wave1" / "combo_diagnostics.csv") as handle:
        diag = list(csv.DictReader(handle))
    assert [d["reference"] for d in diag] == ["False", "False", "False", "True"]
    assert diag[2]["discriminability_margin"] == "" and "outside" in diag[2]["rejected"]   # offset 60
    assert float(diag[1]["discriminability_margin"]) == 0.2 and diag[1]["rejected"] == ""
```

Note: `test_measures_table.py` already imports `csv`, `json`, `make_sweep` and `SPECS` (offsets 0, 0, 60, reference 0); also import `read_params` in the existing import line if it is not there (`from scripts.measures_table import ... read_params ...` already includes it).

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_measures_table.py -q`
Expected: the four new tests fail (`ImportError: cannot import name 'find_waves'`), the existing ones still pass.

- [ ] **Step 4: Implement in `scripts/measures_table.py`**

Add after `find_wave`:

```python
def find_waves(optimize_dir, wave: str | None = None) -> list[Path]:
    """The wave directories a command works on: every wave for `all`, otherwise the one `find_wave` picks."""
    if wave == "all":
        waves = sorted({p.parent for p in Path(optimize_dir).glob("*/combos")})
        if not waves:
            raise ValueError(f"{optimize_dir}: no waves found")
        return waves
    return [find_wave(optimize_dir, wave)]
```

Add after `read_meta`:

```python
def read_meta_waves(waves: list[Path]) -> dict:
    """read_meta over several waves; ids are wave-scoped (`wave1/sweep_0001`) when there is more than one."""
    scoped = len(waves) > 1
    return {(f"{w.name}/{cid}" if scoped else cid): m for w in waves for cid, m in read_meta(w).items()}
```

In `write_measures`, replace the wave and combo discovery and the later uses of `wave_dir`/`combos`/`meta`:

```python
    optimize_dir = Path(optimize_dir)
    waves = find_waves(optimize_dir, wave)
    scoped = len(waves) > 1
    combos = [(f"{w.name}/{c.name}" if scoped else c.name, c)
              for w in waves for c in sorted((w / "combos").glob("sweep_*"))]
    if not combos:
        raise ValueError(f"{optimize_dir}: no specifications found")
    pairs = next((sorted(m) for m in map(collect_matrices, (c for _, c in combos)) if m), None)
```
(the `pairs is None`, `chosen` and error-message lines are unchanged), then

```python
    jobs = [(name, c, atlas, metric, repeat, extended) for name, c in combos]
```
and `meta = read_meta_waves(waves)`, and the sidecar becomes:

```python
    (out / "measures.json").write_text(json.dumps(
        {"atlas": atlas, "metric": metric, "repeat": repeat, "extended": extended,
         "n_specifications": len(combos), "n_specifications_without_matrices": missing,
         "wave": waves[0].name if len(waves) == 1 else "all", "waves": [w.name for w in waves],
         "wave_setting": wave, "opticonn_version": ver}, indent=2) + "\n")
```

Add after `read_params`:

```python
def tables_current(out_dir, atlas, metric, repeat, wave, n_specifications) -> bool:
    """True if out_dir holds measure tables written with these settings for a sweep of this size."""
    out = Path(out_dir)
    if not ((out / "measures.csv").exists() and (out / "measures.json").exists()):
        return False
    s = json.loads((out / "measures.json").read_text())
    return (atlas in (None, s["atlas"]) and metric in (None, s["metric"]) and s["repeat"] == repeat
            and s.get("wave_setting") == wave and s["n_specifications"] == n_specifications)
```

In `main`, change the `--wave` argument: `ap.add_argument("--wave", default=None, help="a wave name, or 'all' for every wave (ids are then wave-scoped)")`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_measures_table.py tests/test_outcome_curve.py tests/test_pca_curve.py tests/test_outcome_cli.py -q -W error`
Expected: all pass (the existing tests are unaffected: single-wave ids and the sidecar `wave` value are unchanged).

- [ ] **Step 6: Commit**

```bash
git add tests/sweep_fixture.py scripts/measures_table.py tests/test_measures_table.py
git commit -m "feat(measures): --wave all with wave-scoped ids; cache check; fixture with repeats and diagnostics

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Statistics from margins and densities

**Files:**
- Create: `scripts/report_stats.py` (this task adds the first functions)
- Test: `tests/test_report_stats.py`

**Interfaces:**
- Produces: `MIN_SUBJECTS = 3`; `wave_of(combo_id) -> str` (the part before `/`, or `""` for an unscoped id); `leader_test(m1, m2, discr1) -> dict` where `m1`, `m2` map sweep id to a margin (float, `None` or NaN for missing) for waves 1 and 2 (reference rows already removed) and `discr1` is the list of non-missing wave-1 discriminability values; returns the keys `n_candidates, discriminability_saturated, cross_wave_rho, winner_wave1, winner_wave2, winners_agree, top_two_separation, wave_variation, winner_determined` (value `None` where undefined) plus `notes` (list of str); `density_relation(means, reference_ids) -> dict` where `means` is `{spec_id: {measure: float}}` containing `"density"`; returns `{"<measure>_vs_density": float | None}` for every other measure; `reference_percentile(density_by_wave, reference_by_wave) -> float | None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_report_stats.py
import numpy as np
import pytest

from scripts.report_stats import density_relation, leader_test, reference_percentile, wave_of


def test_wave_of_splits_a_scoped_id_and_leaves_a_plain_id_alone():
    assert wave_of("wave1/sweep_0003") == "wave1" and wave_of("sweep_0003") == ""


def test_a_determined_leader_has_a_gap_larger_than_the_wave_to_wave_variation():
    m1 = {"a": 0.90, "b": 0.50, "c": 0.30, "d": 0.10}
    m2 = {"a": 0.88, "b": 0.52, "c": 0.31, "d": 0.09}
    got = leader_test(m1, m2, [1.0, 1.0])
    assert got["n_candidates"] == 4 and got["discriminability_saturated"] is True
    assert got["cross_wave_rho"] == pytest.approx(1.0)
    assert got["winner_wave1"] == got["winner_wave2"] == "a" and got["winners_agree"] is True
    assert got["top_two_separation"] == pytest.approx(0.38) and got["wave_variation"] == pytest.approx(0.015)
    assert got["winner_determined"] is True and got["notes"] == []


def test_an_undetermined_leader_and_disagreeing_winners():
    m1 = {"a": 0.300, "b": 0.299, "c": 0.100}
    m2 = {"a": 0.250, "b": 0.251, "c": 0.050}
    got = leader_test(m1, m2, [1.0, 0.9])
    assert got["winners_agree"] is False and got["winner_determined"] is False
    assert got["discriminability_saturated"] is False
    assert got["top_two_separation"] < got["wave_variation"]


def test_a_candidate_with_a_missing_margin_is_excluded_and_few_candidates_skip_the_rest():
    got = leader_test({"a": 0.3, "b": 0.2, "c": None}, {"a": 0.3, "b": float("nan"), "c": 0.1}, [])
    assert got["n_candidates"] == 1 and got["winner_wave1"] is None and got["winner_determined"] is None
    assert got["discriminability_saturated"] is None
    assert any("fewer than 3" in n for n in got["notes"]) and any("discriminability" in n for n in got["notes"])


def test_a_constant_ordering_has_no_cross_wave_rho_but_still_a_winner():
    got = leader_test({"a": 0.2, "b": 0.2, "c": 0.2}, {"a": 0.1, "b": 0.2, "c": 0.3}, [1.0])
    assert got["cross_wave_rho"] is None and any("constant" in n for n in got["notes"])
    assert got["winner_wave1"] == "a" and got["winner_wave2"] == "c"        # ties go to the first


def test_density_relation_is_the_spearman_of_each_measure_against_density_over_kept_specifications():
    means = {f"s{i}": {"density": 0.1 * i, "sw": 5 - i, "flat": 1.0} for i in range(1, 6)}
    means["ref"] = {"density": 0.9, "sw": 100.0, "flat": 1.0}
    got = density_relation(means, {"ref"})
    assert got["sw_vs_density"] == pytest.approx(-1.0)
    assert got["flat_vs_density"] is None and "density_vs_density" not in got


def test_reference_percentile_is_the_mean_over_waves_of_the_share_of_less_dense_specifications():
    by_wave = {"w1": {"a": 0.1, "b": 0.2, "c": 0.3, "ref": 0.25}, "w2": {"a": 0.1, "b": 0.2, "ref": 0.5}}
    got = reference_percentile(by_wave, {"w1": "ref", "w2": "ref"})
    assert got == pytest.approx((2 / 3 + 1.0) / 2)
    assert reference_percentile(by_wave, {}) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_report_stats.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'scripts.report_stats'`.

- [ ] **Step 3: Implement**

```python
# scripts/report_stats.py
"""Sweep-level sensitivity statistics as pure functions (no I/O); `opticonn report` assembles them.

Design: docs/superpowers/specs/2026-10-25-sensitivity-report-design.md
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr

MIN_SUBJECTS = 3


def wave_of(combo_id: str) -> str:
    """The wave part of a wave-scoped specification id; empty for a plain id."""
    return combo_id.split("/", 1)[0] if "/" in combo_id else ""


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def leader_test(m1: dict, m2: dict, discr1: list) -> dict:
    """Is the leading specification determined? Margins of the two waves, reference excluded."""
    out = {k: None for k in ("n_candidates", "discriminability_saturated", "cross_wave_rho", "winner_wave1",
                             "winner_wave2", "winners_agree", "top_two_separation", "wave_variation",
                             "winner_determined")}
    notes = []
    shared_all = sorted(set(m1) & set(m2))
    shared = [s for s in shared_all if not _missing(m1[s]) and not _missing(m2[s])]
    out["n_candidates"] = len(shared)
    if len(shared_all) > len(shared):
        notes.append(f"{len(shared_all) - len(shared)} candidate(s) excluded: discriminability_margin blank or NaN")
    if discr1:
        out["discriminability_saturated"] = all(d == 1.0 for d in discr1)
    else:
        notes.append("discriminability_saturated unavailable: no gradeable discriminability in wave 1")
    if len(shared) < 3:
        notes.append(f"fewer than 3 shared candidates ({len(shared)})")
    else:
        a, b = [m1[s] for s in shared], [m2[s] for s in shared]
        if len(set(a)) == 1 or len(set(b)) == 1:
            notes.append("cross_wave_rho unavailable: the candidate ordering was constant across candidates")
        else:
            out["cross_wave_rho"] = float(spearmanr(a, b)[0])
        out["winner_wave1"] = max(shared, key=lambda s: m1[s])
        out["winner_wave2"] = max(shared, key=lambda s: m2[s])
        out["winners_agree"] = out["winner_wave1"] == out["winner_wave2"]
        mean = {s: (m1[s] + m2[s]) / 2 for s in shared}
        top = sorted(mean, key=lambda s: -mean[s])[:2]
        out["top_two_separation"] = abs(mean[top[0]] - mean[top[1]])
        out["wave_variation"] = sum(abs(m1[s] - m2[s]) for s in shared) / len(shared)
        out["winner_determined"] = bool(out["top_two_separation"] > out["wave_variation"])
    out["notes"] = notes
    return out


def density_relation(means: dict, reference_ids: set) -> dict:
    """Spearman of each measure's specification mean against density, over the non-reference specifications."""
    kept = sorted(s for s in means if s not in reference_ids)
    dens = np.array([means[s]["density"] for s in kept], dtype=float)
    out = {}
    for name in sorted({n for s in kept for n in means[s]} - {"density"}):
        vals = np.array([means[s][name] for s in kept], dtype=float)
        ok = np.isfinite(vals) & np.isfinite(dens)
        usable = ok.sum() > 2 and np.ptp(vals[ok]) > 0 and np.ptp(dens[ok]) > 0
        out[f"{name}_vs_density"] = float(spearmanr(vals[ok], dens[ok])[0]) if usable else None
    return out


def reference_percentile(density_by_wave: dict, reference_by_wave: dict) -> float | None:
    """Mean over waves of the share of a wave's other specifications that are less dense than its reference."""
    shares = []
    for wave, dens in density_by_wave.items():
        ref = reference_by_wave.get(wave)
        if ref is None or ref not in dens:
            continue
        others = [d for s, d in dens.items() if s != ref]
        if others:
            shares.append(float(np.mean([d < dens[ref] for d in others])))
    return float(np.mean(shares)) if shares else None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_report_stats.py -q -W error`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/report_stats.py tests/test_report_stats.py
git commit -m "feat(report): leader test, density relation and reference percentile

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Fragility, effective dimensionality, reference rho and displacement

**Files:**
- Modify: `scripts/report_stats.py` (append)
- Test: `tests/test_report_stats.py` (append)

**Interfaces:**
- Consumes: `MIN_SUBJECTS`, `wave_of` (Task 2); `scripts.reliability._distance`; `scripts.utils.discovery.parse_subject_session`.
- Produces:
  - `fragility(per) -> dict`, `effective_dimensionality(per) -> dict` where `per = {combo_id: {scan: [measures_repeat1, measures_repeat2]}}` (each `measures_*` a `{measure: float}` dict; only scans with both repeats);
  - `reference_rho(rep1, reference_ids) -> dict` where `rep1 = {combo_id: {scan: {measure: float}}}`;
  - `reference_displacement(vectors, reference_ids) -> dict` where `vectors = {combo_id: {scan: ndarray}}` (repeat-1 edge vectors).
  Returned keys: `fragility`: `seed_rho, param_rho, gap, n_measures, n_seed_pairs, n_param_pairs, n_seed_measures, n_param_measures, min_n_subjects`; `effective_dimensionality`: `effective_dimensionality, n_ed_rows, n_ed_subjects, n_ed_dropped`; `reference_rho`: `reference_rho, n_reference_ordering_pairs, note`; `reference_displacement`: `available, reference_over_between_subject, n_reference_pairs, note`.

- [ ] **Step 1: Append the failing tests**

```python
# append to tests/test_report_stats.py
from scripts.report_stats import effective_dimensionality, fragility, reference_displacement, reference_rho


def _scans(values, scale=1.0):
    return {f"sub-{i + 1:02d}_ses-1_dwi": {"m": v * scale} for i, v in enumerate(values)}


def test_fragility_contrasts_repeats_of_one_specification_with_different_specifications():
    up = [1.0, 2.0, 3.0, 4.0, 5.0]
    per = {"w1/s1": {k: [m, dict(m)] for k, m in _scans(up).items()},                 # repeats identical
           "w1/s2": {k: [m, dict(m)] for k, m in _scans(up[::-1]).items()},          # reversed ordering
           "w2/s1": {k: [m, dict(m)] for k, m in _scans(up).items()}}                # alone in its wave
    got = fragility(per)
    assert got["seed_rho"] == pytest.approx(1.0) and got["param_rho"] == pytest.approx(-1.0)
    assert got["gap"] == pytest.approx(2.0)
    assert got["n_param_pairs"] == 1                   # w1/s1 with w1/s2 only: no pairing across waves
    assert got["n_seed_pairs"] == 3 and got["n_measures"] == 1 and got["min_n_subjects"] == 5


def test_fragility_skips_specifications_with_too_few_subjects():
    per = {"w1/s1": {k: [m, dict(m)] for k, m in _scans([1.0, 2.0]).items()}}
    got = fragility(per)
    assert np.isnan(got["seed_rho"]) and got["n_seed_pairs"] == 0


def _two_repeat(values_by_measure, n):
    rng = np.random.default_rng(0)
    per = {}
    for c in range(n // 15):
        per[f"w1/s{c}"] = {f"sub-{i:02d}_ses-1_dwi": [{k: float(v[c * 15 + i]) for k, v in values_by_measure.items()}] * 2
                           for i in range(15)}
    return per


def test_effective_dimensionality_is_one_for_perfectly_correlated_measures_and_the_count_for_independent_ones():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(600)
    corr = {"a": x, "b": 2 * x + 1, "c": -x}
    indep = {"a": rng.standard_normal(600), "b": rng.standard_normal(600), "c": rng.standard_normal(600)}
    assert effective_dimensionality(_two_repeat(corr, 600))["effective_dimensionality"] == pytest.approx(1.0, abs=1e-6)
    got = effective_dimensionality(_two_repeat(indep, 600))
    assert got["effective_dimensionality"] == pytest.approx(3.0, rel=0.1)
    assert got["n_ed_rows"] == 600 and got["n_ed_subjects"] == 15 and got["n_ed_dropped"] == 0


def test_effective_dimensionality_drops_non_finite_rows_and_needs_enough_subjects():
    per = _two_repeat({"a": np.arange(30.0), "b": np.arange(30.0) ** 2}, 30)
    first = next(iter(per["w1/s0"]))
    per["w1/s0"][first] = [{"a": float("nan"), "b": 1.0}] * 2
    assert effective_dimensionality(per)["n_ed_dropped"] == 1
    tiny = {"w1/s0": {f"sub-0{i}_ses-1_dwi": [{"a": float(i), "b": 1.0}] * 2 for i in range(1, 3)}}
    assert np.isnan(effective_dimensionality(tiny)["effective_dimensionality"])


def test_reference_rho_is_the_agreement_of_each_candidate_with_the_reference_in_its_wave():
    ref = _scans([1.0, 2.0, 3.0, 4.0])
    rep1 = {"w1/ref": ref, "w1/s1": _scans([1.0, 2.0, 3.0, 4.0], 2.0), "w1/s2": _scans([4.0, 3.0, 2.0, 1.0]),
            "w2/s1": _scans([1.0, 2.0, 3.0, 4.0])}                       # its wave has no reference
    got = reference_rho(rep1, {"w1/ref"})
    assert got["reference_rho"] == pytest.approx(0.0) and got["n_reference_ordering_pairs"] == 2
    assert reference_rho({"w1/s1": ref}, set())["reference_rho"] != reference_rho({"w1/s1": ref}, set())["reference_rho"]  # NaN
    assert "no reference" in reference_rho({"w1/s1": ref}, set())["note"]


def test_displacement_is_the_distance_to_the_reference_relative_to_the_between_subject_distance():
    s1, s2 = np.array([1.0, 2.0, 3.0, 4.0]), np.array([4.0, 3.0, 2.0, 1.0])
    c1, c2 = np.array([1.0, 2.0, 4.0, 3.0]), np.array([4.0, 3.0, 1.0, 2.0])
    k1, k2 = "sub-01_ses-1_dwi", "sub-02_ses-1_dwi"
    vectors = {"w1/ref": {k1: s1, k2: s2}, "w1/cand": {k1: c1, k2: c2}}
    got = reference_displacement(vectors, {"w1/ref"})
    # candidate vs reference: 1 - 0.8 = 0.2 for both subjects; between subjects: 1 - (-1) = 2 in both specifications
    assert got["available"] is True and got["n_reference_pairs"] == 2
    assert got["reference_over_between_subject"] == pytest.approx(0.1)


def test_displacement_without_a_reference_is_reported_as_unavailable():
    got = reference_displacement({"w1/s1": {"sub-01_ses-1_dwi": np.array([1.0, 2.0, 3.0])}}, set())
    assert got["available"] is False and got["reference_over_between_subject"] is None and got["note"]
```

(Remove the unused helper arguments if your linter complains; the tests as written are what the plan requires.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_report_stats.py -q`
Expected: the new tests fail with `ImportError: cannot import name 'fragility'`.

- [ ] **Step 3: Implement (append to `scripts/report_stats.py`)**

Add to the imports at the top: `import itertools` and

```python
from scripts.reliability import _distance
from scripts.utils.discovery import parse_subject_session
```

```python
def _names(per: dict) -> list[str]:
    first = next(iter(next(iter(per.values())).values()))
    return sorted(first[0])


def fragility(per: dict) -> dict:
    """Subject-ordering agreement across repeats of one specification (seed) versus across different
    specifications of one wave (param), both on single raw repeats so the arms carry equal noise."""
    nan = float("nan")
    empty = {"seed_rho": nan, "param_rho": nan, "gap": nan, "n_measures": 0, "n_seed_pairs": 0,
             "n_param_pairs": 0, "n_seed_measures": 0, "n_param_measures": 0, "min_n_subjects": nan}
    if not per:
        return empty
    names = _names(per)
    seed_rhos, param_rhos, counts = [], [], []
    seed_measures, param_measures = set(), set()
    for name in names:
        for subjects in per.values():
            keys = sorted(subjects)
            if len(keys) < MIN_SUBJECTS:
                continue
            rho = spearmanr([subjects[k][0][name] for k in keys], [subjects[k][1][name] for k in keys])[0]
            if not np.isnan(rho):
                seed_rhos.append(float(rho)); seed_measures.add(name); counts.append(len(keys))
        rep0 = {c: {k: v[0][name] for k, v in subjects.items()} for c, subjects in per.items()}
        for a, b in itertools.combinations(sorted(rep0), 2):
            if wave_of(a) != wave_of(b):                  # pair only within a wave
                continue
            keys = sorted(set(rep0[a]) & set(rep0[b]))
            if len(keys) < MIN_SUBJECTS:
                continue
            rho = spearmanr([rep0[a][k] for k in keys], [rep0[b][k] for k in keys])[0]
            if not np.isnan(rho):
                param_rhos.append(float(rho)); param_measures.add(name); counts.append(len(keys))
    seed = float(np.mean(seed_rhos)) if seed_rhos else nan
    param = float(np.mean(param_rhos)) if param_rhos else nan
    return {"seed_rho": seed, "param_rho": param, "gap": seed - param, "n_measures": len(names),
            "n_seed_pairs": len(seed_rhos), "n_param_pairs": len(param_rhos),
            "n_seed_measures": len(seed_measures), "n_param_measures": len(param_measures),
            "min_n_subjects": float(min(counts)) if counts else nan}


def effective_dimensionality(per: dict) -> dict:
    """(sum of eigenvalues)^2 / sum of squared eigenvalues of the z-scored measure battery, one row per
    (specification, scan) using the mean of the two repeats; non-finite rows are dropped and counted."""
    nan = float("nan")
    undefined = {"effective_dimensionality": nan, "n_ed_rows": 0, "n_ed_subjects": 0, "n_ed_dropped": 0}
    if not per:
        return undefined
    names = _names(per)
    rows = [[float(np.mean([reps[0][n], reps[1][n]])) for n in names]
            for subjects in per.values() for reps in subjects.values()]
    n_subjects = len({parse_subject_session(s)[0] or s for subjects in per.values() for s in subjects})
    scorable = [r for r in rows if np.all(np.isfinite(r))]
    composition = {"n_ed_rows": len(scorable), "n_ed_subjects": n_subjects, "n_ed_dropped": len(rows) - len(scorable)}
    if n_subjects < MIN_SUBJECTS or len(scorable) < 3:
        return {"effective_dimensionality": nan, **composition}
    x = np.asarray(scorable, dtype=float)
    std = x.std(0, ddof=0)
    std[std == 0] = 1.0
    z = (x - x.mean(0)) / std
    eig = np.linalg.eigvalsh(np.cov(z, rowvar=False))
    eig = eig[eig > 0]
    if eig.size == 0:
        return {"effective_dimensionality": nan, **composition}
    return {"effective_dimensionality": float(eig.sum() ** 2 / (eig ** 2).sum()), **composition}


def _reference_by_wave(reference_ids) -> dict:
    return {wave_of(c): c for c in reference_ids}


def reference_rho(rep1: dict, reference_ids: set) -> dict:
    """Mean Spearman agreement of subject ordering on each measure between every non-reference
    specification and its wave's reference (repeat 1, at least MIN_SUBJECTS shared scans)."""
    nan = float("nan")
    by_wave = _reference_by_wave(reference_ids)
    rhos = []
    if rep1:
        first = next(iter(next(iter(rep1.values())).values()))
        for name in sorted(first):
            vals = {c: {k: v[name] for k, v in subjects.items()} for c, subjects in rep1.items()}
            for c in vals:
                ref = by_wave.get(wave_of(c))
                if c in reference_ids or ref is None or ref not in vals:
                    continue
                keys = sorted(set(vals[c]) & set(vals[ref]))
                if len(keys) < MIN_SUBJECTS:
                    continue
                rho = spearmanr([vals[c][k] for k in keys], [vals[ref][k] for k in keys])[0]
                if not np.isnan(rho):
                    rhos.append(float(rho))
    if not rhos:
        return {"reference_rho": nan, "n_reference_ordering_pairs": 0,
                "note": "no reference specification in the same wave as any other specification"}
    return {"reference_rho": float(np.mean(rhos)), "n_reference_ordering_pairs": len(rhos), "note": None}


def reference_displacement(vectors: dict, reference_ids: set) -> dict:
    """Mean connectome distance (1 - r of the edge vectors) from each non-reference specification to its
    wave's reference, divided by the mean between-subject distance over all specifications."""
    by_wave = _reference_by_wave(reference_ids)
    ref_distances = []
    for c, scans in vectors.items():
        ref = by_wave.get(wave_of(c))
        if c in reference_ids or ref is None or ref not in vectors:
            continue
        for key, vec in scans.items():
            if key in vectors[ref]:
                ref_distances.append(_distance(vec, vectors[ref][key]))
    if not ref_distances:
        return {"available": False, "reference_over_between_subject": None, "n_reference_pairs": 0,
                "note": "no reference specification in the same wave as any other specification"}
    between = []
    for scans in vectors.values():
        keys = sorted(k for k in scans if parse_subject_session(k)[0])
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if parse_subject_session(keys[i])[0] != parse_subject_session(keys[j])[0]:
                    between.append(_distance(scans[keys[i]], scans[keys[j]]))
    if not between or np.mean(between) == 0:
        return {"available": False, "reference_over_between_subject": None,
                "n_reference_pairs": len(ref_distances), "note": "between-subject baseline unavailable or zero"}
    return {"available": True, "reference_over_between_subject": float(np.mean(ref_distances) / np.mean(between)),
            "n_reference_pairs": len(ref_distances), "note": None}
```

Note on the test `reference_rho(...)[...] != ...` NaN line: `float('nan') != float('nan')` is True, which is how that assertion checks for NaN; keep it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_report_stats.py -q -W error`
Expected: all pass (7 from Task 2 + 8 new). If `test_effective_dimensionality_is_one...` is off by more than 1e-6, check that rows with eigenvalues of numerical noise (positive ~1e-17) do not change `(sum)^2/sum of squares` beyond tolerance; they cannot at that scale.

- [ ] **Step 5: Commit**

```bash
git add scripts/report_stats.py tests/test_report_stats.py
git commit -m "feat(report): fragility, effective dimensionality, reference rho and displacement

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The `report` command, hub registration, docs

**Files:**
- Create: `scripts/report.py`, `tests/test_report.py`
- Modify: `scripts/opticonn_hub.py`, `docs/outcome_sensitivity.md`, `README.md`

**Interfaces:**
- Consumes: `find_waves`, `read_meta_waves`, `tables_current`, `write_measures`, `read_measures`, `read_params`, `measure_columns` (`scripts.measures_table`); every function of `scripts.report_stats`; `scripts.reliability.collect_matrices`, `edge_vector`.
- Produces: `build_report(optimize_dir, out_dir=None, atlas=None, metric=None, workers=8, recompute=False) -> dict` (the report; also writes `report.json` and `report.csv` under `out_dir`, default `<optimize>/report`); `flat_row(report) -> dict`; `main(argv=None) -> int`. The report dict has the keys `sweep`, `leader`, `fragility`, `effective_dimensionality`, `displacement`, `reference_rho`, `density_relation`, `notes`; each statistic section is `None` when skipped, with the reason in `notes`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_report.py
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.report as report_module
from scripts.report import build_report, flat_row
from tests.sweep_fixture import make_sweep

ROOT = Path(__file__).resolve().parents[1]
SPECS = [({"fa_threshold": 0.0}, False, 0), ({"fa_threshold": 0.1}, False, 0), ({"fa_threshold": 0.2}, False, 0),
         ({"fa_threshold": 0.3}, False, 50),                    # rejected by the (fixture's) gate
         ({"fa_threshold": 0.4}, False, 0), ({"fa_threshold": None}, True, 0)]


def _two_waves(root, repeats=2):
    make_sweep(root, SPECS, n_subjects=6, wave="wave1", repeats=repeats, margin=lambda i: 0.05 * i)
    make_sweep(root, SPECS, n_subjects=6, wave="wave2", repeats=repeats, margin=lambda i: 0.05 * i + 0.001)
    return root


def test_a_full_sweep_gets_every_statistic(tmp_path):
    rep = build_report(_two_waves(tmp_path / "opt"), workers=1)
    assert rep["sweep"]["waves"] == ["wave1", "wave2"] and rep["sweep"]["n_specifications"] == 12
    assert rep["sweep"]["n_rejected"] == 2 and rep["sweep"]["reference"] == ["wave1/sweep_0006", "wave2/sweep_0006"]
    lead = rep["leader"]
    assert lead["n_candidates"] == 4 and lead["winners_agree"] is True and lead["winner_determined"] is True
    assert lead["cross_wave_rho"] == pytest.approx(1.0)
    assert rep["fragility"]["n_seed_pairs"] > 0 and rep["fragility"]["seed_rho"] <= 1.0
    assert rep["effective_dimensionality"]["n_ed_rows"] > 0
    assert rep["displacement"]["available"] is True and rep["reference_rho"]["n_reference_ordering_pairs"] > 0
    assert rep["density_relation"]["reference_rejected"] is False
    assert 0.0 <= rep["density_relation"]["reference_density_percentile"] <= 1.0
    assert "small_worldness(binary)_vs_density" in rep["density_relation"]["relations"]
    assert rep["notes"] == []


def test_the_rejected_specifications_do_not_enter_the_pooled_statistics(tmp_path):
    rep = build_report(_two_waves(tmp_path / "opt"), workers=1)
    # 12 specifications, 2 rejected: fragility seed arm = measures x kept specifications
    assert rep["fragility"]["n_seed_pairs"] == rep["fragility"]["n_measures"] * 10


def test_report_files_are_written_and_the_json_is_strict(tmp_path):
    opt = _two_waves(tmp_path / "opt")
    build_report(opt, workers=1)
    text = (opt / "report" / "report.json").read_text()
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-standard {c}")))
    with open(opt / "report" / "report.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]["n_candidates"] == "4" and rows[0]["winner_determined"] == "True"
    assert "seed_rho" in rows[0] and "reference_over_between_subject" in rows[0]


def test_a_single_wave_single_repeat_sweep_skips_what_it_cannot_do_and_says_why(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=6, repeats=1, margin=lambda i: 0.05 * i)
    rep = build_report(tmp_path / "opt", workers=1)
    assert rep["leader"] is None and rep["fragility"] is None and rep["effective_dimensionality"] is None
    assert rep["displacement"]["available"] is True and rep["density_relation"] is not None
    notes = " ".join(rep["notes"])
    assert "two waves" in notes and "two repeats" in notes
    assert flat_row(rep)["n_candidates"] is None and "n_candidates" in flat_row(rep)


def test_a_sweep_without_a_reference_skips_displacement_and_reference_statistics(tmp_path):
    specs = [s for s in SPECS if not s[1]]
    make_sweep(tmp_path / "opt", specs, n_subjects=6, repeats=2, margin=lambda i: 0.05 * i)
    rep = build_report(tmp_path / "opt", workers=1)
    assert rep["displacement"] is None and rep["reference_rho"] is None and "reference" in " ".join(rep["notes"])
    assert rep["density_relation"]["reference_density_percentile"] is None


def test_existing_tables_are_reused_until_recompute_is_asked(tmp_path, monkeypatch):
    calls = []
    real = report_module.write_measures
    monkeypatch.setattr(report_module, "write_measures", lambda *a, **k: (calls.append(k["repeat"]), real(*a, **k))[1])
    opt = _two_waves(tmp_path / "opt")
    build_report(opt, workers=1)
    assert calls == [1, 2]
    build_report(opt, workers=1)
    assert calls == [1, 2]                                   # cached
    build_report(opt, workers=1, recompute=True)
    assert calls == [1, 2, 1, 2]


def test_an_ambiguous_metric_is_a_clean_error_and_the_command_runs_through_the_hub(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=6, repeats=2, metrics=("count", "qa"), margin=lambda i: 0.05 * i)
    with pytest.raises(SystemExit, match="more than one atlas/metric"):
        report_module.main(["-i", str(tmp_path / "opt"), "--workers", "1"])
    out = subprocess.run([sys.executable, str(ROOT / "opticonn.py"), "report", "-i", str(tmp_path / "opt"),
                          "--metric", "count", "--workers", "1"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "opt" / "report" / "report.json").exists()
    help_out = subprocess.run([sys.executable, str(ROOT / "opticonn.py"), "report", "--help"], cwd=ROOT,
                              capture_output=True, text=True, timeout=60)
    assert help_out.returncode == 0 and "--recompute" in help_out.stdout
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_report.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'scripts.report'`.

- [ ] **Step 3: Implement `scripts/report.py`**

```python
# scripts/report.py
"""Sweep-level sensitivity report: is the leader determined, how fragile is the subject ordering,
how many independent axes do the graph measures carry, how far are the specifications from the defaults.

    opticonn report -i <optimize> [--atlas A] [--metric M] [--workers N] [--recompute]

Writes <optimize>/report/report.json and report.csv. A statistic the sweep cannot support (one wave, one
repeat, no reference) is skipped with a stated reason. Design:
docs/superpowers/specs/2026-10-25-sensitivity-report-design.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from scripts import report_stats as stats
from scripts.measures_table import (find_waves, measure_columns, read_measures, read_params, tables_current,
                                    write_measures)
from scripts.reliability import collect_matrices, edge_vector


def _num(text):
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _diagnostics(wave_dir: Path) -> list[dict] | None:
    path = Path(wave_dir) / "combo_diagnostics.csv"
    if not path.exists():
        return None
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _clean(obj):
    """NaN and infinity become None, numpy scalars become Python ones: the JSON stays strict."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _ensure_table(optimize_dir, out, repeat, atlas, metric, n_specs, workers, recompute):
    d = out / "tables" / f"repeat{repeat}"
    if recompute or not tables_current(d, atlas, metric, repeat, "all", n_specs):
        write_measures(optimize_dir, out_dir=d, atlas=atlas, metric=metric, repeat=repeat, workers=workers, wave="all")
    return d


def build_report(optimize_dir, out_dir=None, atlas=None, metric=None, workers=8, recompute=False) -> dict:
    optimize_dir = Path(optimize_dir)
    out = Path(out_dir) if out_dir else optimize_dir / "report"
    waves = find_waves(optimize_dir, "all")
    scoped = len(waves) > 1
    n_specs = sum(len(list((w / "combos").glob("sweep_*"))) for w in waves)
    notes = []

    t1 = _ensure_table(optimize_dir, out, 1, atlas, metric, n_specs, workers, recompute)
    rows1 = read_measures(t1)
    params = read_params(t1)
    side = json.loads((t1 / "measures.json").read_text())
    t2 = None
    try:
        t2 = _ensure_table(optimize_dir, out, 2, side["atlas"], side["metric"], n_specs, workers, recompute)
        rows2 = read_measures(t2)
    except ValueError:
        notes.append("fragility and effective dimensionality skipped: they need two repeats")

    reference_ids = {cid for cid, m in params.items() if m["reference"]}
    diag = {w.name: _diagnostics(w) for w in waves}
    rejected = set()
    for w in waves:
        for r in diag[w.name] or []:
            if (r.get("rejected") or "").strip():
                rejected.add(f"{w.name}/{r['sweep_id']}" if scoped else r["sweep_id"])
    kept_ids = {c for c in params if c not in rejected}
    names = measure_columns(rows1)

    def by_scan(rows, ids):
        d = {}
        for r in rows:
            if r["combo_id"] in ids:
                d.setdefault(r["combo_id"], {})[r["scan"]] = {n: r[n] for n in names}
        return d

    rep1 = by_scan(rows1, kept_ids)

    # leader test
    leader = None
    if len(waves) == 2 and all(diag[w.name] is not None for w in waves):
        def margins(w):
            return {r["sweep_id"]: _num(r.get("discriminability_margin")) for r in diag[w.name]
                    if (r.get("reference") or "").strip().lower() not in ("true", "1")}
        discr1 = [v for r in diag[waves[0].name] if (r.get("reference") or "").strip().lower() not in ("true", "1")
                  for v in [_num(r.get("discriminability"))] if v is not None]
        leader = stats.leader_test(margins(waves[0]), margins(waves[1]), discr1)
        notes += leader.pop("notes")
    else:
        notes.append("leader test skipped: it needs exactly two waves with a combo_diagnostics.csv each")

    # fragility and effective dimensionality
    frag = ed = None
    if t2 is not None:
        r2 = by_scan(rows2, kept_ids)
        per = {c: {k: [rep1[c][k], r2[c][k]] for k in rep1[c] if k in r2.get(c, {})} for c in rep1}
        per = {c: s for c, s in per.items() if s}
        frag, ed = stats.fragility(per), stats.effective_dimensionality(per)

    # displacement and reference rho
    displacement = ref_rho = None
    if reference_ids:
        vectors = {}
        for w in waves:
            for combo in sorted((w / "combos").glob("sweep_*")):
                cid = f"{w.name}/{combo.name}" if scoped else combo.name
                if cid in kept_ids:
                    mats = collect_matrices(combo).get((side["atlas"], side["metric"]), {})
                    vectors[cid] = {k: edge_vector(reps[0]) for k, reps in mats.items() if reps}
        displacement = stats.reference_displacement(vectors, reference_ids & kept_ids)
        ref_rho = stats.reference_rho(rep1, reference_ids & kept_ids)
        for section in (displacement, ref_rho):
            if section.get("note"):
                notes.append(section["note"])
    else:
        notes.append("displacement and reference rho skipped: the sweep has no reference specification")

    # density relation, reference verdict and percentile
    rows_all = rows1 + (rows2 if t2 is not None else [])
    cols = ["density"] + [n for n in names if n != "density"]
    by_combo_rows = {}
    for r in rows_all:
        if r["combo_id"] in kept_ids:
            by_combo_rows.setdefault(r["combo_id"], []).append(r)
    by_sweep, wave_density = {}, {}
    for cid, rs in by_combo_rows.items():
        wave_density.setdefault(stats.wave_of(cid), {})[cid] = float(np.mean([r["density"] for r in rs]))
        if cid not in reference_ids:                       # the relation is over the non-reference specifications
            by_sweep.setdefault(cid.split("/", 1)[-1], []).extend(rs)
    spec_means = {sid: {n: float(np.mean([r[n] for r in rs])) for n in cols} for sid, rs in by_sweep.items()}
    relation = {"relations": stats.density_relation(spec_means, set()),
                "reference_rejected": bool(reference_ids & rejected),
                "reference_density_percentile": stats.reference_percentile(
                    wave_density, {stats.wave_of(c): c for c in reference_ids if c in kept_ids})}

    report = {"sweep": {"optimize_dir": str(optimize_dir), "atlas": side["atlas"], "metric": side["metric"],
                        "waves": [w.name for w in waves], "n_specifications": n_specs,
                        "n_rejected": len(rejected), "reference": sorted(reference_ids)},
              "leader": leader, "fragility": frag, "effective_dimensionality": ed,
              "displacement": displacement, "reference_rho": ref_rho, "density_relation": relation,
              "notes": notes}
    report = _clean(report)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    row = flat_row(report)
    with open(out / "report.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return report


COLUMNS = {
    "leader": ["n_candidates", "discriminability_saturated", "cross_wave_rho", "winner_wave1", "winner_wave2",
               "winners_agree", "top_two_separation", "wave_variation", "winner_determined"],
    "fragility": ["seed_rho", "param_rho", "gap", "n_measures", "n_seed_pairs", "n_param_pairs",
                  "n_seed_measures", "n_param_measures", "min_n_subjects"],
    "effective_dimensionality": ["effective_dimensionality", "n_ed_rows", "n_ed_subjects", "n_ed_dropped"],
    "displacement": ["reference_over_between_subject", "n_reference_pairs"],
    "reference_rho": ["reference_rho", "n_reference_ordering_pairs"],
}


def flat_row(report: dict) -> dict:
    """One flat row of scalars with fixed column names (None where a statistic was skipped)."""
    row = {}
    for section, cols in COLUMNS.items():
        data = report.get(section) or {}
        row.update({c: data.get(c) for c in cols})
    rel = report.get("density_relation") or {}
    row.update({k: v for k, v in (rel.get("relations") or {}).items()})
    row["reference_rejected"] = rel.get("reference_rejected")
    row["reference_density_percentile"] = rel.get("reference_density_percentile")
    row["notes"] = " | ".join(report.get("notes") or [])
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="opticonn report", description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--input", required=True, help="a sweep's optimize/ directory")
    ap.add_argument("-o", "--output-dir", default=None, help="default <input>/report")
    ap.add_argument("--atlas", default=None)
    ap.add_argument("--metric", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--recompute", action="store_true", help="rebuild the cached per-subject tables")
    a = ap.parse_args(argv)
    try:
        report = build_report(a.input, a.output_dir, a.atlas, a.metric, a.workers, a.recompute)
    except (ValueError, OSError) as err:
        raise SystemExit(f"opticonn report: {err}")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Register the command in `scripts/opticonn_hub.py`**

In the `PASS_THROUGH` dict add `"report": "report"`; in the `for name, summary in (...)` registration tuple add
`("report", "Sweep-level sensitivity report: leader test, fragility, effective dimensionality, displacement from the defaults"),`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_report.py tests/test_report_stats.py tests/test_outcome_cli.py tests/test_cli_help.py -q -W error`
Expected: all pass. If `test_a_full_sweep_gets_every_statistic` fails on `n_candidates == 4`, check that the fixture marked specification 4 (offset 50) as rejected with a blank margin and that specification 6 is the reference (excluded from the margins), leaving specifications 1, 2, 3, 5.

- [ ] **Step 6: Document it**

Append to `docs/outcome_sensitivity.md`:

````markdown
## Sweep-level report

```bash
opticonn report -i studies/demo/sweep-*/optimize
```

For a sweep with two waves and two repeats this writes `report/report.json` and a one-row `report.csv`:
whether the leading specification is determined (its margin over the runner-up against the wave-to-wave
variation), how well the ordering of subjects on each graph measure survives re-running a specification
(`seed_rho`) against changing it (`param_rho`), how many independent axes the graph measures carry
(effective dimensionality), how far the specifications sit from the reference (DSI Studio's untouched
defaults) relative to the difference between people, and how much each measure simply tracks connectome
density. A statistic the sweep cannot support (one wave, one repeat, no reference) is skipped with the
reason in `notes`. The per-subject tables it needs are built once into `report/tables/` and reused;
`--recompute` rebuilds them.
````

Add to `README.md`, after the "How much does your own estimate depend" block, one line: ``opticonn report -i studies/demo_grid/sweep-*/optimize   # leader test, fragility, dimensionality, displacement from the defaults``.

- [ ] **Step 7: Commit**

```bash
git add scripts/report.py scripts/opticonn_hub.py tests/test_report.py docs/outcome_sensitivity.md README.md
git commit -m "feat(report): opticonn report, the sweep-level sensitivity report

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Acceptance against the paper's committed numbers

**Files:**
- Create: `tests/test_report_acceptance.py`

**Interfaces:**
- Consumes: `build_report`, `flat_row` (Task 4); environment variables `OPTICONN_ACCEPT_RESULTS` (the study repository's `results/` directory, holding `<dataset>/sweep-*/optimize` for the seven original sweeps) and `OPTICONN_ACCEPT_BATTERY` (the study repository root with `fragility_summary.csv`, `cross_dataset_summary.csv`, `defaults_gates.csv`). Skipped unless both are set.

- [ ] **Step 1: Write the test**

```python
# tests/test_report_acceptance.py
"""`opticonn report` reproduces the study repository's committed sweep-level numbers (proof the move
changed no result). Skipped unless the acceptance data is configured; run once per release."""

import csv
import glob
import math
import os
from pathlib import Path

import pytest

from scripts.report import build_report, flat_row

RESULTS = os.environ.get("OPTICONN_ACCEPT_RESULTS")
BATTERY = os.environ.get("OPTICONN_ACCEPT_BATTERY")
pytestmark = pytest.mark.skipif(not (RESULTS and BATTERY), reason="acceptance data not configured")

DATASETS = ["ds000221", "ds005270", "ds005256", "ds005613", "ds005577", "ds004958", "ds005360"]
FRAGILITY = ["seed_rho", "param_rho", "gap", "effective_dimensionality", "reference_over_between_subject",
             "reference_rho", "n_measures", "n_seed_pairs", "n_param_pairs", "n_seed_measures", "n_param_measures",
             "min_n_subjects", "n_ed_rows", "n_ed_subjects", "n_ed_dropped", "n_reference_pairs",
             "n_reference_ordering_pairs"]
LEADER = ["n_candidates", "discriminability_saturated", "cross_wave_rho", "winner_wave1", "winner_wave2",
          "winners_agree", "top_two_separation", "wave_variation", "winner_determined"]


def _rows(name, key="dataset_id"):
    with open(Path(BATTERY) / name, newline="") as handle:
        csv.field_size_limit(10 ** 9)
        out = {}
        for r in csv.DictReader(handle):
            out.setdefault(r[key], r)                     # dataset-level columns repeat on every metric row
        return out


def _same(new, old):
    if old in ("", None):
        return new in ("", None)
    if new in ("", None):
        return False
    if old in ("True", "False"):
        return str(new) == old
    try:
        return abs(float(new) - float(old)) < 1e-9
    except ValueError:
        return str(new) == old


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    out = {}
    for ds in DATASETS:
        optimize = glob.glob(f"{RESULTS}/{ds}/sweep-*/optimize")[0]
        out[ds] = build_report(optimize, out_dir=tmp_path_factory.mktemp(ds), atlas="AAL3", metric="count", workers=16)
    return out


def test_fragility_dimensionality_and_displacement_match_the_committed_table(reports):
    old, compared = _rows("fragility_summary.csv"), 0
    for ds in DATASETS:
        row = flat_row(reports[ds])
        for col in FRAGILITY:
            assert _same(row.get(col), old[ds][col]), (ds, col, row.get(col), old[ds][col])
            compared += old[ds][col] != ""
    assert compared > 7 * 10


def test_the_leader_test_matches_the_committed_table(reports):
    old, compared = _rows("cross_dataset_summary.csv"), 0
    for ds in DATASETS:
        row = flat_row(reports[ds])
        for col in LEADER:
            assert _same(row.get(col), old[ds][col]), (ds, col, row.get(col), old[ds][col])
            compared += old[ds][col] != ""
    assert compared > 7 * 7


def test_the_density_relation_and_the_reference_verdict_agree_with_the_committed_defaults_table(reports):
    old = _rows("defaults_gates.csv")
    for ds in DATASETS:
        row = flat_row(reports[ds])
        assert str(row["reference_rejected"]) == old[ds]["reference_rejected"], ds
        # the committed numbers average OptiConn's per-combo diagnostics; the report averages repeat tables
        assert abs(float(row["reference_density_percentile"]) - float(old[ds]["reference_density_percentile"])) < 0.02, ds
        a, b = float(row["small_worldness(binary)_vs_density"]), float(old[ds]["small_worldness_vs_density"])
        assert abs(a - b) < 0.01 and a * b > 0, (ds, a, b)
```

- [ ] **Step 2: Run it without data to verify it skips**

Run: `python -m pytest tests/test_report_acceptance.py -q`
Expected: `3 skipped`.

- [ ] **Step 3: Run it for real (once the earlier tasks are merged into the worktree and the study repository's `results/` directory is readable)**

```bash
OPTICONN_ACCEPT_RESULTS=/data/local/software/opticonn-multiverse-battery/results \
OPTICONN_ACCEPT_BATTERY=/data/local/software/opticonn-multiverse-battery \
python -m pytest tests/test_report_acceptance.py -q -p no:cacheprovider
```
Expected: `3 passed`. A difference beyond the stated tolerance in the first two tests is a bug in the move: find it (the likeliest causes are the gating, the pooling of the reference, or the subject set), never widen the tolerance. The third test's looser tolerances (0.02 and 0.01) are by design (see the spec).

- [ ] **Step 4: Commit**

```bash
git add tests/test_report_acceptance.py
git commit -m "test(report): acceptance against the study repository's committed sweep-level numbers

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage.** Command and flags: Task 4 (`-i -o --atlas --metric --workers --recompute`). Statistics 1-5: Tasks 2 (leader test, density relation) and 3 (fragility, effective dimensionality, displacement, reference rho). Gating by the sweep's `rejected` flag, reference pooling rules, strict JSON: Task 4 and the Global Constraints. `measures --wave all` and cache reuse: Task 1. Skip-with-reason behaviour: Task 4 tests. Acceptance and the density-relation tolerance: Task 5. Out-of-scope items are not implemented.

**Placeholder scan.** None. The Task 3 note about the NaN assertion and the Task 4 replacement paragraph for the density block are explicit instructions, not gaps.

**Type consistency.** `per` is `{combo: {scan: [m1, m2]}}` for `fragility`/`effective_dimensionality`; `rep1` is `{combo: {scan: m}}` for `reference_rho`; `vectors` is `{combo: {scan: ndarray}}` for `reference_displacement`; all three are built in `build_report` from `read_measures` rows and `collect_matrices`. `density_relation` takes `{sweep_id: {measure: mean}}`; `reference_percentile` takes `{wave: {combo: density}}`.
