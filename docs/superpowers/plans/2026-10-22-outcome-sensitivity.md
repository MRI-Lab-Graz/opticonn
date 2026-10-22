# Outcome sensitivity (measures, outcome, pca) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three OptiConn commands (`measures`, `outcome`, `pca`) that report how a user-defined outcome depends on the tractography specification, reproducing the study repository's age, sex and PCA analyses without any dataset- or analysis-specific code.

**Architecture:** `measures` reads a completed sweep's matrices once into `measures.csv` and `params.csv`; `outcome` and `pca` read those plus a per-subject variables table. New modules in `scripts/`, registered in `scripts/opticonn_hub.py` as pass-through subcommands. Existing code is reused: `reliability.collect_matrices`, `reliability.gate_reason`, `compute_network_measures_from_connectivity.measures_from_matrix`.

**Tech Stack:** Python 3, numpy, scipy, networkx, matplotlib, pytest (all already OptiConn dependencies).

Design: `docs/superpowers/specs/2026-10-22-outcome-sensitivity-design.md`. Work in the worktree `/data/local/software/opticonn-wt-outcome` on branch `feature/outcome-sensitivity`. Run tests from the worktree root with `python -m pytest`.

## Global Constraints

- Partial Spearman is the only estimator: Pearson correlation of the ranks after regressing the ranks of the covariates out; p from the t distribution with `n - 2 - k` degrees of freedom for `k` covariates; 95% interval by Fisher's z with standard error `1/sqrt(n - 3 - k)`.
- Gates are `scripts.reliability.DEFAULT_GATES` (density 0.02-0.6, at most 10% isolated nodes), judged on a specification's mean density and mean isolated fraction.
- A column with exactly two distinct values is coded 0/1 in sorted order and the coding is recorded; more than two levels is an error; no dummy coding.
- The reference specification (the `reference` flag in each combo's `diagnostics.json`) is never a candidate and is always reported.
- A subject missing the predictor, a covariate or a measure is dropped from that specification and counted.
- No existing OptiConn module changes except registering the subcommands in `scripts/opticonn_hub.py`.
- Every step that changes behaviour starts with a failing test.
- Commit messages end with the line `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## File Structure

- Create `scripts/variables.py`: read the variables table, normalise ids, code columns.
- Create `scripts/measures_table.py`: the five extra measures, per-matrix rows, wave and meta discovery, `write_measures`, the `measures` CLI.
- Create `scripts/outcome_curve.py`: `partial_spearman`, per-specification rows, summary, plot, the `outcome` CLI.
- Create `scripts/pca_curve.py`: per-specification PCA, summary, separating parameters, the `pca` CLI.
- Modify `scripts/opticonn_hub.py`: register the three subcommands.
- Create `tests/sweep_fixture.py` (synthetic sweep builder) and one test module per source module, plus `tests/test_outcome_cli.py` and `tests/test_outcome_acceptance.py`.
- Create `docs/outcome_sensitivity.md` and add a README section.

---

### Task 1: Variables table

**Files:**
- Create: `scripts/variables.py`
- Test: `tests/test_variables.py`

**Interfaces:**
- Produces: `subject_id(raw: str) -> str` ("sub-" form); `load_variables(path, columns: list[str]) -> tuple[dict[str, dict[str, float | None]], dict[str, dict[str, str]]]` returning `({"sub-01": {col: float|None}}, {col: {"0": level0, "1": level1}})` where the second dict has an entry only for columns that were two-level categorical.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_variables.py
import pytest

from scripts.variables import load_variables, subject_id


def _write(tmp_path, text, name="participants.tsv"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_ids_match_with_and_without_the_sub_prefix(tmp_path):
    p = _write(tmp_path, "participant_id\tage\n01\t20\nsub-02\t30\n")
    table, _ = load_variables(p, ["age"])
    assert table == {"sub-01": {"age": 20.0}, "sub-02": {"age": 30.0}}
    assert subject_id("sub-7") == "sub-7" and subject_id(" 7 ") == "sub-7"


def test_missing_values_become_none(tmp_path):
    p = _write(tmp_path, "participant_id\tage\n01\tn/a\n02\t\n03\t41\n")
    table, _ = load_variables(p, ["age"])
    assert table["sub-01"]["age"] is None and table["sub-02"]["age"] is None
    assert table["sub-03"]["age"] == 41.0


def test_a_two_level_column_is_coded_in_sorted_order_and_the_coding_is_recorded(tmp_path):
    p = _write(tmp_path, "participant_id\tsex\n01\tM\n02\tF\n03\tM\n")
    table, coding = load_variables(p, ["sex"])
    assert [table[s]["sex"] for s in ("sub-01", "sub-02", "sub-03")] == [1.0, 0.0, 1.0]
    assert coding == {"sex": {"0": "F", "1": "M"}}


def test_a_numeric_two_valued_column_is_left_as_it_is(tmp_path):
    p = _write(tmp_path, "participant_id\tgroup\n01\t1\n02\t0\n")
    table, coding = load_variables(p, ["group"])
    assert table["sub-01"]["group"] == 1.0 and coding == {}


def test_more_than_two_levels_is_an_error_naming_the_column(tmp_path):
    p = _write(tmp_path, "participant_id\tsite\n01\tA\n02\tB\n03\tC\n")
    with pytest.raises(ValueError, match="site.*3 levels"):
        load_variables(p, ["site"])


def test_an_unknown_column_lists_the_available_ones(tmp_path):
    p = _write(tmp_path, "participant_id\tage\n01\t20\n")
    with pytest.raises(ValueError, match="no column named height.*age"):
        load_variables(p, ["height"])


def test_participant_id_is_required(tmp_path):
    p = _write(tmp_path, "id\tage\n01\t20\n")
    with pytest.raises(ValueError, match="participant_id"):
        load_variables(p, ["age"])


def test_a_comma_separated_file_is_read_too(tmp_path):
    p = _write(tmp_path, "participant_id,age\n01,20\n", name="vars.csv")
    table, _ = load_variables(p, ["age"])
    assert table == {"sub-01": {"age": 20.0}}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_variables.py -q`
Expected: FAIL / collection error `ModuleNotFoundError: No module named 'scripts.variables'`.

- [ ] **Step 3: Implement**

```python
# scripts/variables.py
"""Read a per-subject variables table (a BIDS participants.tsv works) for outcome analyses."""

from __future__ import annotations

import csv
from pathlib import Path

MISSING = {"", "n/a", "na", "nan", "none"}


def subject_id(raw: str) -> str:
    """Normalise a participant id to the 'sub-<id>' form of scan names."""
    raw = str(raw).strip()
    return raw if raw.startswith("sub-") else f"sub-{raw}"


def _read_table(path) -> list[dict]:
    path = Path(path)
    delimiter = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _value(raw):
    raw = (raw or "").strip()
    return None if raw.lower() in MISSING else raw


def load_variables(path, columns: list[str]) -> tuple[dict, dict]:
    """({subject: {column: float | None}}, {column: {"0": level, "1": level}}).

    Numeric columns are kept. A non-numeric column with exactly two distinct values is coded
    0/1 in sorted order (the coding is returned); any other non-numeric column is an error.
    """
    rows = _read_table(path)
    if not rows or "participant_id" not in rows[0]:
        raise ValueError(f"{path}: a 'participant_id' column is required")
    absent = [c for c in columns if c not in rows[0]]
    if absent:
        raise ValueError(f"{path}: no column named {', '.join(absent)}; columns are {', '.join(rows[0])}")
    table = {subject_id(r["participant_id"]): {c: _value(r[c]) for c in columns} for r in rows}
    coding = {}
    for c in dict.fromkeys(columns):
        raw = {sid: v[c] for sid, v in table.items()}
        try:
            coded = {sid: None if s is None else float(s) for sid, s in raw.items()}
        except ValueError:
            levels = sorted({s for s in raw.values() if s is not None})
            if len(levels) != 2:
                raise ValueError(f"column '{c}' has {len(levels)} levels ({', '.join(levels[:4])}): "
                                 "only numeric or two-level columns are supported")
            coding[c] = {"0": levels[0], "1": levels[1]}
            coded = {sid: None if s is None else float(s == levels[1]) for sid, s in raw.items()}
        for sid, x in coded.items():
            table[sid][c] = x
    return table, coding
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_variables.py -q`
Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/variables.py tests/test_variables.py
git commit -m "feat(variables): read a per-subject variables table with id matching and binary coding

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Partial Spearman estimator

**Files:**
- Create: `scripts/outcome_curve.py` (this task adds only `partial_spearman`)
- Test: `tests/test_outcome_curve.py`

**Interfaces:**
- Produces: `partial_spearman(x, y, covariates=()) -> dict` with keys `rho, p, ci_low, ci_high, n` (floats; `n` int). `covariates` is a sequence of columns (each a sequence of length n). If `n < len(covariates) + 4` every statistic except `n` is `None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_outcome_curve.py
import numpy as np
import pytest
from scipy.stats import rankdata, spearmanr

from scripts.outcome_curve import partial_spearman


def _inverse_formula(x, y, covariates):
    """Partial correlation from the inverse correlation matrix of the ranks (independent route)."""
    cols = [rankdata(x), rankdata(y)] + [rankdata(c) for c in covariates]
    p = np.linalg.inv(np.corrcoef(cols))
    return -p[0, 1] / np.sqrt(p[0, 0] * p[1, 1])


X = [21, 25, 30, 34, 41, 47, 52, 60, 66, 71, 38, 44]
Y = [3.1, 2.9, 3.4, 2.8, 2.6, 2.9, 2.2, 2.4, 2.0, 2.1, 3.0, 2.5]
Z1 = [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0]
Z2 = [5, 3, 8, 1, 9, 2, 7, 4, 6, 0, 10, 11]


def test_no_covariate_is_the_plain_spearman_with_its_p_value():
    got = partial_spearman(X, Y)
    ref = spearmanr(X, Y)
    assert got["rho"] == pytest.approx(ref.statistic) and got["p"] == pytest.approx(ref.pvalue)
    assert got["n"] == 12


def test_one_and_two_covariates_match_the_inverse_matrix_formula():
    assert partial_spearman(X, Y, [Z1])["rho"] == pytest.approx(_inverse_formula(X, Y, [Z1]))
    assert partial_spearman(X, Y, [Z1, Z2])["rho"] == pytest.approx(_inverse_formula(X, Y, [Z1, Z2]))


def test_the_interval_brackets_the_estimate_and_p_uses_n_minus_2_minus_k_df():
    from scipy.stats import t
    got = partial_spearman(X, Y, [Z1, Z2])
    assert got["ci_low"] < got["rho"] < got["ci_high"]
    df = 12 - 2 - 2
    tval = got["rho"] * np.sqrt(df / (1 - got["rho"] ** 2))
    assert got["p"] == pytest.approx(2 * t.sf(abs(tval), df))


def test_a_perfect_fit_gives_p_zero_instead_of_crashing():
    got = partial_spearman(list(range(10)), list(range(10)), [[0, 1] * 5])
    assert got["rho"] == pytest.approx(1.0) and got["p"] == 0.0


def test_too_few_subjects_gives_no_estimate():
    got = partial_spearman([1, 2, 3, 4], [1, 2, 4, 3], [[0, 1, 0, 1]])
    assert got["n"] == 4 and got["rho"] is None and got["p"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_outcome_curve.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'scripts.outcome_curve'`.

- [ ] **Step 3: Implement**

```python
# scripts/outcome_curve.py
"""Outcome sensitivity: the same association, estimated in every specification of a sweep.

    opticonn outcome -i <optimize> --variables participants.tsv --predictor age \\
        --covariate sex --measure "global_efficiency(weighted)"

Design: docs/superpowers/specs/2026-10-22-outcome-sensitivity-design.md
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from scipy.stats import t as t_dist


def partial_spearman(x, y, covariates=()) -> dict:
    """Rank correlation of x and y after regressing the ranks of the covariates out of both.

    p from the t distribution with n - 2 - k degrees of freedom, 95% interval by Fisher's z
    with standard error 1/sqrt(n - 3 - k), for k covariates. None where n < k + 4.
    """
    n, k = len(x), len(covariates)
    out = {"rho": None, "p": None, "ci_low": None, "ci_high": None, "n": n}
    if n < k + 4:
        return out
    design = np.column_stack([np.ones(n)] + [rankdata(c) for c in covariates])
    res = [v - design @ np.linalg.lstsq(design, v, rcond=None)[0] for v in (rankdata(x), rankdata(y))]
    rho = np.corrcoef(res)[0, 1]          # numpy float: a perfect fit gives p = 0, not a ZeroDivisionError
    with np.errstate(divide="ignore", invalid="ignore"):
        t = rho * np.sqrt((n - 2 - k) / (1 - rho ** 2))
        half = 1.96 / np.sqrt(n - 3 - k)
        lo, hi = np.tanh(np.arctanh(rho) - half), np.tanh(np.arctanh(rho) + half)
    out.update(rho=float(rho), p=float(2 * t_dist.sf(abs(t), n - 2 - k)), ci_low=float(lo), ci_high=float(hi))
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_outcome_curve.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/outcome_curve.py tests/test_outcome_curve.py
git commit -m "feat(outcome): partial Spearman for any number of covariates

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Measures table (`opticonn measures`)

**Files:**
- Create: `tests/sweep_fixture.py`, `scripts/measures_table.py`
- Test: `tests/test_measures_table.py`

**Interfaces:**
- Produces (in `scripts/measures_table.py`):
  - `EXTRA: list[str]` (five names), `extra_measures(mat) -> dict[str, float]`
  - `matrix_row(mat, extended: bool) -> dict[str, float]` (the seven `measures_from_matrix` values, `isolated_fraction`, plus `EXTRA` if extended)
  - `find_wave(optimize_dir, wave=None) -> Path`
  - `read_meta(wave_dir) -> dict[str, dict]` ({"sweep_0001": {"reference": bool, "parameters": dict}})
  - `write_measures(optimize_dir, out_dir=None, atlas=None, metric=None, repeat=1, extended=False, workers=8, wave=None) -> Path` (returns the output directory)
  - `read_measures(measures_dir) -> list[dict]` (floats for measure columns; `combo_id`, `subject`, `scan` strings); `measure_columns(rows) -> list[str]`; `read_params(measures_dir) -> dict[str, dict]` ({"sweep_0001": {"reference": bool, "parameters": {name: str}}})
  - `main(argv=None) -> int`
- Produces (in `tests/sweep_fixture.py`): `make_sweep(root, specs, n_subjects=15, wave="wave1", metrics=("count",), base_edges=40) -> Path`; `specs` is a list of `(parameters: dict, reference: bool, edge_offset: int)`. Subject `i` (0-based) of every specification has `base_edges + edge_offset + 2*i` edges, so density rises with subject index; files are `sub-NN_ses-1_dwi.qsdr.<metric>..pass.connectivity.mat` under `<root>/<wave>/combos/sweep_NNNN/rep_1/AAL3/`, with a `diagnostics.json` per combo.

- [ ] **Step 1: Write the synthetic sweep fixture**

```python
# tests/sweep_fixture.py
"""A tiny synthetic sweep in the layout OptiConn's collect_matrices reads (no DSI Studio needed)."""

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


def make_sweep(root, specs, n_subjects=15, wave="wave1", metrics=("count",), base_edges=40):
    """specs: [(parameters, reference, edge_offset)]. Subject i has base_edges + offset + 2*i edges
    in every specification, so density (and the rank of every subject on it) is known exactly.
    An offset of 60 or more pushes density above the 0.6 gate."""
    for index, (params, reference, offset) in enumerate(specs, start=1):
        combo = Path(root) / wave / "combos" / f"sweep_{index:04d}"
        for i in range(n_subjects):
            rng = np.random.default_rng(1000 * index + i)
            d = combo / "rep_1" / "AAL3"
            d.mkdir(parents=True, exist_ok=True)
            for metric in metrics:
                scipy.io.savemat(d / f"sub-{i + 1:02d}_ses-1_dwi.qsdr.{metric}..pass.connectivity.mat",
                                 {"connectivity": _matrix(rng, base_edges + offset + 2 * i)})
        (combo / "diagnostics.json").write_text(json.dumps(
            {"status": "ok", "wave": wave, "combo_index": index, "sampler": "grid",
             "reference": reference, "parameters": params}))
    return Path(root)
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_measures_table.py
import csv
import json
import math

import numpy as np
import pytest

from scripts.measures_table import (EXTRA, extra_measures, find_wave, matrix_row, measure_columns,
                                    read_measures, read_params, write_measures)
from tests.sweep_fixture import make_sweep

SPECS = [({"fa_threshold": 0.0}, False, 0), ({"fa_threshold": 0.1}, False, 0),
         ({"fa_threshold": 0.2}, False, 60),          # gated later: density above 0.6
         ({"fa_threshold": None}, True, 0)]


def _path4(w=1.0):
    m = np.zeros((4, 4))
    for i in range(3):
        m[i, i + 1] = m[i + 1, i] = w
    return m


def test_a_path_graph_has_hand_computable_extra_measures():
    got = extra_measures(_path4(2.0))
    assert set(got) == set(EXTRA)
    assert got["local_efficiency(binary)"] == 0.0
    assert got["mean_strength(weighted)"] == pytest.approx(3.0)
    assert got["characteristic_path_length(binary)"] == pytest.approx(10 / 6)
    assert got["mean_betweenness(binary)"] == pytest.approx(1 / 3)
    assert got["degree_assortativity(binary)"] == pytest.approx(-0.5)


def test_a_complete_graph_is_maximal_locally_and_has_undefined_assortativity():
    got = extra_measures(np.ones((5, 5)) - np.eye(5))
    assert got["local_efficiency(binary)"] == pytest.approx(1.0)
    assert got["characteristic_path_length(binary)"] == pytest.approx(1.0)
    assert got["mean_betweenness(binary)"] == 0.0
    assert math.isnan(got["degree_assortativity(binary)"])


def test_path_length_uses_the_largest_connected_component():
    m = np.zeros((5, 5))
    m[0, 1] = m[1, 0] = m[1, 2] = m[2, 1] = 1.0
    assert extra_measures(m)["characteristic_path_length(binary)"] == pytest.approx(4 / 3)


def test_matrix_row_has_the_standard_measures_and_the_isolated_fraction():
    row = matrix_row(_path4(), extended=False)
    assert "density" in row and "global_efficiency(weighted)" in row
    assert not set(EXTRA) & set(row)
    m = np.zeros((4, 4)); m[0, 1] = m[1, 0] = m[1, 2] = m[2, 1] = 1.0     # node 3 isolated
    assert matrix_row(m, extended=False)["isolated_fraction"] == pytest.approx(0.25)
    assert set(EXTRA) <= set(matrix_row(_path4(), extended=True))


def test_find_wave_handles_one_wave_two_waves_and_a_named_wave(tmp_path):
    make_sweep(tmp_path, SPECS, n_subjects=5, wave="wave1")
    assert find_wave(tmp_path).name == "wave1"
    make_sweep(tmp_path, SPECS, n_subjects=5, wave="wave2")
    with pytest.raises(ValueError, match="2 waves"):
        find_wave(tmp_path)
    assert find_wave(tmp_path, "wave2").name == "wave2"


def test_write_measures_makes_one_row_per_specification_and_subject(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=5)
    out = write_measures(tmp_path / "opt", workers=1)
    rows = read_measures(out)
    assert len(rows) == 4 * 5
    assert {r["subject"] for r in rows} == {f"sub-0{i}" for i in range(1, 6)}
    assert "sub-01_ses-1_dwi" in {r["scan"] for r in rows} and isinstance(rows[0]["density"], float)
    assert "isolated_fraction" not in measure_columns(rows) and "density" in measure_columns(rows)
    side = json.loads((out / "measures.json").read_text())
    assert side["atlas"] == "AAL3" and side["metric"] == "count" and side["repeat"] == 1
    assert side["wave"] == "wave1" and side["extended"] is False


def test_params_records_the_reference_and_the_swept_parameters(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=5)
    params = read_params(write_measures(tmp_path / "opt", workers=1))
    assert params["sweep_0004"]["reference"] is True and params["sweep_0001"]["reference"] is False
    assert params["sweep_0002"]["parameters"] == {"fa_threshold": "0.1"}


def test_extended_adds_the_five_measures_to_the_table(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=5)
    rows = read_measures(write_measures(tmp_path / "opt", extended=True, workers=1))
    assert set(EXTRA) <= set(measure_columns(rows))


def test_an_ambiguous_metric_is_an_error_until_chosen(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=5, metrics=("count", "qa"))
    with pytest.raises(ValueError, match="more than one.*count.*qa"):
        write_measures(tmp_path / "opt", workers=1)
    out = write_measures(tmp_path / "opt", metric="qa", workers=1)
    assert json.loads((out / "measures.json").read_text())["metric"] == "qa"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_measures_table.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'scripts.measures_table'`.

- [ ] **Step 4: Implement**

```python
# scripts/measures_table.py
"""Tidy table of graph measures per subject and specification, from a completed sweep.

    opticonn measures -i <optimize> [--atlas A] [--metric M] [--repeat N] [--extended]

Writes <optimize>/measures/measures.csv, params.csv and measures.json; `outcome` and `pca` read
them, so the slow step (reading every matrix) runs once.
Design: docs/superpowers/specs/2026-10-22-outcome-sensitivity-design.md
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import networkx as nx
import numpy as np

from scripts.compute_network_measures_from_connectivity import _binary_graph, measures_from_matrix
from scripts.reliability import collect_matrices
from scripts.utils.discovery import parse_subject_session

EXTRA = ["local_efficiency(binary)", "mean_strength(weighted)", "characteristic_path_length(binary)",
         "mean_betweenness(binary)", "degree_assortativity(binary)"]
ID_COLUMNS = ("combo_id", "subject", "scan")


def extra_measures(mat: np.ndarray) -> dict:
    """The five extra global measures of one connectivity matrix (non-finite -> 0, zero diagonal)."""
    mat = np.nan_to_num(np.asarray(mat, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).copy()
    np.fill_diagonal(mat, 0.0)
    g = _binary_graph(mat)
    big = g.subgraph(max(nx.connected_components(g), key=len)) if g.number_of_edges() else g
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # constant degrees: assortativity is NaN
        return {
            "local_efficiency(binary)": float(nx.local_efficiency(g)),
            "mean_strength(weighted)": float(mat.sum(axis=1).mean()),
            "characteristic_path_length(binary)": (float(nx.average_shortest_path_length(big))
                                                   if big.number_of_nodes() > 1 else float("nan")),
            "mean_betweenness(binary)": float(np.mean(list(nx.betweenness_centrality(g).values()))),
            "degree_assortativity(binary)": float(nx.degree_assortativity_coefficient(g)),
        }


def _isolated_fraction(mat: np.ndarray) -> float:
    edges = np.nan_to_num(np.asarray(mat, dtype=float)) != 0
    np.fill_diagonal(edges, False)
    return float(np.mean(~edges.any(axis=0) & ~edges.any(axis=1)))


def matrix_row(mat: np.ndarray, extended: bool) -> dict:
    """The seven standard measures, the isolated-node fraction and, if extended, the five more."""
    row = measures_from_matrix(mat)
    row["isolated_fraction"] = _isolated_fraction(mat)
    if extended:
        row.update(extra_measures(mat))
    return row


def find_wave(optimize_dir, wave: str | None = None) -> Path:
    waves = sorted({p.parent for p in Path(optimize_dir).glob("*/combos")})
    if wave:
        match = [w for w in waves if w.name == wave]
        if not match:
            raise ValueError(f"no wave named {wave}; waves are {', '.join(w.name for w in waves)}")
        return match[0]
    if len(waves) != 1:
        raise ValueError(f"{optimize_dir} has {len(waves)} waves ({', '.join(w.name for w in waves)}); "
                         "choose one with --wave")
    return waves[0]


def read_meta(wave_dir: Path) -> dict:
    """{combo name: {"reference": bool, "parameters": dict}} from each combo's diagnostics.json."""
    meta = {}
    for path in sorted(Path(wave_dir).glob("combos/sweep_*/diagnostics.json")):
        d = json.loads(path.read_text())
        meta[path.parent.name] = {"reference": bool(d.get("reference")), "parameters": d.get("parameters") or {}}
    return meta


def _combo(args):
    name, combo_dir, atlas, metric, repeat, extended = args
    rows = []
    for scan, reps in collect_matrices(combo_dir)[(atlas, metric)].items():
        if len(reps) < repeat:
            continue
        rows.append({"combo_id": name, "subject": parse_subject_session(scan)[0] or scan, "scan": scan,
                     **matrix_row(reps[repeat - 1], extended)})
    return rows


def write_measures(optimize_dir, out_dir=None, atlas=None, metric=None, repeat=1, extended=False,
                   workers=8, wave=None) -> Path:
    optimize_dir = Path(optimize_dir)
    wave_dir = find_wave(optimize_dir, wave)
    combos = sorted((wave_dir / "combos").glob("sweep_*"))
    if not combos:
        raise ValueError(f"{wave_dir}: no specifications found")
    pairs = sorted(collect_matrices(combos[0]))
    chosen = [p for p in pairs if (atlas in (None, p[0])) and (metric in (None, p[1]))]
    if len(chosen) != 1:
        found = ", ".join(f"{a}/{m}" for a, m in pairs)
        raise ValueError(f"more than one atlas/metric in the sweep ({found}); choose with --atlas/--metric"
                         if chosen else f"no atlas/metric matches --atlas/--metric; the sweep has {found}")
    atlas, metric = chosen[0]
    with ProcessPoolExecutor(workers) as pool:
        rows = [r for part in pool.map(_combo, [(c.name, c, atlas, metric, repeat, extended) for c in combos])
                for r in part]
    if not rows:
        raise ValueError(f"no subject has repeat {repeat}")
    out = Path(out_dir) if out_dir else optimize_dir / "measures"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "measures.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = read_meta(wave_dir)
    names = list(dict.fromkeys(k for m in meta.values() for k in m["parameters"]))
    with open(out / "params.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["combo_id", "reference", *names])
        writer.writeheader()
        for cid, m in meta.items():
            writer.writerow({"combo_id": cid, "reference": m["reference"],
                             **{k: "" if v is None else v for k, v in m["parameters"].items()}})
    try:
        from importlib.metadata import version
        ver = version("opticonn")
    except Exception:
        ver = "unknown"
    (out / "measures.json").write_text(json.dumps(
        {"atlas": atlas, "metric": metric, "repeat": repeat, "extended": extended,
         "wave": wave_dir.name, "opticonn_version": ver}, indent=2) + "\n")
    return out


def read_measures(measures_dir) -> list[dict]:
    path = Path(measures_dir) / "measures.csv"
    if not path.exists():
        raise ValueError(f"{path} not found; run `opticonn measures -i <optimize>` first")
    with open(path, newline="") as handle:
        return [{k: (v if k in ID_COLUMNS else float(v)) for k, v in r.items()} for r in csv.DictReader(handle)]


def measure_columns(rows: list[dict]) -> list[str]:
    return [k for k in rows[0] if k not in ID_COLUMNS and k != "isolated_fraction"]


def read_params(measures_dir) -> dict:
    """{combo: {"reference": bool, "parameters": {name: str}}} from params.csv."""
    with open(Path(measures_dir) / "params.csv", newline="") as handle:
        return {r["combo_id"]: {"reference": r["reference"] == "True",
                                "parameters": {k: v for k, v in r.items() if k not in ("combo_id", "reference")}}
                for r in csv.DictReader(handle)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="opticonn measures", description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--input", required=True, help="a sweep's optimize/ directory")
    ap.add_argument("-o", "--output-dir", default=None, help="default <input>/measures")
    ap.add_argument("--wave", default=None)
    ap.add_argument("--atlas", default=None)
    ap.add_argument("--metric", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--extended", action="store_true", help="add local efficiency, mean strength, path length, "
                    "mean betweenness and degree assortativity")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    try:
        out = write_measures(a.input, a.output_dir, a.atlas, a.metric, a.repeat, a.extended, a.workers, a.wave)
    except ValueError as err:
        raise SystemExit(f"opticonn measures: {err}")
    print(f"wrote {out}/measures.csv, params.csv, measures.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_measures_table.py -q`
Expected: `9 passed`.

- [ ] **Step 6: Commit**

```bash
git add tests/sweep_fixture.py scripts/measures_table.py tests/test_measures_table.py
git commit -m "feat(measures): tidy per-subject, per-specification measures table with five optional extra measures

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Outcome curve (`opticonn outcome`)

**Files:**
- Modify: `scripts/outcome_curve.py` (add the rows, summary, plot and CLI)
- Test: `tests/test_outcome_curve.py` (append)

**Interfaces:**
- Consumes: `partial_spearman` (Task 2); `read_measures`, `read_params`, `measure_columns` (Task 3); `load_variables` (Task 1); `scripts.reliability.DEFAULT_GATES`, `gate_reason`.
- Produces: `outcome_rows(measures, meta, variables, predictor, covariates, measure_names, gates=DEFAULT_GATES) -> list[dict]` (one row per specification with keys `combo_id, reference, density, isolated_fraction, gate, n_subjects, n_missing_variables` and, per measure `m`, `n:m, rho:m, p:m, ci_low:m, ci_high:m`); `summarize_outcome(rows, measure_names, coding) -> dict`; `plot_outcome(rows, measure, path) -> None`; `main(argv=None) -> int`. `meta` is the `read_params` shape.

- [ ] **Step 1: Append the failing tests**

```python
# append to tests/test_outcome_curve.py
import json

from scripts.measures_table import read_measures, read_params, write_measures
from scripts.outcome_curve import main, outcome_rows, summarize_outcome
from tests.sweep_fixture import make_sweep

SPECS = [({"fa_threshold": 0.0}, False, 0), ({"fa_threshold": 0.1}, False, 0),
         ({"fa_threshold": 0.2}, False, 60), ({"fa_threshold": None}, True, 0)]


def _setup(tmp_path, n=15):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=n)
    out = write_measures(tmp_path / "opt", workers=1)
    (tmp_path / "participants.tsv").write_text(
        "participant_id\tage\tsex\n" + "".join(
            f"{i + 1:02d}\t{20 + 3 * i}\t{'M' if i % 2 else 'F'}\n" for i in range(n)))
    return tmp_path / "opt", out


def _rows(tmp_path, **kw):
    from scripts.variables import load_variables
    opt, out = _setup(tmp_path)
    variables, coding = load_variables(tmp_path / "participants.tsv", ["age", "sex"])
    rows = outcome_rows(read_measures(out), read_params(out), variables, "age", ["sex"], ["density"], **kw)
    return rows, coding


def test_each_specification_gets_a_gate_verdict_and_the_effect_if_it_passes(tmp_path):
    rows, _ = _rows(tmp_path)
    by = {r["combo_id"]: r for r in rows}
    assert by["sweep_0001"]["gate"] == "" and by["sweep_0001"]["rho:density"] == pytest.approx(1.0)
    assert by["sweep_0001"]["n:density"] == 15 and by["sweep_0001"]["n_missing_variables"] == 0
    assert "density" in by["sweep_0003"]["gate"] and by["sweep_0003"]["rho:density"] is None
    assert by["sweep_0004"]["reference"] is True and by["sweep_0004"]["rho:density"] == pytest.approx(1.0)


def test_a_subject_without_the_predictor_is_dropped_and_counted(tmp_path):
    from scripts.variables import load_variables
    opt, out = _setup(tmp_path)
    (tmp_path / "participants.tsv").write_text(
        (tmp_path / "participants.tsv").read_text().replace("\t20\t", "\tn/a\t", 1))
    variables, _ = load_variables(tmp_path / "participants.tsv", ["age", "sex"])
    rows = outcome_rows(read_measures(out), read_params(out), variables, "age", ["sex"], ["density"])
    first = next(r for r in rows if r["combo_id"] == "sweep_0001")
    assert first["n_missing_variables"] == 1 and first["n:density"] == 14


def test_the_summary_describes_the_curve_the_gates_and_the_reference(tmp_path):
    rows, coding = _rows(tmp_path)
    s = summarize_outcome(rows, ["density"], coding)
    assert s["n_specifications"] == 2 and s["n_gated"] == 1
    assert s["coding"] == {"sex": {"0": "F", "1": "M"}}
    m = s["measures"]["density"]
    assert m["rho_min"] == pytest.approx(1.0) and m["n_opposite_sign"] == 0
    assert m["fraction_p_below_05"] == 1.0 and m["reference_rho"] == pytest.approx(1.0)
    assert s["reference_gate"] == "" and s["reference_density_percentile"] is not None


def test_the_command_writes_the_table_summary_and_figure(tmp_path):
    opt, out = _setup(tmp_path)
    assert main(["-i", str(opt), "--variables", str(tmp_path / "participants.tsv"), "--predictor", "age",
                 "--covariate", "sex", "--measure", "density"]) == 0
    o = opt / "outcome"
    assert (o / "outcome.csv").exists() and (o / "outcome.png").stat().st_size > 0
    assert json.loads((o / "outcome_summary.json").read_text())["n_specifications"] == 2


def test_missing_measures_table_and_unknown_measure_are_clear_errors(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=5)
    (tmp_path / "v.tsv").write_text("participant_id\tage\n01\t20\n")
    with pytest.raises(SystemExit, match="opticonn measures"):
        main(["-i", str(tmp_path / "opt"), "--variables", str(tmp_path / "v.tsv"), "--predictor", "age",
              "--measure", "density"])
    opt, _ = _setup(tmp_path / "x")
    with pytest.raises(SystemExit, match="no measure named nope"):
        main(["-i", str(opt), "--variables", str(tmp_path / "x" / "participants.tsv"), "--predictor", "age",
              "--measure", "nope"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_outcome_curve.py -q`
Expected: the new tests fail with `ImportError: cannot import name 'outcome_rows'`.

- [ ] **Step 3: Implement (append to `scripts/outcome_curve.py`; add these imports at the top)**

```python
# add to the imports at the top of scripts/outcome_curve.py
import argparse
import csv
import json
import math
from pathlib import Path

from scipy.stats import spearmanr

from scripts.measures_table import measure_columns, read_measures, read_params
from scripts.reliability import DEFAULT_GATES, gate_reason
from scripts.variables import load_variables

STATS = ("n", "rho", "p", "ci_low", "ci_high")
```

```python
def outcome_rows(measures, meta, variables, predictor, covariates, measure_names, gates=DEFAULT_GATES) -> list[dict]:
    """One row per specification: its gate verdict and, if it passes, the effect of each measure."""
    by_combo: dict[str, list[dict]] = {}
    for r in measures:
        by_combo.setdefault(r["combo_id"], []).append(r)
    needed = [predictor, *covariates]
    rows = []
    for cid in sorted(by_combo):
        subj = by_combo[cid]
        density = float(np.mean([r["density"] for r in subj]))
        isolated = float(np.mean([r["isolated_fraction"] for r in subj]))
        usable = [r for r in subj if r["subject"] in variables
                  and all(variables[r["subject"]][c] is not None for c in needed)]
        row = {"combo_id": cid, "reference": meta[cid]["reference"], "density": density,
               "isolated_fraction": isolated,
               "gate": gate_reason(density, isolated, gates["density_range"], gates["max_isolated_fraction"]),
               "n_subjects": len(subj), "n_missing_variables": len(subj) - len(usable)}
        for name in measure_names:
            row.update({f"{s}:{name}": None for s in STATS})
            ok = [r for r in usable if not math.isnan(r[name])]
            if not row["gate"] and len(ok) >= len(covariates) + 4:
                fit = partial_spearman([variables[r["subject"]][predictor] for r in ok], [r[name] for r in ok],
                                       [[variables[r["subject"]][c] for r in ok] for c in covariates])
                row.update({f"{s}:{name}": fit[s] for s in STATS})
        rows.append(row)
    return rows


def summarize_outcome(rows, measure_names, coding) -> dict:
    """Headline numbers of each measure's curve over the specifications that pass the gates."""
    cands = [r for r in rows if not r["reference"]]
    kept = [r for r in cands if not r["gate"]]
    ref = next((r for r in rows if r["reference"]), None)
    dens_all = np.array([r["density"] for r in kept])
    out = {"n_specifications": len(kept), "n_gated": len(cands) - len(kept), "coding": coding,
           "n_missing_variables_max": max((r["n_missing_variables"] for r in rows), default=0),
           "reference_gate": ref["gate"] if ref else None,
           "reference_density_percentile": float(np.mean(dens_all < ref["density"])) if ref and len(kept) else None,
           "measures": {}}
    for name in measure_names:
        fit = [r for r in kept if r[f"rho:{name}"] is not None and not math.isnan(r[f"rho:{name}"])]
        if not fit:
            out["measures"][name] = {"note": "no specification with an estimable effect"}
            continue
        rho = np.array([r[f"rho:{name}"] for r in fit])
        dens = np.array([r["density"] for r in fit])
        out["measures"][name] = {
            "rho_min": float(rho.min()), "rho_max": float(rho.max()), "rho_median": float(np.median(rho)),
            "n_opposite_sign": int(np.sum(rho * np.median(rho) < 0)),
            "fraction_p_below_05": float(np.mean([r[f"p:{name}"] < 0.05 for r in fit])),
            "rho_vs_density": float(spearmanr(rho, dens)[0]) if len(fit) > 2 else None,
            "reference_rho": ref[f"rho:{name}"] if ref else None,
            "reference_p": ref[f"p:{name}"] if ref else None}
    return out


def plot_outcome(rows, measure, path) -> None:
    """Left: the effect in every specification, sorted, with intervals. Right: effect against density."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kept = sorted((r for r in rows if not r["reference"] and not r["gate"] and r[f"rho:{measure}"] is not None),
                  key=lambda r: r[f"rho:{measure}"])
    ref = next((r for r in rows if r["reference"]), None)
    rho = np.array([r[f"rho:{measure}"] for r in kept])
    p = np.array([r[f"p:{measure}"] for r in kept])
    x = np.arange(len(kept))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    a.vlines(x, [r[f"ci_low:{measure}"] for r in kept], [r[f"ci_high:{measure}"] for r in kept], color="0.8", lw=1)
    a.scatter(x[p < 0.05], rho[p < 0.05], s=14, color="C0", label="p < 0.05")
    a.scatter(x[p >= 0.05], rho[p >= 0.05], s=14, facecolors="none", edgecolors="C0", label="p >= 0.05")
    b.scatter([r["density"] for r in kept], rho, s=14, color="C0")
    if ref and ref[f"rho:{measure}"] is not None:
        a.axhline(ref[f"rho:{measure}"], color="C3", ls="--", label="reference specification")
        b.scatter([ref["density"]], [ref[f"rho:{measure}"]], marker="D", color="C3", label="reference")
    a.axhline(0, color="k", lw=0.5)
    a.set(xlabel="specification (sorted by effect)", ylabel=f"partial Spearman rho, {measure}")
    b.set(xlabel="mean connectome density", ylabel="rho")
    a.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_rows(rows, path) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="opticonn outcome", description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--input", required=True, help="a sweep's optimize/ directory")
    ap.add_argument("--measures-dir", default=None, help="default <input>/measures")
    ap.add_argument("--variables", required=True, help="table with a participant_id column")
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--covariate", action="append", default=[])
    ap.add_argument("--measure", action="append", required=True, help="repeatable; the first is plotted")
    ap.add_argument("--density-range", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"))
    ap.add_argument("--max-isolated", type=float, default=None)
    ap.add_argument("-o", "--output-dir", default=None, help="default <input>/outcome")
    a = ap.parse_args(argv)
    mdir = Path(a.measures_dir or Path(a.input) / "measures")
    try:
        measures, meta = read_measures(mdir), read_params(mdir)
        unknown = [m for m in a.measure if m not in measure_columns(measures)]
        if unknown:
            raise ValueError(f"no measure named {', '.join(unknown)}; measures are {', '.join(measure_columns(measures))}")
        variables, coding = load_variables(a.variables, [a.predictor, *a.covariate])
    except ValueError as err:
        raise SystemExit(f"opticonn outcome: {err}")
    gates = {"density_range": a.density_range or DEFAULT_GATES["density_range"],
             "max_isolated_fraction": DEFAULT_GATES["max_isolated_fraction"] if a.max_isolated is None else a.max_isolated}
    rows = outcome_rows(measures, meta, variables, a.predictor, a.covariate, a.measure, gates)
    if not any(not r["gate"] and not r["reference"] for r in rows):
        raise SystemExit("opticonn outcome: no specification passes the gates")
    out = Path(a.output_dir or Path(a.input) / "outcome")
    out.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, out / "outcome.csv")
    summary = summarize_outcome(rows, a.measure, coding)
    (out / "outcome_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_outcome(rows, a.measure[0], out / "outcome.png")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_outcome_curve.py -q`
Expected: `10 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/outcome_curve.py tests/test_outcome_curve.py
git commit -m "feat(outcome): effect of any measure on any predictor in every specification

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: PCA (`opticonn pca`)

**Files:**
- Create: `scripts/pca_curve.py`
- Test: `tests/test_pca_curve.py`

**Interfaces:**
- Consumes: `partial_spearman` (Task 2); `read_measures`, `read_params`, `measure_columns` (Task 3); `load_variables` (Task 1).
- Produces: `pca_row(combo_id, subjects, names, anchor, variables=None, predictor=None, covariates=()) -> dict` (keys `combo_id, n, n_dropped, reason, evr1, evr2, rho1, p1, rho2, p2, load1:<m>, load2:<m>`); `summarize_pca(rows, names) -> dict`; `separating_parameters(rows, params) -> dict`; `main(argv=None) -> int`. `subjects` is `{subject: {measure: float}}`; `params` is `{combo_id: {param: value}}`; rows passed to `summarize_pca` carry `reference` and `gate` keys.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pca_curve.py
import json

import numpy as np
import pytest

from scripts.measures_table import write_measures
from scripts.pca_curve import main, pca_row, separating_parameters, summarize_pca
from tests.sweep_fixture import make_sweep

NAMES = ["a", "b", "c", "d", "e", "f", "g"]
ANCHOR = "c"


def _planted(n=60, sign=-1.0, seed=0):
    """Measures a, b, c share one factor that changes with the predictor; the others are noise."""
    rng = np.random.default_rng(seed)
    age = np.linspace(20, 80, n)
    factor = sign * (age - 50) / 30 + 0.3 * rng.standard_normal(n)
    subjects, variables = {}, {}
    for i in range(n):
        row = {m: float(rng.standard_normal()) for m in NAMES}
        for m in "abc":
            row[m] = float(factor[i] + 0.2 * rng.standard_normal())
        subjects[f"sub-{i:02d}"] = row
        variables[f"sub-{i:02d}"] = {"age": float(age[i]), "sex": float(i % 2)}
    return subjects, variables


def test_pc1_picks_up_the_shared_factor_and_its_association_with_the_predictor():
    subjects, variables = _planted(sign=-1.0)
    row = pca_row("s1", subjects, NAMES, ANCHOR, variables, "age", ["sex"])
    assert row["reason"] == "" and row["n"] == 60 and row["n_dropped"] == 0
    assert row["evr1"] > 0.35 and row["load1:c"] > 0
    assert row["rho1"] < -0.8 and row["p1"] < 0.001
    assert abs(row["load1:f"]) < 0.3


def test_the_sign_convention_does_not_depend_on_the_sign_of_the_data():
    subjects, variables = _planted(sign=-1.0)
    a = pca_row("a", subjects, NAMES, ANCHOR, variables, "age")
    subjects, variables = _planted(sign=+1.0)
    b = pca_row("b", subjects, NAMES, ANCHOR, variables, "age")
    assert a["load1:c"] > 0 and b["load1:c"] > 0 and a["rho1"] < 0 < b["rho1"]


def test_without_a_predictor_the_components_are_still_reported():
    subjects, _ = _planted()
    row = pca_row("s1", subjects, NAMES, ANCHOR)
    assert row["evr1"] > 0.35 and row["rho1"] is None


def test_a_subject_with_an_undefined_measure_is_dropped_and_counted():
    subjects, variables = _planted()
    subjects["sub-00"]["f"] = float("nan")
    row = pca_row("s1", subjects, NAMES, ANCHOR, variables, "age")
    assert row["n"] == 59 and row["n_dropped"] == 1


def test_a_constant_measure_is_reported_by_name_and_not_analysed():
    subjects, variables = _planted()
    for m in subjects.values():
        m["e"] = 1.0
    row = pca_row("s1", subjects, NAMES, ANCHOR, variables, "age")
    assert "e" in row["reason"] and row["evr1"] is None


def _prow(i, evr1, rho1, p1, load, ref=False, gate=""):
    r = {"combo_id": f"s{i}", "reference": ref, "gate": gate, "reason": "", "evr1": evr1, "evr2": 0.2,
         "rho1": rho1, "p1": p1, "rho2": 0.0, "p2": 0.5}
    r.update({f"load1:{m}": v for m, v in zip(NAMES, load)})
    return r


def test_summary_reports_variance_association_and_loading_stability():
    base = [0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.1]
    rows = [_prow(1, 0.50, 0.30, 0.001, base), _prow(2, 0.60, 0.20, 0.04, base),
            _prow(3, 0.55, -0.10, 0.30, [0.5, -0.4, 0.4, 0.3, -0.3, 0.2, 0.1]), _prow(4, 0.52, 0.10, 0.20, base),
            _prow(5, 0.90, 0.9, 0.001, base, gate="dense"), _prow(6, 0.70, 0.45, 0.001, [0.1] * 7, ref=True)]
    s = summarize_pca(rows, NAMES)
    assert s["n_specifications"] == 4
    assert (s["evr1_min"], s["evr1_max"]) == (0.50, 0.60)
    assert (s["rho1_min"], s["rho1_max"]) == (-0.10, 0.30)
    assert s["rho1_n_opposite_sign"] == 1 and s["rho1_fraction_p_below_05"] == pytest.approx(0.5)
    assert 0 < s["loading_cosine_min"] < 1 and s["loading_cosine_median"] == pytest.approx(1.0, abs=1e-6)
    assert s["reference_evr1"] == 0.70 and s["reference_rho1"] == 0.45


def test_separating_parameters_names_a_parameter_whose_levels_do_not_overlap():
    rows = [{"combo_id": f"c{i}", "reference": False, "gate": "", "evr1": v}
            for i, v in enumerate([0.50, 0.55, 0.60, 0.80, 0.85, 0.90])]
    rows.append({"combo_id": "gated", "reference": False, "gate": "dense", "evr1": 0.1})
    params = {f"c{i}": {"a": 0 if i < 3 else 1, "b": i % 2} for i in range(6)}
    params["gated"] = {"a": 0, "b": 0}
    assert separating_parameters(rows, params) == {"a": {0: (0.50, 0.60), 1: (0.80, 0.90)}}


def test_the_command_runs_on_a_sweep_and_writes_its_outputs(tmp_path):
    specs = [({"fa_threshold": 0.0}, False, 0), ({"fa_threshold": 0.1}, False, 2), ({"fa_threshold": 0.2}, False, 4),
             ({"fa_threshold": None}, True, 0)]
    make_sweep(tmp_path / "opt", specs, n_subjects=12)
    write_measures(tmp_path / "opt", workers=1)
    assert main(["-i", str(tmp_path / "opt"), "--measures", "density", "global_efficiency(binary)",
                 "global_efficiency(weighted)", "modularity", "--anchor", "global_efficiency(weighted)"]) == 0
    o = tmp_path / "opt" / "pca"
    assert (o / "pca.csv").exists() and (o / "pca.png").stat().st_size > 0
    assert json.loads((o / "pca_summary.json").read_text())["n_specifications"] == 3
    with pytest.raises(SystemExit, match="anchor"):
        main(["-i", str(tmp_path / "opt"), "--anchor", "nope"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_pca_curve.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'scripts.pca_curve'`.

- [ ] **Step 3: Implement**

```python
# scripts/pca_curve.py
"""PCA of the global graph measures in every specification of a sweep.

    opticonn pca -i <optimize> [--measures NAME ...] [--variables FILE --predictor COL [--covariate COL]]

Per specification: z-score the measures across subjects, principal components by SVD, PC1 and PC2
(sign fixed on the anchor measure), and, with a predictor, their partial Spearman association with it.
Design: docs/superpowers/specs/2026-10-22-outcome-sensitivity-design.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from scripts.measures_table import measure_columns, read_measures, read_params
from scripts.outcome_curve import _write_rows, partial_spearman
from scripts.reliability import DEFAULT_GATES, gate_reason
from scripts.variables import load_variables

ANCHOR = "global_efficiency(weighted)"


def pca_row(combo_id, subjects, names, anchor, variables=None, predictor=None, covariates=()) -> dict:
    """PC1 and PC2 of one specification: variance explained, loadings and, with a predictor, associations."""
    row = {"combo_id": combo_id, "n": None, "n_dropped": 0, "reason": "", "evr1": None, "evr2": None,
           "rho1": None, "p1": None, "rho2": None, "p2": None}
    rows = [(s, [m[n] for n in names]) for s, m in subjects.items()]
    complete = [(s, v) for s, v in rows if not any(math.isnan(x) for x in v)]
    row["n_dropped"] = len(rows) - len(complete)
    if len(complete) < 4:
        row["reason"] = "fewer than 4 subjects with every measure"
        return row
    x = np.array([v for _, v in complete])
    constant = [n for n, sd in zip(names, x.std(axis=0, ddof=1)) if sd == 0]
    if constant:
        row["reason"] = "zero spread: " + ", ".join(constant)
        return row
    z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    _, sing, vt = np.linalg.svd(z, full_matrices=False)
    evr = sing ** 2 / np.sum(sing ** 2)
    row["n"] = len(complete)
    needed = [predictor, *covariates] if predictor else []
    ok = [i for i, (s, _) in enumerate(complete)
          if predictor and s in variables and all(variables[s][c] is not None for c in needed)]
    for k in (1, 2):
        vec = vt[k - 1] * (1.0 if vt[k - 1][names.index(anchor)] >= 0 else -1.0)
        row[f"evr{k}"] = float(evr[k - 1])
        row.update({f"load{k}:{n}": float(v) for n, v in zip(names, vec)})
        if len(ok) >= len(covariates) + 4:
            scores = z @ vec
            fit = partial_spearman([variables[complete[i][0]][predictor] for i in ok], [scores[i] for i in ok],
                                   [[variables[complete[i][0]][c] for i in ok] for c in covariates])
            row[f"rho{k}"], row[f"p{k}"] = fit["rho"], fit["p"]
    return row


def _effects(kept, rho_key, p_key, prefix):
    kept = [r for r in kept if r[rho_key] is not None]
    if not kept:
        return {}
    rho = np.array([r[rho_key] for r in kept])
    return {f"{prefix}_min": float(rho.min()), f"{prefix}_max": float(rho.max()),
            f"{prefix}_n_opposite_sign": int(np.sum(rho * np.median(rho) < 0)),
            f"{prefix}_fraction_p_below_05": float(np.mean([r[p_key] < 0.05 for r in kept]))}


def summarize_pca(rows, names) -> dict:
    """Headline numbers over the specifications that pass the gates and were analysed."""
    kept = [r for r in rows if not r["reference"] and not r["gate"] and r["evr1"] is not None]
    ref = next((r for r in rows if r["reference"] and r["evr1"] is not None), None)
    load = np.array([[r[f"load1:{m}"] for m in names] for r in kept])
    med = np.median(load, axis=0)
    cos = load @ med / (np.linalg.norm(load, axis=1) * np.linalg.norm(med))
    evr = np.array([r["evr1"] for r in kept])
    out = {"n_specifications": len(kept), "evr1_min": float(evr.min()), "evr1_max": float(evr.max()),
           "evr1_median": float(np.median(evr)), "evr2_median": float(np.median([r["evr2"] for r in kept])),
           "loading_cosine_min": float(cos.min()), "loading_cosine_median": float(np.median(cos))}
    out.update(_effects(kept, "rho1", "p1", "rho1"))
    out.update(_effects(kept, "rho2", "p2", "rho2"))
    out.update({"reference_evr1": ref["evr1"] if ref else None, "reference_rho1": ref["rho1"] if ref else None,
                "reference_rho2": ref["rho2"] if ref else None})
    return out


def separating_parameters(rows, params) -> dict:
    """{parameter: {level: (min, max) of PC1 variance explained}} for every swept parameter whose
    levels give non-overlapping ranges: a parameter that alone sorts the specifications into regimes."""
    kept = [r for r in rows if not r["reference"] and not r["gate"] and r["evr1"] is not None]
    out = {}
    for name in sorted({k for p in params.values() for k in p}):
        levels: dict = {}
        for r in kept:
            levels.setdefault(params[r["combo_id"]][name], []).append(r["evr1"])
        ranges = sorted((min(v), max(v), lvl) for lvl, v in levels.items())
        if len(ranges) > 1 and all(a[1] < b[0] for a, b in zip(ranges, ranges[1:])):
            out[name] = {lvl: (lo, hi) for lo, hi, lvl in ranges}
    return out


def plot_pca(rows, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kept = sorted((r for r in rows if not r["reference"] and not r["gate"] and r["evr1"] is not None),
                  key=lambda r: r["evr1"])
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.scatter(range(len(kept)), [r["evr1"] for r in kept], s=14, color="C0")
    ax.set(xlabel="specification (sorted)", ylabel="variance explained by PC1")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="opticonn pca", description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--input", required=True, help="a sweep's optimize/ directory")
    ap.add_argument("--measures-dir", default=None, help="default <input>/measures")
    ap.add_argument("--measures", nargs="+", default=None, help="default: every measure in measures.csv")
    ap.add_argument("--anchor", default=ANCHOR, help="measure whose loading is made positive on every component")
    ap.add_argument("--variables", default=None)
    ap.add_argument("--predictor", default=None)
    ap.add_argument("--covariate", action="append", default=[])
    ap.add_argument("-o", "--output-dir", default=None, help="default <input>/pca")
    a = ap.parse_args(argv)
    mdir = Path(a.measures_dir or Path(a.input) / "measures")
    try:
        measures, meta = read_measures(mdir), read_params(mdir)
        names = a.measures or measure_columns(measures)
        unknown = [m for m in [*names, a.anchor] if m not in measure_columns(measures)]
        if a.anchor in unknown:
            raise ValueError(f"the anchor measure {a.anchor} is not in measures.csv; choose another with --anchor")
        if unknown:
            raise ValueError(f"no measure named {', '.join(unknown)}")
        if bool(a.variables) != bool(a.predictor):
            raise ValueError("--variables and --predictor go together")
        variables = load_variables(a.variables, [a.predictor, *a.covariate])[0] if a.variables else None
    except ValueError as err:
        raise SystemExit(f"opticonn pca: {err}")
    by_combo: dict[str, dict] = {}
    for r in measures:
        by_combo.setdefault(r["combo_id"], {})[r["subject"]] = {n: r[n] for n in names}
    gate_inputs = {}
    for r in measures:
        gate_inputs.setdefault(r["combo_id"], []).append((r["density"], r["isolated_fraction"]))
    rows = []
    for cid, subjects in sorted(by_combo.items()):
        row = pca_row(cid, subjects, names, a.anchor, variables, a.predictor, a.covariate)
        dens = float(np.mean([d for d, _ in gate_inputs[cid]]))
        iso = float(np.mean([i for _, i in gate_inputs[cid]]))
        row.update(reference=meta[cid]["reference"],
                   gate=gate_reason(dens, iso, DEFAULT_GATES["density_range"], DEFAULT_GATES["max_isolated_fraction"]))
        rows.append(row)
    out = Path(a.output_dir or Path(a.input) / "pca")
    out.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, out / "pca.csv")
    summary = summarize_pca(rows, names)
    params = {cid: m["parameters"] for cid, m in meta.items()}
    if any(params.values()):
        summary["separating_parameters"] = {n: {str(k): v for k, v in lv.items()}
                                            for n, lv in separating_parameters(rows, params).items()}
    (out / "pca_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_pca(rows, out / "pca.png")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_pca_curve.py -q`
Expected: `9 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/pca_curve.py tests/test_pca_curve.py
git commit -m "feat(pca): per-specification PCA of the graph measures, with predictor association

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Register the commands and document them

**Files:**
- Modify: `scripts/opticonn_hub.py` (parsers next to the `view` parser near line 94; dispatch next to `if args.command == "view":` near line 795)
- Create: `tests/test_outcome_cli.py`, `docs/outcome_sensitivity.md`
- Modify: `README.md` (one short section)

**Interfaces:**
- Consumes: `scripts.measures_table.main`, `scripts.outcome_curve.main`, `scripts.pca_curve.main` (each takes `argv: list[str]`).
- Produces: `scripts.opticonn_hub.PASS_THROUGH` (command name to module name).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_outcome_cli.py
import subprocess
import sys
from pathlib import Path

from scripts.measures_table import write_measures  # noqa: F401  (import check)
from tests.sweep_fixture import make_sweep

ROOT = Path(__file__).resolve().parents[1]
SPECS = [({"fa_threshold": 0.0}, False, 0), ({"fa_threshold": 0.1}, False, 2), ({"fa_threshold": None}, True, 0)]


def _run(*args):
    return subprocess.run([sys.executable, str(ROOT / "opticonn.py"), *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=300)


def test_each_command_has_help_through_the_hub():
    for cmd, flag in (("measures", "--extended"), ("outcome", "--predictor"), ("pca", "--anchor")):
        out = _run(cmd, "--help")
        assert out.returncode == 0, out.stderr
        assert flag in out.stdout


def test_the_three_commands_run_in_sequence_on_a_sweep(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=12)
    (tmp_path / "participants.tsv").write_text(
        "participant_id\tage\tsex\n" + "".join(f"{i + 1:02d}\t{20 + 3 * i}\t{'M' if i % 2 else 'F'}\n" for i in range(12)))
    opt = str(tmp_path / "opt")
    assert _run("measures", "-i", opt, "--workers", "1").returncode == 0
    out = _run("outcome", "-i", opt, "--variables", str(tmp_path / "participants.tsv"), "--predictor", "age",
               "--covariate", "sex", "--measure", "density")
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "opt" / "outcome" / "outcome.csv").exists()
    assert _run("pca", "-i", opt, "--measures", "density", "global_efficiency(binary)", "global_efficiency(weighted)",
                "modularity", "--variables", str(tmp_path / "participants.tsv"), "--predictor", "age").returncode == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_outcome_cli.py -q`
Expected: FAIL (`invalid choice: 'measures'`).

- [ ] **Step 3: Register the subcommands**

In `scripts/opticonn_hub.py`, add a module constant after the imports (after the line `from scripts.utils.runtime import configure_stdio, propagate_no_emoji`):

```python
# measures / outcome / pca define and document their own arguments; the hub only forwards them.
PASS_THROUGH = {"measures": "measures_table", "outcome": "outcome_curve", "pca": "pca_curve"}
```

as the first statements of `main()` (before `parser = argparse.ArgumentParser(`):

```python
    if len(sys.argv) > 1 and sys.argv[1] in PASS_THROUGH:
        import importlib

        return importlib.import_module(f"scripts.{PASS_THROUGH[sys.argv[1]]}").main(sys.argv[2:])
```

and, directly after the `p_view` parser block, register them so they appear in `opticonn --help`:

```python
    for name, summary in (
        ("measures", "Table of graph measures per subject and specification from a completed sweep"),
        ("outcome", "How an association (any predictor, any measure) varies across the specifications of a sweep"),
        ("pca", "PCA of the graph measures in every specification of a sweep"),
    ):
        subparsers.add_parser(name, help=summary)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_outcome_cli.py tests/test_cli_help.py -q`
Expected: all pass.

- [ ] **Step 5: Write the documentation**

Create `docs/outcome_sensitivity.md`:

````markdown
# Outcome sensitivity: how much does your estimate depend on the specification?

A completed sweep (`opticonn tune-grid`) holds one connectome per subject for every tractography
specification. These commands turn that into the question a study has: *how much does the number I
will report depend on the specification?* They are not tied to any outcome: you name the predictor,
the covariates and the measure.

```bash
# 1. one table of graph measures per subject and specification (add --extended for five more)
opticonn measures -i studies/demo/sweep-*/optimize --extended

# 2. the effect of a predictor on a measure in every specification
opticonn outcome -i studies/demo/sweep-*/optimize --variables participants.tsv \
    --predictor age --covariate sex --measure "global_efficiency(weighted)"

# 3. PCA of the measures in every specification (optionally with a predictor)
opticonn pca -i studies/demo/sweep-*/optimize --variables participants.tsv --predictor age --covariate sex
```

## Inputs

`participants.tsv` (or any delimited file) needs a `participant_id` column, with or without the
`sub-` prefix; all other column names are yours. A numeric column is used as it is; a column with
exactly two distinct values (for example `F`/`M`) is coded 0/1 in sorted order and the coding is
written to the summary; more than two levels is an error. Subjects with a missing value are
dropped from that analysis and counted.

## What you get

`outcome/outcome.csv` has one row per specification: its gate verdict (density 0.02-0.6, at most
10% isolated nodes; override with `--density-range` and `--max-isolated`), and for each measure the
partial Spearman correlation with the predictor, its p value and 95% interval. The summary gives
the range of the effect, how many specifications differ in sign from the median, the fraction with
p < 0.05, how the effect relates to density, and where the reference specification (DSI Studio's
defaults, if the sweep included it) falls. `pca/` reports variance explained, loadings and their
stability, the associations of PC1 and PC2 with a predictor, and which swept parameter, if any,
separates the specifications into regimes.

Age and sex above are only an example of a continuous and a group outcome; any column works.
````

Append to `README.md` (after the "Seeing how much the choice mattered" block) this short section:

```markdown
### How much does your own estimate depend on the specification?

```bash
opticonn measures -i studies/demo_grid/sweep-*/optimize
opticonn outcome  -i studies/demo_grid/sweep-*/optimize --variables participants.tsv \
    --predictor <column> --covariate <column> --measure "global_efficiency(weighted)"
opticonn pca      -i studies/demo_grid/sweep-*/optimize
```

Any predictor, covariate and graph measure; see [docs/outcome_sensitivity.md](docs/outcome_sensitivity.md).
```

- [ ] **Step 6: Run the whole new test set and the existing CLI tests**

Run: `python -m pytest tests/test_variables.py tests/test_outcome_curve.py tests/test_measures_table.py tests/test_pca_curve.py tests/test_outcome_cli.py tests/test_cli_help.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/opticonn_hub.py tests/test_outcome_cli.py docs/outcome_sensitivity.md README.md
git commit -m "feat(hub): register measures, outcome and pca; document them

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Acceptance against the study repository's committed results

**Files:**
- Create: `tests/test_outcome_acceptance.py`

**Interfaces:**
- Consumes: the three commands (Tasks 3-5); environment variables `OPTICONN_ACCEPT_SWEEP` (the ds005270 sweep's `optimize/` directory), `OPTICONN_ACCEPT_METADATA` (`metadata/ds005270`, containing `participants.tsv`), `OPTICONN_ACCEPT_BATTERY` (the study repository root holding `speccurve_ds005270.csv`, `pcacurve_ds005270.csv`, `pcacurve12_ds005270.csv`).

This test is skipped unless all three variables are set; it needs the 60 GB sweep and about 20 minutes.

- [ ] **Step 1: Write the test**

```python
# tests/test_outcome_acceptance.py
"""The new commands reproduce the study repository's committed numbers (proof the move changed nothing)."""

import csv
import os
from pathlib import Path

import pytest

from scripts import measures_table, outcome_curve, pca_curve

SWEEP = os.environ.get("OPTICONN_ACCEPT_SWEEP")
META = os.environ.get("OPTICONN_ACCEPT_METADATA")
BATTERY = os.environ.get("OPTICONN_ACCEPT_BATTERY")
pytestmark = pytest.mark.skipif(not (SWEEP and META and BATTERY), reason="acceptance data not configured")

PRIMARY = "global_efficiency(weighted)"
SEVEN = ["density", "global_efficiency(binary)", "global_efficiency(weighted)", "clustering_coeff_average(binary)",
         "clustering_coeff_average(weighted)", "modularity", "small_worldness(binary)"]


def _csv(path):
    with open(path, newline="") as handle:
        return {r["combo_id"]: r for r in csv.DictReader(handle)}


def _close(a, b):
    if a in ("", None) or b in ("", None):
        return a in ("", None) and b in ("", None)
    return abs(float(a) - float(b)) < 1e-9


@pytest.fixture(scope="module")
def measures(tmp_path_factory):
    out = tmp_path_factory.mktemp("accept")
    measures_table.write_measures(SWEEP, out_dir=out / "measures", extended=True, workers=16)
    return out


def test_age_and_sex_curves_match_speccurve(measures):
    variables = str(Path(META) / "participants.tsv")
    for predictor, covariate, prefix in (("age", "sex", ""), ("sex", "age", "sex_")):
        o = measures / predictor
        outcome_curve.main(["-i", SWEEP, "--measures-dir", str(measures / "measures"), "--variables", variables,
                            "--predictor", predictor, "--covariate", covariate, "--measure", PRIMARY, "-o", str(o)])
        new, old = _csv(o / "outcome.csv"), _csv(Path(BATTERY) / "speccurve_ds005270.csv")
        for cid, row in new.items():
            for stat in ("rho", "p", "ci_low", "ci_high"):
                assert _close(row[f"{stat}:{PRIMARY}"], old[cid][f"{prefix}{stat}"]), (predictor, cid, stat)


def test_the_pca_matches_the_seven_and_twelve_measure_runs(measures):
    variables = str(Path(META) / "participants.tsv")
    for names, committed in ((SEVEN, "pcacurve_ds005270.csv"), (None, "pcacurve12_ds005270.csv")):
        o = measures / committed
        argv = ["-i", SWEEP, "--measures-dir", str(measures / "measures"), "--variables", variables,
                "--predictor", "age", "--covariate", "sex", "-o", str(o)]
        pca_curve.main(argv + (["--measures", *names] if names else []))
        new, old = _csv(o / "pca.csv"), _csv(Path(BATTERY) / committed)
        for cid, row in new.items():
            if row["gate"]:
                continue
            for key in ("evr1", "evr2", "rho1", "p1", "rho2", "p2"):
                assert _close(row[key], old[cid][key]), (committed, cid, key)
```

- [ ] **Step 2: Run it without data to verify it skips**

Run: `python -m pytest tests/test_outcome_acceptance.py -q`
Expected: `2 skipped`.

- [ ] **Step 3: Run it for real once the sweeps have finished (the live checkout is busy until then)**

```bash
OPTICONN_ACCEPT_SWEEP=/data/local/software/opticonn-multiverse-battery/results_age/ds005270/sweep-age132/optimize \
OPTICONN_ACCEPT_METADATA=/data/local/software/opticonn-multiverse-battery/metadata/ds005270 \
OPTICONN_ACCEPT_BATTERY=/data/local/software/opticonn-multiverse-battery \
python -m pytest tests/test_outcome_acceptance.py -q
```

Expected: `2 passed`. Any difference beyond 1e-9 is a bug in the move, not a tolerance to widen: stop and find it (the most likely causes are the order of measures, a different gate, or the PCA subject set).

- [ ] **Step 4: Commit**

```bash
git add tests/test_outcome_acceptance.py
git commit -m "test: acceptance of the new commands against the study repository's committed results

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage.** Commands and flags: Tasks 3-5 and 6 (`measures` `-i --atlas --metric --repeat --extended --workers --wave`; `outcome` `--variables --predictor --covariate --measure --density-range --max-isolated`; `pca` `--measures --anchor --variables --predictor --covariate`). Conventions 1-6: Task 1 (1, 2), Task 2 (3), Task 4 (4, 6) and Task 3 (5, via `read_meta`). Architecture, errors, testing: Tasks 3-6; acceptance: Task 7. Out-of-scope items are not implemented.

**Placeholder scan.** None; every step has code or a command. The one correction note in Task 3 (a redundant assert) and the one in Task 5 (a fallback rewrite of two calls) are explicit.

**Type consistency.** `read_params` returns `{combo: {"reference", "parameters"}}` (Task 3) and is consumed as such in Tasks 4 and 5; `outcome_rows` row keys `rho:<m>` etc. match `summarize_outcome` and `plot_outcome`; `pca_row` keys match `summarize_pca` and `separating_parameters`; `_write_rows` is defined in Task 4 and imported in Task 5.

**Spec clarification.** `outcome` and `pca` read the measures directory rather than a wave directory, so `--wave` applies to `measures` only; a sweep with two waves gets one `measures` run per wave with `-o`, and `--measures-dir` points the other commands at it.
