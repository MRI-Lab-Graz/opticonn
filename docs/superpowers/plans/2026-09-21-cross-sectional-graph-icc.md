# Cross-sectional Optimization and Graph-measure ICC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make OptiConn cross-sectional only (one baseline scan per subject, no session features anywhere), and add a per-candidate ICC report of the graph measures a study will analyse.

**Architecture:** A new `baseline_scans()` in `scripts/utils/discovery.py` reduces any pool to one scan per subject; both DSI sampling paths (grid staging, Bayesian) use it. Every `sessions_per_subject` code path is deleted, and old configs fail loudly. The variance decomposition loses its `between_session` stratum. A new `scripts/graph_icc.py` computes one-way ICC(1,1) with F-based 95% CIs per (candidate, graph measure) from the matrices a sweep already writes. It reuses a matrix-level split of the existing network-measure code, which gains Louvain modularity. It runs from the same end-of-sweep hook as the decomposition and never affects ranking.

**Tech Stack:** Python 3.10, numpy, scipy (`scipy.stats.f`), networkx (`louvain_communities`, `modularity`), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-21-cross-sectional-graph-icc-design.md`

## Global Constraints

- Cross-sectional only: no code path may use, require or report repeat sessions.
- One scan per subject: the first session in natural order of the session id (`ses-2` before `ses-10`). Paths whose subject does not parse are kept, each as its own subject.
- `exclude_scans` is applied *after* `baseline_scans`; a subject whose baseline is excluded drops out and is never replaced by a later session.
- A config containing `data_selection.sessions_per_subject` raises `ValueError` in `run_wave_pipeline` before staging, with the message: `data_selection.sessions_per_subject was removed: OptiConn stages one scan per subject (the first session). Remove the key from <config path>.`
- `--subjects` default is 10 (optimizer and `tune-grid`); `--quick` without `--subjects` uses 3.
- Variance decomposition strata: `tracking_noise`, `parameter`, `between_subject`. Headline ratio `parameter_over_between_subject`.
- ICC: one-way random ICC(1,1), 95% CI from the F distribution. `MIN_SUBJECTS_FOR_ICC = 10` (below it, `low_confidence=True`). With fewer than 3 subjects, or zero total variance, no number is reported: empty numeric fields plus a `reason`, never zero.
- Graph measures for ICC = the default output of `measures_from_matrix`: `density`, `global_efficiency(binary)`, `clustering_coeff_average(binary)`, `small_worldness(binary)`, `clustering_coeff_average(weighted)`, `global_efficiency(weighted)`, `modularity`.
- Modularity: weighted Louvain seeded with the `seed` argument (default 42).
- Neither the ICC nor the variance decomposition may be consumed by `rank()` or `rank_with_fallback()`. Ranking stays: discriminability, margin, repeatability, fewer tracts.
- No new dependencies. Run tests with `source braingraph_pipeline/bin/activate && python -m pytest ...` from the repo root.
- Commit messages end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`. Never `git add -A`; add the files listed per task.

## File Structure

| File | Responsibility | Tasks |
| --- | --- | --- |
| `scripts/utils/discovery.py` | `baseline_scans` (new), `select_scans` (simplified) | 1 |
| `scripts/cross_validation_bootstrap_optimizer.py` | staging order, removed-key error, flag removal, `--subjects` default, ICC hook | 1, 2, 5 |
| `scripts/bayesian_optimizer.py` | Bayesian sampling pool = baseline scans | 1 |
| `scripts/opticonn_hub.py` | `tune-grid` flag removal, `--subjects` default | 2 |
| `scripts/json_validator.py` | drop the `sessions_per_subject` rule | 2 |
| `scripts/variance_decomposition.py` | three strata, new headline | 3 |
| `scripts/compute_network_measures_from_connectivity.py` | `measures_from_matrix` split + modularity | 4 |
| `scripts/graph_icc.py` (new) | ICC computation, report writer, CLI | 5 |
| `tests/test_wave_staging.py` (renamed from `test_session_selection.py`) | staging integration tests | 1, 2 |
| `tests/test_discovery.py` | discovery unit tests | 1 |
| `tests/test_opticonn_hub_backend.py` | hub forwarding tests | 2 |
| `tests/test_sessions_flag_warnings.py` | deleted | 2 |
| `tests/test_variance_decomposition.py` | updated for three strata | 3 |
| `tests/test_compute_network_measures_from_connectivity.py` | matrix split + modularity | 4 |
| `tests/test_graph_icc.py` (new) | ICC tests, hook guard | 5 |
| docs, `paper.md`, `paper.bib`, specs | documentation | 6 |

---

### Task 1: One baseline scan per subject

**Files:**
- Modify: `scripts/utils/discovery.py:77-131` (replace `select_scans`, add `baseline_scans`)
- Modify: `scripts/cross_validation_bootstrap_optimizer.py:28` (import) and `:520-540` (staging)
- Modify: `scripts/bayesian_optimizer.py:26` (import) and `:250-264` (`_get_all_subjects`)
- Rename + rewrite: `tests/test_session_selection.py` → `tests/test_wave_staging.py`
- Modify: `tests/test_discovery.py:7` (import) and `:76-149` (replace the `select_scans` block)

**Interfaces:**
- Consumes: existing `parse_subject_session(path) -> tuple[str | None, str | None]` and `_natural_key(text) -> list` in `scripts/utils/discovery.py`.
- Produces: `baseline_scans(pool: list[Path]) -> list[Path]` and `select_scans(pool: list[Path], n_subjects: int, seed: int) -> list[Path]` (the fourth parameter `sessions_per_subject` is gone). The staging log line becomes `" Staged %d scans, one per subject (n_subjects=%d)"`.

- [ ] **Step 1: Replace the `select_scans` tests in `tests/test_discovery.py`**

Change the import on line 7 to:

```python
from scripts.utils.discovery import (
    baseline_scans,
    find_subject_files,
    parse_subject_session,
    select_scans,
)
```

Replace everything from `def _scan(sub, ses, ext="fz"):` (line 76) to the end of the file with:

```python
def _scan(sub, ses, ext="fz"):
    return Path(f"/d/sub-{sub}/fib/sub-{sub}_ses-{ses}.odf.qsdr.{ext}")


_POOL = (
    [_scan("A", 1), _scan("A", 2), _scan("A", 3)]
    + [_scan("B", 1), _scan("B", 2)]
    + [_scan("C", 1)]
    + [_scan("D", 1), _scan("D", 2)]
)


def test_baseline_scans_takes_the_first_session_of_each_subject():
    assert baseline_scans(_POOL) == [_scan("A", 1), _scan("B", 1), _scan("C", 1), _scan("D", 1)]


def test_baseline_scans_orders_sessions_naturally():
    assert baseline_scans([_scan("A", 10), _scan("A", 2)]) == [_scan("A", 2)]


def test_baseline_scans_prefers_the_first_listed_copy_of_the_same_session():
    # run_wave_pipeline lists .fz before .fib.gz, so the .fz copy wins
    assert baseline_scans([_scan("A", 1), _scan("A", 1, "fib.gz")]) == [_scan("A", 1)]


def test_baseline_scans_keeps_a_session_less_scan():
    no_session = Path("/d/sub-E/fib/sub-E.odf.qsdr.fz")
    assert baseline_scans([no_session, _scan("A", 1)]) == [no_session, _scan("A", 1)]


def test_baseline_scans_keeps_unparseable_paths_as_own_subjects_with_one_warning(caplog):
    annex = Path("/d/MD5E-abc.qsdr.fz")
    with caplog.at_level(logging.WARNING):
        got = baseline_scans(_POOL + [annex])
    assert annex in got and len(got) == 5
    assert "1 of 9 scans" in caplog.text


def test_baseline_scans_preserves_input_order():
    assert baseline_scans([_scan("B", 1), _scan("A", 1)]) == [_scan("B", 1), _scan("A", 1)]


def test_select_scans_matches_seeded_random_sample():
    pool = [_scan(str(i), 1) for i in range(10)]
    random.seed(42)
    legacy = random.sample(pool, 3)
    assert select_scans(pool, 3, 42) == legacy


def test_select_scans_uses_whole_pool_when_asked_for_more_than_exists():
    pool = [_scan(str(i), 1) for i in range(4)]
    assert select_scans(pool, 10, 42) == pool


def test_select_scans_is_deterministic():
    assert select_scans(_POOL, 2, 7) == select_scans(_POOL, 2, 7)
```

- [ ] **Step 2: Run the discovery tests to verify they fail**

Run: `python -m pytest tests/test_discovery.py -q`
Expected: collection error `ImportError: cannot import name 'baseline_scans'`.

- [ ] **Step 3: Implement `baseline_scans` and the simplified `select_scans`**

In `scripts/utils/discovery.py`, replace the whole `select_scans` function (from `def select_scans(` to its final `return selected`) with:

```python
def baseline_scans(pool: list[Path]) -> list[Path]:
    """One scan per subject: the first session in natural order (ses-2 before ses-10).

    OptiConn is cross-sectional: a later session usually carries the effect a
    study measures (e.g. an intervention), so it is never used to choose
    parameters. A subject without a session id contributes its scan as is. Paths
    whose subject does not parse (git-annex hashes, synthetic test ids) are kept,
    each counted as its own subject. When two paths share a subject and session
    (a .fz and its .fib.gz copy), the first listed wins. Output keeps input order.
    """
    first: dict[str, tuple[list, Path]] = {}
    unparseable: list[Path] = []
    for path in pool:
        subject, session = parse_subject_session(path)
        if subject is None:
            unparseable.append(path)
            continue
        key = _natural_key(session or "")
        if subject not in first or key < first[subject][0]:
            first[subject] = (key, path)
    if unparseable:
        logging.warning(
            "%d of %d scans have no parseable sub-<id>; each is kept as its own subject (e.g. %s)",
            len(unparseable),
            len(pool),
            ", ".join(p.name for p in unparseable[:3]),
        )
    keep = {path for _, path in first.values()} | set(unparseable)
    return [p for p in pool if p in keep]


def select_scans(pool: list[Path], n_subjects: int, seed: int) -> list[Path]:
    """Seeded sample of n_subjects scans, or the whole pool when it has no more.

    Pass a pool from baseline_scans() so each sampled scan is a different subject.
    """
    if n_subjects >= len(pool):
        return list(pool)
    return random.Random(seed).sample(pool, n_subjects)
```

- [ ] **Step 4: Run the discovery tests to verify they pass**

Run: `python -m pytest tests/test_discovery.py -q`
Expected: all pass.

- [ ] **Step 5: Rename the staging test file and rewrite it**

```bash
git mv tests/test_session_selection.py tests/test_wave_staging.py
```

Write `tests/test_wave_staging.py` with this full content. `test_config_generators_write_sessions_per_subject` stays unchanged for now; Task 2 replaces it.

```python
"""Dry-run integration: run_wave_pipeline stages one baseline scan per subject."""

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.bayesian_optimizer import BayesianOptimizer
from scripts.cross_validation_bootstrap_optimizer import (
    generate_single_wave_config,
    generate_wave_configs,
    run_wave_pipeline,
)
from scripts.utils.discovery import baseline_scans, find_subject_files, select_scans

# subject -> number of sessions
_LAYOUT = {"001": 3, "002": 1, "003": 2, "004": 1, "005": 3, "006": 2}


def _dataset(root):
    for sub, n in _LAYOUT.items():
        d = root / f"sub-{sub}" / "fib"
        d.mkdir(parents=True)
        for ses in range(1, n + 1):
            (d / f"sub-{sub}_ses-{ses}.odf.qsdr.fz").write_bytes(b"")
    return root


def _run(tmp_path, **selection):
    data = _dataset(tmp_path / "data")
    sel = {"source_dir": str(data), "n_subjects": 2, "random_seed": 42, "file_pattern": "*.fz"}
    sel.update(selection)
    cfg = {"test_config": {"name": "w"}, "data_selection": sel}
    cfg_path = tmp_path / "wave.json"
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "out"
    run_wave_pipeline(str(cfg_path), str(out), dry_run=True)
    lines = (out / "w" / "selected_files.txt").read_text().splitlines()
    return data, [Path(x) for x in lines]


def test_wave_stages_one_baseline_scan_per_subject(tmp_path):
    _, got = _run(tmp_path, n_subjects=20)
    assert sorted(p.name for p in got) == [f"sub-{s}_ses-1.odf.qsdr.fz" for s in sorted(_LAYOUT)]


def test_wave_staging_is_the_seeded_sample_of_baseline_scans(tmp_path):
    data, got = _run(tmp_path, n_subjects=3)
    pool = baseline_scans(find_subject_files(data, ["*.fz"]))
    assert got == select_scans(pool, 3, 42)
    assert len(got) == 3


def test_excluding_a_baseline_scan_drops_the_subject(tmp_path):
    _, got = _run(tmp_path, n_subjects=20, exclude_scans=["sub-001_ses-1"])
    names = {p.name for p in got}
    assert not any(n.startswith("sub-001_") for n in names)  # no fallback to ses-2
    assert len(names) == 5


def test_config_generators_write_sessions_per_subject(tmp_path):
    def sel(path):
        return json.loads(Path(path).read_text())["data_selection"]["sessions_per_subject"]

    w1, w2 = generate_wave_configs("d", tmp_path / "a")
    assert sel(w1) == 2 and sel(w2) == 2
    w1, w2 = generate_wave_configs("d", tmp_path / "b", sessions_per_subject=1)
    assert sel(w1) == 1 and sel(w2) == 1
    assert sel(generate_single_wave_config("d", tmp_path / "c")) == 2
    assert sel(generate_single_wave_config("d", tmp_path / "d", sessions_per_subject=3)) == 3


def test_staging_warns_when_two_selected_scans_share_a_file_name(tmp_path, caplog):
    data = tmp_path / "data"
    for sub in ("007", "008"):
        d = data / f"sub-{sub}" / "ses-1"
        d.mkdir(parents=True)
        (d / "dwi.odf.qsdr.fz").write_bytes(b"")  # same name, different subjects
    cfg = {
        "test_config": {"name": "w"},
        "data_selection": {
            "source_dir": str(data), "n_subjects": 2, "random_seed": 42, "file_pattern": "*.fz",
        },
    }
    cfg_path = tmp_path / "wave.json"
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "out"
    with caplog.at_level("WARNING"):
        run_wave_pipeline(str(cfg_path), str(out), dry_run=True)
    listed = (out / "w" / "selected_files.txt").read_text().splitlines()
    assert len(listed) == 2
    assert len(list((out / "w" / "selected_data").iterdir())) == 1
    assert sum("fewer scans than selected" in r.getMessage() for r in caplog.records) == 1


def test_staging_logs_scan_count_one_per_subject(tmp_path, caplog):
    with caplog.at_level("INFO"):
        _run(tmp_path)
    assert any("Staged 2 scans, one per subject (n_subjects=2)" in r.getMessage() for r in caplog.records)


def test_bayesian_sampling_pool_holds_one_baseline_scan_per_subject(tmp_path):
    data = _dataset(tmp_path / "data")
    got = BayesianOptimizer._get_all_subjects(SimpleNamespace(data_dir=data))
    assert sorted(p.name for p in got) == [f"sub-{s}_ses-1.odf.qsdr.fz" for s in sorted(_LAYOUT)]
```

- [ ] **Step 6: Run the staging tests to verify they fail**

Run: `python -m pytest tests/test_wave_staging.py -q`
Expected: several FAIL. `run_wave_pipeline` still calls `select_scans` with 4 arguments (`TypeError`), and the Bayesian test lists all 12 files.

- [ ] **Step 7: Update staging in `run_wave_pipeline`**

In `scripts/cross_validation_bootstrap_optimizer.py`, change the import on line 28 to:

```python
from scripts.utils.discovery import baseline_scans, find_subject_files, select_scans
```

Replace this block:

```python
    pool = fz_files + fib_files
    exclude = {str(x) for x in wave_config["data_selection"].get("exclude_scans") or []}
    if exclude:
        # scan ids look like "sub-043_ses-1" (see scripts/qc_gate.py)
        kept = [
            p for p in pool
            if not any(re.match(re.escape(e) + r"(?![A-Za-z0-9])", p.name) for e in exclude)
        ]
        logging.info(" Excluded %d scans listed in exclude_scans", len(pool) - len(kept))
        pool = kept
    if not pool:
        logging.error(" No candidate files found for selection")
        return False
    sessions_per_subject = int(wave_config["data_selection"].get("sessions_per_subject") or 0)
    selected = select_scans(pool, n_subjects, seed, sessions_per_subject)
    logging.info(
        " Staged %d scans (n_subjects=%d, sessions_per_subject=%d)",
        len(selected), n_subjects, sessions_per_subject,
    )
```

with:

```python
    # Cross-sectional: one baseline scan per subject. Exclusion comes after, so a
    # subject whose baseline is excluded drops out instead of falling back to a
    # later (possibly post-intervention) session.
    pool = baseline_scans(fz_files + fib_files)
    exclude = {str(x) for x in wave_config["data_selection"].get("exclude_scans") or []}
    if exclude:
        # scan ids look like "sub-043_ses-1" (see scripts/qc_gate.py)
        kept = [
            p for p in pool
            if not any(re.match(re.escape(e) + r"(?![A-Za-z0-9])", p.name) for e in exclude)
        ]
        logging.info(
            " Excluded %d baseline scans listed in exclude_scans (those subjects are dropped)",
            len(pool) - len(kept),
        )
        pool = kept
    if not pool:
        logging.error(" No candidate files found for selection")
        return False
    selected = select_scans(pool, n_subjects, seed)
    logging.info(" Staged %d scans, one per subject (n_subjects=%d)", len(selected), n_subjects)
```

- [ ] **Step 8: Make the Bayesian sampling pool baseline-only**

In `scripts/bayesian_optimizer.py`, change line 26 to:

```python
from scripts.utils.discovery import baseline_scans, find_subject_files
```

In `_get_all_subjects`, replace the last line `return all_files` with:

```python
        return baseline_scans(all_files)
```

and append this sentence to its docstring: `Returns one baseline scan per subject (see discovery.baseline_scans).`

- [ ] **Step 9: Run the task's tests and the full suite**

Run: `python -m pytest tests/test_discovery.py tests/test_wave_staging.py -q`
Expected: all pass.

Run: `python -m pytest tests -q`
Expected: all pass except `tests/test_sessions_flag_warnings.py`, which may fail where it passes `--sessions-per-subject` into staging; Task 2 deletes it. Any other failure is a defect to fix here.

- [ ] **Step 10: Commit**

```bash
git add scripts/utils/discovery.py scripts/cross_validation_bootstrap_optimizer.py scripts/bayesian_optimizer.py tests/test_discovery.py tests/test_wave_staging.py
git commit -m "feat: stage one baseline scan per subject in grid and Bayesian sampling

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

(`git mv` already staged the deletion of `tests/test_session_selection.py`.)

---

### Task 2: Remove session options; fail loudly on old configs; `--subjects` default 10

**Files:**
- Modify: `scripts/cross_validation_bootstrap_optimizer.py` (generators `:71-150`, `:152-195`; `run_wave_pipeline` start `:455-470`; argparse `:1174-1190`; wave-config resolution `:1346-1402`)
- Modify: `scripts/opticonn_hub.py:155-178` (arguments) and `:911-914` (forwarding)
- Modify: `scripts/json_validator.py:174-177`
- Delete: `tests/test_sessions_flag_warnings.py`
- Modify: `tests/test_wave_staging.py` (replace the generator test)
- Modify: `tests/test_opticonn_hub_backend.py:298-307`

**Interfaces:**
- Consumes: Task 1's staging code.
- Produces: `generate_wave_configs(data_dir, output_dir, n_subjects: int = 10, extraction_cfg: str | None = None)` and `generate_single_wave_config(data_dir, output_dir, n_subjects: int = 10, extraction_cfg: str | None = None)`, with no `sessions_per_subject` parameter and no such key in their output.

- [ ] **Step 1: Write the failing tests**

In `tests/test_wave_staging.py`, add `import pytest` below `import json`, and replace `test_config_generators_write_sessions_per_subject` with:

```python
def test_config_generators_write_no_session_key_and_default_to_ten_subjects(tmp_path):
    def sel(path):
        return json.loads(Path(path).read_text())["data_selection"]

    w1, w2 = generate_wave_configs("d", tmp_path / "a")
    single = generate_single_wave_config("d", tmp_path / "b")
    for path in (w1, w2, single):
        assert "sessions_per_subject" not in sel(path)
        assert sel(path)["n_subjects"] == 10


def test_old_config_with_sessions_per_subject_fails_before_staging(tmp_path):
    with pytest.raises(ValueError, match="sessions_per_subject was removed"):
        _run(tmp_path, sessions_per_subject=2)
    assert not (tmp_path / "out" / "w" / "selected_files.txt").exists()
```

In `tests/test_opticonn_hub_backend.py`, replace the three tests from `test_tune_grid_forwards_nothing_when_sessions_per_subject_not_given` to the end of `test_tune_grid_forwards_zero_opt_out` with:

```python
def test_tune_grid_rejects_removed_sessions_per_subject_flag(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    proc = _run(
        ["--dry-run", "tune-grid", "--no-validation", "-i", str(data_dir),
         "-o", str(tmp_path / "out"), "--sessions-per-subject", "2"]
    )
    assert proc.returncode != 0
    assert "unrecognized arguments" in proc.stderr


def test_tune_grid_forwards_ten_subjects_by_default(tmp_path) -> None:
    assert "--subjects 10" in _optimizer_cmd(tmp_path, [])


def test_tune_grid_quick_defaults_to_three_subjects(tmp_path) -> None:
    assert "--subjects 3" in _optimizer_cmd(tmp_path, ["--quick"])


def test_tune_grid_forwards_explicit_subjects(tmp_path) -> None:
    assert "--subjects 4" in _optimizer_cmd(tmp_path, ["--quick", "--subjects", "4"])
```

Delete the old warning tests:

```bash
git rm tests/test_sessions_flag_warnings.py
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_wave_staging.py tests/test_opticonn_hub_backend.py -q`
Expected: the generator test fails (key present, `n_subjects` 3), the removed-key test fails (no `ValueError`), the flag-rejection test fails (flag accepted), and the default-subjects tests fail (3 forwarded).

- [ ] **Step 3: Add the removed-key error to `run_wave_pipeline`**

In `scripts/cross_validation_bootstrap_optimizer.py`, directly after `wave_name = wave_config["test_config"]["name"]` in `run_wave_pipeline`, insert:

```python
    if "sessions_per_subject" in wave_config.get("data_selection", {}):
        raise ValueError(
            "data_selection.sessions_per_subject was removed: OptiConn stages one scan per "
            f"subject (the first session). Remove the key from {wave_config_file}."
        )
```

- [ ] **Step 4: Remove `sessions_per_subject` from the config generators**

In `generate_wave_configs`, change the signature to:

```python
def generate_wave_configs(
    data_dir, output_dir, n_subjects: int = 10,
    extraction_cfg: str | None = None,
):
```

In `generate_single_wave_config`, change it to:

```python
def generate_single_wave_config(
    data_dir, output_dir, n_subjects: int = 10,
    extraction_cfg: str | None = None,
):
```

Delete all three lines `            "sessions_per_subject": int(sessions_per_subject),` (two in `generate_wave_configs`, one in `generate_single_wave_config`), and remove any `sessions_per_subject` entry from those functions' docstring parameter lists.

- [ ] **Step 5: Remove the optimizer flag and its handling; set `--subjects` default 10**

Replace:

```python
    parser.add_argument(
        "--subjects", type=int, default=3, help="Subjects per wave (default: 3)"
    )
```

with:

```python
    parser.add_argument(
        "--subjects", type=int, default=10,
        help="Subjects per wave, one baseline scan each (default: 10)",
    )
```

Delete the whole `parser.add_argument("--sessions-per-subject", ...)` call that follows it.

In the wave-config resolution in `main()`, delete these three lines:

```python
    sessions_flag_given = args.sessions_per_subject is not None
    if args.sessions_per_subject is None:
        args.sessions_per_subject = 2
```

Delete the `if sessions_flag_given:` block (warning plus condition) inside `if args.wave1_config and args.wave2_config:`. Delete the `if sessions_flag_given and wave1_config and wave2_config:` block in the `elif args.config:` branch. In all three `generate_wave_configs(...)` / `generate_single_wave_config(...)` calls in `main()`, delete the argument line `sessions_per_subject=args.sessions_per_subject,`.

Check that nothing is left: `grep -n "sessions_per_subject\|sessions-per-subject" scripts/cross_validation_bootstrap_optimizer.py` should print only the `run_wave_pipeline` error from Step 3.

- [ ] **Step 6: Remove the hub flag; resolve the `--subjects` default**

In `scripts/opticonn_hub.py`, replace the `--subjects` argument of `p_tune_grid` with:

```python
    p_tune_grid.add_argument(
        "--subjects",
        type=int,
        default=None,
        help="Subjects per wave, one baseline scan each (default: 10; 3 with --quick).",
    )
```

and delete the whole `p_tune_grid.add_argument("--sessions-per-subject", ...)` call.

Replace the forwarding lines:

```python
        if args.subjects:
            cmd += ["--subjects", str(int(args.subjects))]
        if getattr(args, "sessions_per_subject", None) is not None:
            cmd += ["--sessions-per-subject", str(int(args.sessions_per_subject))]
```

with:

```python
        subjects = args.subjects if args.subjects is not None else (3 if args.quick else 10)
        cmd += ["--subjects", str(int(subjects))]
```

- [ ] **Step 7: Remove the validator rule**

In `scripts/json_validator.py`, delete this block:

```python
            if "sessions_per_subject" in data_sel:
                sps = data_sel["sessions_per_subject"]
                if isinstance(sps, bool) or not isinstance(sps, int) or sps < 0:
                    errors.append("data_selection.sessions_per_subject must be an integer >= 0")
```

- [ ] **Step 8: Run the task's tests and the full suite**

Run: `python -m pytest tests/test_wave_staging.py tests/test_opticonn_hub_backend.py -q`
Expected: all pass.

Run: `python -m pytest tests -q`
Expected: all pass.

Run: `grep -rn "sessions_per_subject\|sessions-per-subject" scripts opticonn.py tests`
Expected: only the error in `run_wave_pipeline` and the two tests that assert on it and on the rejected flag.

- [ ] **Step 9: Commit**

```bash
git add scripts/cross_validation_bootstrap_optimizer.py scripts/opticonn_hub.py scripts/json_validator.py tests/test_wave_staging.py tests/test_opticonn_hub_backend.py
git commit -m "feat!: remove sessions_per_subject; old configs fail loudly; --subjects defaults to 10

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

(`git rm` already staged the deletion of `tests/test_sessions_flag_warnings.py`.)

---

### Task 3: Variance decomposition without `between_session`

**Files:**
- Modify: `scripts/variance_decomposition.py` (docstring `:1-12`, `compute_strata` `:57-158`, `_RATIO_PAIRS` `:180-186`, `headline_text` `:207-222`, `_STRATUM_ORDER` `:229`)
- Modify: `tests/test_variance_decomposition.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `compute_strata(...)` returns keys `tracking_noise`, `parameter`, `between_subject` only. `compute_ratios(...)` returns `tracking_noise_over_parameter`, `tracking_noise_over_between_subject`, `parameter_over_between_subject`. `headline_text(atlas, metric, ratios)` keys on `parameter_over_between_subject`. `collect_sweep_matrices` is unchanged; Task 5 imports it.

- [ ] **Step 1: Update the tests**

In `tests/test_variance_decomposition.py`:

1. Delete `test_compute_strata_between_session_only_pairs_same_subject` and `test_compute_strata_between_session_unavailable_for_single_session_cohort` entirely.
2. Replace `test_compute_ratios_divides_means`, `test_compute_ratios_none_when_a_stratum_unavailable`, `test_headline_text_reports_ratio_and_noise_fraction` and `test_headline_text_handles_missing_ratio` with:

```python
def test_compute_ratios_divides_means():
    summaries = {
        "parameter": {"available": True, "mean": 0.2, "n": 20},
        "tracking_noise": {"available": True, "mean": 0.02, "n": 20},
        "between_subject": {"available": True, "mean": 0.4, "n": 20},
    }

    ratios = compute_ratios(summaries)

    assert ratios["parameter_over_between_subject"] == pytest.approx(0.5)
    assert ratios["tracking_noise_over_parameter"] == pytest.approx(0.1)
    assert ratios["tracking_noise_over_between_subject"] == pytest.approx(0.05)
    assert set(ratios) == {
        "parameter_over_between_subject",
        "tracking_noise_over_parameter",
        "tracking_noise_over_between_subject",
    }


def test_compute_ratios_none_when_a_stratum_unavailable():
    summaries = {
        "parameter": {"available": False, "mean": None, "n": 0},
        "tracking_noise": {"available": True, "mean": 0.02, "n": 20},
        "between_subject": {"available": True, "mean": 0.4, "n": 20},
    }

    ratios = compute_ratios(summaries)

    assert ratios["parameter_over_between_subject"] is None
    assert ratios["tracking_noise_over_between_subject"] is not None


def test_headline_text_reports_ratio_and_noise_fraction():
    ratios = {"parameter_over_between_subject": 0.43, "tracking_noise_over_parameter": 0.025}

    text = headline_text("AAL3", "count", ratios)

    assert "AAL3/count" in text
    assert "0.43x" in text
    assert "two subjects" in text
    assert "2.5%" in text


def test_headline_text_handles_missing_ratio():
    text = headline_text("AAL3", "count", {"parameter_over_between_subject": None})

    assert "not available" in text.lower()
```

3. In `test_write_decomposition_creates_csv_with_header_once`, delete the `"between_session": ...` line from `summaries`, change `assert len(rows) == 8  # 4 strata x 2 metrics` to `assert len(rows) == 6  # 3 strata x 2 metrics`, and replace the stratum-set assertion with:

```python
    assert {r["stratum"] for r in rows[:3]} == {"tracking_noise", "parameter", "between_subject"}
```

4. In `test_compute_strata_warns_and_excludes_unparseable_keys`, delete the line `assert len(strata["between_session"]["dissimilarities"]) == 1`, and add after it:

```python
    assert set(strata) == {"tracking_noise", "parameter", "between_subject"}
```

5. In `test_run_twice_does_not_duplicate_output`, change `== 4` to `== 3` in the `csv.DictReader` row-count assertion.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_variance_decomposition.py -q`
Expected: the ratio, headline, write and unparseable-key tests fail (the old keys are still present).

- [ ] **Step 3: Remove `between_session` from the module**

In `scripts/variance_decomposition.py`:

1. Change the first docstring paragraph to:

```python
"""Variance decomposition across a completed OptiConn sweep.

Reports how much the connectome moves under three sources of variation --
tracking noise, parameter choice and between-subject differences -- as
comparable dissimilarity distributions, so a user can judge whether their
parameter choice matters relative to real individual differences. OptiConn is
cross-sectional: repeat sessions are never used (they usually carry the effect a
study measures).
```

Keep the rest of the docstring as is.

2. In `compute_strata`, delete `session_of: dict[str, str | None] = {}`. Replace

```python
            if key not in subject_of:
                subject, session = parse_subject_session(key)
                subject_of[key] = subject
                session_of[key] = session
```

with:

```python
            if key not in subject_of:
                subject_of[key] = parse_subject_session(key)[0]
```

Change the warning text `"excluded from the between_session and between_subject strata"` to `"excluded from the between_subject stratum"`. Delete `between_session: list[float] = []`, and delete the whole `subject_to_keys` block inside the `for combo_id in combo_ids:` loop: from `subject_to_keys: dict[str, list[str]] = {}` through `between_session.append(_distance(a, b))`. Keep `keys = [...]` and the `between_subject` loop. In the returned dict, delete the `"between_session": _entry(...)` entry.

3. Replace `_RATIO_PAIRS` with:

```python
_RATIO_PAIRS = [
    ("parameter_over_between_subject", "parameter", "between_subject"),
    ("tracking_noise_over_parameter", "tracking_noise", "parameter"),
    ("tracking_noise_over_between_subject", "tracking_noise", "between_subject"),
]
```

4. Replace `headline_text` with:

```python
def headline_text(atlas: str, metric: str, ratios: dict) -> str:
    parameter_ratio = ratios.get("parameter_over_between_subject")
    if parameter_ratio is None:
        return (
            f"{atlas}/{metric}: parameter-vs-subject ratio not available "
            "(needs >=2 candidates and >=2 subjects)"
        )
    noise_note = ""
    noise_ratio = ratios.get("tracking_noise_over_parameter")
    if noise_ratio is not None:
        noise_note = f" (tracking noise is {noise_ratio * 100:.1f}% of the parameter effect)"
    return (
        f"{atlas}/{metric}: parameter choice moves the connectome "
        f"{parameter_ratio:.2f}x as far as the difference between two subjects{noise_note}"
    )
```

5. Replace `_STRATUM_ORDER` with:

```python
_STRATUM_ORDER = ["tracking_noise", "parameter", "between_subject"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_variance_decomposition.py -q`
Expected: all pass.

Run: `grep -n "between_session\|session" scripts/variance_decomposition.py`
Expected: only the docstring sentence about sessions.

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest tests -q`
Expected: all pass.

```bash
git add scripts/variance_decomposition.py tests/test_variance_decomposition.py
git commit -m "feat: variance decomposition reports tracking noise, parameter and between-subject only

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Matrix-level network measures with modularity

**Files:**
- Modify: `scripts/compute_network_measures_from_connectivity.py:208-271` (`compute_measures`)
- Test: `tests/test_compute_network_measures_from_connectivity.py`

**Interfaces:**
- Consumes: existing private helpers `_density`, `_binary_graph`, `_weighted_graph`, `_global_efficiency_weighted_from_strength`, `_global_efficiency_weighted_from_distance`, `_invert_to_strength_from_distance`, `_small_world_sigma`, `_small_worldness_fast` in the same module.
- Produces: `measures_from_matrix(mat: np.ndarray, compute_smallworld: bool = False, smallworld_nrand: int = 10, seed: int = 42, weight_type: Literal["strength", "distance"] = "strength") -> Dict[str, float]`. It returns the keys `density`, `global_efficiency(binary)`, `clustering_coeff_average(binary)`, `small_worldness(binary)`, `clustering_coeff_average(weighted)`, `global_efficiency(weighted)` and `modularity`, plus `small_worldness(binary,sigma)` only when `compute_smallworld=True`. `compute_measures(connectivity_csv, compute_smallworld, smallworld_nrand, seed, weight_type="strength")` keeps its signature and delegates to it.

- [ ] **Step 1: Write the failing tests**

In `tests/test_compute_network_measures_from_connectivity.py`, extend the import to:

```python
from scripts.compute_network_measures_from_connectivity import (
    compute_measures,
    measures_from_matrix,
    write_network_measures_csv,
)
```

and append:

```python
def _two_communities(n=12):
    """Two dense blocks joined by one weak edge: modularity must be clearly positive."""
    m = np.zeros((n, n))
    half = n // 2
    m[:half, :half] = 5.0
    m[half:, half:] = 5.0
    m[0, half] = m[half, 0] = 0.1
    np.fill_diagonal(m, 0.0)
    return m


def test_measures_from_matrix_matches_compute_measures_via_csv(tmp_path):
    rng = np.random.default_rng(1)
    upper = np.triu(rng.random((10, 10)) * (rng.random((10, 10)) < 0.5), 1)
    matrix = upper + upper.T
    csv_path = tmp_path / "sub-1_atlas.count.connectivity.csv"
    _write_connectivity_csv(csv_path, matrix)

    from_csv = compute_measures(csv_path, compute_smallworld=False, smallworld_nrand=5, seed=0)
    direct = measures_from_matrix(matrix, seed=0)

    assert from_csv.keys() == direct.keys()
    for key in direct:
        assert direct[key] == pytest.approx(from_csv[key], nan_ok=True), key


def test_measures_from_matrix_reports_the_default_measure_set():
    assert set(measures_from_matrix(_two_communities())) == {
        "density",
        "global_efficiency(binary)",
        "clustering_coeff_average(binary)",
        "small_worldness(binary)",
        "clustering_coeff_average(weighted)",
        "global_efficiency(weighted)",
        "modularity",
    }


def test_modularity_is_positive_for_two_communities_and_seed_reproducible():
    first = measures_from_matrix(_two_communities(), seed=7)["modularity"]
    second = measures_from_matrix(_two_communities(), seed=7)["modularity"]
    assert first > 0.3
    assert first == second


def test_modularity_is_nan_not_an_error_for_an_empty_graph():
    assert math.isnan(measures_from_matrix(np.zeros((5, 5)))["modularity"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_compute_network_measures_from_connectivity.py -q`
Expected: `ImportError: cannot import name 'measures_from_matrix'`.

- [ ] **Step 3: Split `compute_measures` and add modularity**

In `scripts/compute_network_measures_from_connectivity.py`, rename the existing function to `measures_from_matrix`, change its signature, replace its first body line (`mat, _labels = _read_connectivity_csv(connectivity_csv)`) with the sanitisation below, and append the modularity block before `return measures`. The result:

```python
def measures_from_matrix(
    mat: np.ndarray,
    compute_smallworld: bool = False,
    smallworld_nrand: int = 10,
    seed: int = 42,
    weight_type: Literal["strength", "distance"] = "strength",
) -> Dict[str, float]:
    """Global graph measures for one connectivity matrix (non-finite -> 0, zero diagonal)."""
    mat = np.nan_to_num(np.asarray(mat, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Connectivity matrix must be square; got {mat.shape}")
    mat = mat.copy()
    np.fill_diagonal(mat, 0.0)

    measures: Dict[str, float] = {}
    # ... the rest of the former compute_measures body, unchanged, from
    # `measures["density"] = _density(mat)` through the weight_type if/elif/else ...

    # Louvain is stochastic; the fixed seed makes it reproducible, but part of
    # modularity's within-subject variance comes from the algorithm, not tractography.
    try:
        strength = mat if weight_type == "strength" else _invert_to_strength_from_distance(mat)
        Gmod = _weighted_graph(strength)
        communities = nx.community.louvain_communities(Gmod, weight="weight", seed=seed)
        measures["modularity"] = float(nx.community.modularity(Gmod, communities, weight="weight"))
    except Exception:
        measures["modularity"] = float("nan")

    return measures


def compute_measures(
    connectivity_csv: Path,
    compute_smallworld: bool,
    smallworld_nrand: int,
    seed: int,
    weight_type: Literal["strength", "distance"] = "strength",
) -> Dict[str, float]:
    mat, _labels = _read_connectivity_csv(connectivity_csv)
    return measures_from_matrix(mat, compute_smallworld, smallworld_nrand, seed, weight_type)
```

The `# ... the rest of the former compute_measures body ...` comment is not literal code: move the existing lines there verbatim and don't keep the comment. Nothing else in the body changes.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_compute_network_measures_from_connectivity.py -q`
Expected: all pass.

If `test_modularity_is_nan_not_an_error_for_an_empty_graph` fails because networkx returns a number for an edgeless graph, don't special-case it to fit. Add an explicit guard before Louvain, `if Gmod.number_of_edges() == 0: raise ValueError("no edges")`, so an edgeless graph maps to NaN like the other measures.

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest tests -q`
Expected: all pass. The two other callers (`scripts/aggregate_network_measures.py:176`, `scripts/mrtrix_tune.py:674`) now also receive `modularity`; they write whatever keys they get, so no change is needed there.

```bash
git add scripts/compute_network_measures_from_connectivity.py tests/test_compute_network_measures_from_connectivity.py
git commit -m "feat: measures_from_matrix for in-memory matrices; add seeded Louvain modularity

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Graph-measure ICC report

**Files:**
- Create: `scripts/graph_icc.py`
- Create: `tests/test_graph_icc.py`
- Modify: `scripts/cross_validation_bootstrap_optimizer.py` (sweep hook after the variance-decomposition `try/except`, around `:1468-1480`)

**Interfaces:**
- Consumes: `measures_from_matrix(mat, ...) -> dict[str, float]` (Task 4); `collect_sweep_matrices(sweep_optimize_dir) -> {(atlas, metric): {combo_id: {scan_key: [matrix, ...]}}}` from `scripts/variance_decomposition.py` (unchanged by Task 3).
- Produces: `icc_1_1(x) -> tuple[float, float, float]`, `compute_graph_icc(combo_matrices) -> list[dict]`, `run(sweep_optimize_dir: Path, output_dir: Path) -> list[dict]`, and the files `optimization_results/graph_icc.csv` and `graph_icc_summary.txt`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_graph_icc.py`:

```python
import ast
import csv
import math
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from scripts.graph_icc import MIN_SUBJECTS_FOR_ICC, compute_graph_icc, icc_1_1, run

N = 16
MEASURES = {
    "density",
    "global_efficiency(binary)",
    "clustering_coeff_average(binary)",
    "small_worldness(binary)",
    "clustering_coeff_average(weighted)",
    "global_efficiency(weighted)",
    "modularity",
}


def _subject_matrix(rng):
    mask = np.triu(rng.random((N, N)) < 0.4, 1)
    upper = np.where(mask, rng.random((N, N)) * 100, 0.0)
    return upper + upper.T


def _combo(n_subjects, seed=0, repeats=2):
    """{scan_key: [repeat matrices]}: each subject distinct, repeats add small noise on its edges."""
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n_subjects):
        base = _subject_matrix(rng)
        out[f"sub-{i:03d}_ses-1"] = [
            base * (1 + 0.01 * rng.standard_normal(base.shape)) * (base > 0) for _ in range(repeats)
        ]
    return out


def test_icc_matches_hand_computed_value():
    icc, low, high = icc_1_1(np.array([[1.0, 2.0], [3.0, 4.0], [6.0, 8.0]]))
    # MSB = 15.5, MSW = 1.0 -> ICC = 14.5 / 16.5
    assert icc == pytest.approx(29 / 33)
    assert low == pytest.approx(-0.017249, abs=1e-4)
    assert high == pytest.approx(0.996711, abs=1e-4)
    assert low <= icc <= high


def test_icc_near_one_when_repeats_agree_and_subjects_differ():
    subjects = np.arange(20, dtype=float)[:, None] * 10
    x = np.hstack([subjects, subjects + 0.01])
    assert icc_1_1(x)[0] > 0.99


def test_icc_near_zero_when_repeats_vary_as_much_as_subjects():
    x = np.random.default_rng(0).standard_normal((400, 2))
    assert abs(icc_1_1(x)[0]) < 0.15


def test_icc_is_nan_for_zero_variance():
    assert all(math.isnan(v) for v in icc_1_1(np.ones((5, 2))))


def test_compute_graph_icc_reports_every_measure_and_flags_low_confidence():
    rows = compute_graph_icc({"wave1/sweep_0001": _combo(5)})
    assert {r["measure"] for r in rows} == MEASURES
    assert all(r["combo_id"] == "wave1/sweep_0001" for r in rows)
    assert all(r["low_confidence"] for r in rows)  # 5 < MIN_SUBJECTS_FOR_ICC
    density = next(r for r in rows if r["measure"] == "density")
    assert density["n_subjects"] == 5 and density["icc"] is not None


def test_compute_graph_icc_not_low_confidence_at_threshold():
    rows = compute_graph_icc({"c": _combo(MIN_SUBJECTS_FOR_ICC)})
    assert not any(r["low_confidence"] for r in rows)


def test_compute_graph_icc_reports_no_number_below_three_subjects():
    rows = compute_graph_icc({"c": _combo(2)})
    for r in rows:
        assert r["icc"] is None and r["ci_low"] is None and r["ci_high"] is None
        assert r["reason"].startswith("fewer than 3 subjects (2)")


def test_compute_graph_icc_drops_subjects_with_fewer_than_two_repeats():
    combo = _combo(4)
    combo["sub-099_ses-1"] = combo["sub-000_ses-1"][:1]
    rows = compute_graph_icc({"c": combo})
    density = next(r for r in rows if r["measure"] == "density")
    assert density["n_subjects"] == 4
    assert "1 subject(s) with <2 repeats dropped" in density["reason"]


def _write_combo(combo_dir, atlas, metric, reps_by_key):
    for key, reps in reps_by_key.items():
        for rep_idx, matrix in enumerate(reps, start=1):
            d = combo_dir / f"rep_{rep_idx}" / "results" / atlas
            d.mkdir(parents=True, exist_ok=True)
            scipy.io.savemat(
                str(d / f"{key}_{atlas}.tt.gz.{metric}..pass.connectivity.mat"),
                {"connectivity": matrix},
            )


def test_run_writes_csv_and_summary_without_duplicates(tmp_path):
    optimize = tmp_path / "optimize"
    for i, combo_id in enumerate(("sweep_0001", "sweep_0002")):
        _write_combo(optimize / "wave1" / "combos" / combo_id, "AAL3", "count", _combo(4, seed=i))
    out = tmp_path / "optimization_results"

    run(optimize, out)
    run(optimize, out)

    rows = list(csv.DictReader((out / "graph_icc.csv").read_text().splitlines()))
    assert len(rows) == 2 * len(MEASURES)
    assert {r["atlas"] for r in rows} == {"AAL3"} and {r["metric"] for r in rows} == {"count"}
    summary = (out / "graph_icc_summary.txt").read_text()
    assert summary.count("=== AAL3 / count ===") == 1
    assert "Louvain" in summary


def test_run_warns_and_writes_nothing_without_matrices(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        assert run(tmp_path / "missing", tmp_path / "out") == []
    assert not (tmp_path / "out" / "graph_icc.csv").exists()


def test_sweep_hook_runs_graph_icc_in_its_own_try_in_the_two_wave_branch():
    """Both reports run on the two-wave path, each isolated so one failing never
    suppresses the other, and graph ICC never runs on the single-wave path."""
    source = Path(__file__).resolve().parents[1] / "scripts" / "cross_validation_bootstrap_optimizer.py"
    tree = ast.parse(source.read_text())

    def imports(node, module):
        return any(isinstance(c, ast.ImportFrom) and c.module == module for c in ast.walk(node))

    single_wave_if = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Attribute)
        and n.test.attr == "single_wave"
        and any(
            isinstance(c, ast.Constant) and isinstance(c.value, str) and "CROSS-VALIDATION COMPLETED" in c.value
            for c in ast.walk(n)
        )
    )
    assert not any(imports(s, "scripts.graph_icc") for s in single_wave_if.body)
    tries = [n for s in single_wave_if.orelse for n in ast.walk(s) if isinstance(n, ast.Try)]
    icc_tries = [t for t in tries if any(imports(s, "scripts.graph_icc") for s in t.body)]
    vd_tries = [t for t in tries if any(imports(s, "scripts.variance_decomposition") for s in t.body)]
    assert len(icc_tries) == 1 and len(vd_tries) == 1
    assert icc_tries[0] is not vd_tries[0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_graph_icc.py -q`
Expected: `ModuleNotFoundError: No module named 'scripts.graph_icc'`.

- [ ] **Step 3: Implement `scripts/graph_icc.py`**

```python
"""Reliability of graph measures across a completed OptiConn sweep.

For each candidate parameter set and each global graph measure, a one-way random
ICC(1,1): variance between subjects against variance between tracking repeats
of the same scan. Discriminability saturates at 1.0 for plausible candidates,
but a scalar graph measure compresses the connectome, so tracking noise is no
longer negligible and candidates can separate. Measures may disagree on which
candidate is more reliable, so this is reported per measure and never ranked:
nothing here is consumed by scripts.reliability.rank() or rank_with_fallback().
See docs/superpowers/specs/2026-09-21-cross-sectional-graph-icc-design.md.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.stats import f as f_dist

from scripts.compute_network_measures_from_connectivity import measures_from_matrix
from scripts.variance_decomposition import collect_sweep_matrices

MIN_SUBJECTS_FOR_ICC = 10
MIN_SUBJECTS_TO_REPORT = 3

_COLUMNS = [
    "atlas", "metric", "combo_id", "measure", "n_subjects",
    "icc", "ci_low", "ci_high", "low_confidence", "reason",
]
_OUTPUTS = ("graph_icc.csv", "graph_icc_summary.txt")


def icc_1_1(x: np.ndarray) -> tuple[float, float, float]:
    """One-way random ICC(1,1) and its 95% CI (F distribution). x: subjects x repeats.

    NaN triple when there is no variance at all; 1.0 when repeats agree exactly.
    """
    x = np.asarray(x, dtype=float)
    n, k = x.shape
    row_means = x.mean(axis=1)
    msb = k * ((row_means - x.mean()) ** 2).sum() / (n - 1)
    msw = ((x - row_means[:, None]) ** 2).sum() / (n * (k - 1))
    if msb + (k - 1) * msw == 0:
        return float("nan"), float("nan"), float("nan")
    if msw == 0:
        return 1.0, 1.0, 1.0
    icc = (msb - msw) / (msb + (k - 1) * msw)
    F = msb / msw
    f_low = F / f_dist.ppf(0.975, n - 1, n * (k - 1))
    f_high = F * f_dist.ppf(0.975, n * (k - 1), n - 1)
    return float(icc), float((f_low - 1) / (f_low + k - 1)), float((f_high - 1) / (f_high + k - 1))


def compute_graph_icc(combo_matrices: dict[str, dict[str, list[np.ndarray]]]) -> list[dict]:
    """One row per (combo_id, measure) for one (atlas, metric).

    combo_matrices: {combo_id: {scan_key: [matrix per repeat]}}, as one value of
    collect_sweep_matrices(). Repeats are truncated to the shortest subject so the
    ICC design is balanced.
    """
    rows: list[dict] = []
    for combo_id in sorted(combo_matrices):
        scans = combo_matrices[combo_id]
        usable = {key: reps for key, reps in scans.items() if len(reps) >= 2}
        too_few_repeats = len(scans) - len(usable)
        if not usable:
            logging.warning("graph ICC: %s has no subject with >=2 repeats; skipped", combo_id)
            continue
        k = min(len(reps) for reps in usable.values())
        # ponytail: sequential per-matrix measures (~0.4 s on AAL3, ~14 s on a dense
        # 400-node graph); parallelise with multiprocessing if report time matters.
        per_scan = {key: [measures_from_matrix(m) for m in reps[:k]] for key, reps in usable.items()}
        names = sorted({name for reps in per_scan.values() for r in reps for name in r})
        for name in names:
            series = [[r.get(name, float("nan")) for r in reps] for reps in per_scan.values()]
            clean = [s for s in series if np.all(np.isfinite(s))]
            notes = []
            if too_few_repeats:
                notes.append(f"{too_few_repeats} subject(s) with <2 repeats dropped")
            if len(series) - len(clean):
                notes.append(f"{len(series) - len(clean)} subject(s) with a non-finite value dropped")
            row = {
                "combo_id": combo_id,
                "measure": name,
                "n_subjects": len(clean),
                "icc": None,
                "ci_low": None,
                "ci_high": None,
                "low_confidence": len(clean) < MIN_SUBJECTS_FOR_ICC,
            }
            if len(clean) < MIN_SUBJECTS_TO_REPORT:
                notes.insert(0, f"fewer than {MIN_SUBJECTS_TO_REPORT} subjects ({len(clean)})")
            else:
                icc, low, high = icc_1_1(np.array(clean))
                if np.isnan(icc):
                    notes.insert(0, "no variance across subjects or repeats")
                else:
                    row.update(icc=icc, ci_low=low, ci_high=high)
            row["reason"] = "; ".join(notes)
            rows.append(row)
    return rows


def _summary_lines(atlas: str, metric: str, rows: list[dict]) -> list[str]:
    lines = [f"\n=== {atlas} / {metric} ==="]
    for name in sorted({r["measure"] for r in rows}):
        measure_rows = [r for r in rows if r["measure"] == name]
        scored = [r for r in measure_rows if r["icc"] is not None]
        if not scored:
            lines.append(f"  {name}: not available ({measure_rows[0]['reason'] or 'no candidate scored'})")
            continue
        best = max(scored, key=lambda r: r["icc"])
        flag = " [low confidence]" if best["low_confidence"] else ""
        lines.append(
            f"  {name}: most reliable {best['combo_id']} ICC={best['icc']:.2f} "
            f"[{best['ci_low']:.2f}, {best['ci_high']:.2f}] n={best['n_subjects']}{flag}"
        )
    return lines


def run(sweep_optimize_dir: Path, output_dir: Path) -> list[dict]:
    output_dir = Path(output_dir)
    for name in _OUTPUTS:
        (output_dir / name).unlink(missing_ok=True)
    grouped = collect_sweep_matrices(Path(sweep_optimize_dir))
    if not grouped:
        logging.warning(
            "graph ICC: no */combos/sweep_* directories with connectivity matrices under %s",
            sweep_optimize_dir,
        )
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    summary = [
        "Graph-measure reliability: one-way ICC(1,1), subjects vs tracking repeats, 95% CI.",
        "Reported per measure, never ranked. Modularity uses seeded Louvain, so part of its",
        "within-subject variance comes from the algorithm, not from tractography.",
    ]
    with (output_dir / "graph_icc.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        for atlas, metric in sorted(grouped):
            rows = [{"atlas": atlas, "metric": metric, **r} for r in compute_graph_icc(grouped[(atlas, metric)])]
            writer.writerows(rows)
            all_rows.extend(rows)
            if rows:
                summary.extend(_summary_lines(atlas, metric, rows))
    (output_dir / "graph_icc_summary.txt").write_text("\n".join(summary) + "\n")
    logging.info("Graph ICC report written to %s", output_dir / "graph_icc.csv")
    return all_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Graph-measure ICC across a completed OptiConn sweep")
    parser.add_argument("sweep_optimize_dir", help="Path to a sweep's optimize/ directory")
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Output directory (default: <sweep_optimize_dir>/optimization_results)",
    )
    args = parser.parse_args()
    sweep_dir = Path(args.sweep_optimize_dir)
    output_dir = Path(args.output_dir) if args.output_dir else sweep_dir / "optimization_results"
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(sweep_dir, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Add the sweep hook**

In `scripts/cross_validation_bootstrap_optimizer.py`, directly after the existing block

```python
            except Exception as exc:
                logging.warning("  Variance decomposition skipped: %s", exc)
```

add, at the same indentation as that `try:`:

```python
            try:
                from scripts.graph_icc import run as run_graph_icc

                run_graph_icc(Path(output_dir), Path(output_dir) / "optimization_results")
            except Exception as exc:
                logging.warning("  Graph ICC report skipped: %s", exc)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_graph_icc.py tests/test_variance_decomposition.py -q`
Expected: all pass, including the existing variance-decomposition AST guard tests.

- [ ] **Step 6: Run the standalone CLI on the real study-129 sweep (read-only check)**

```bash
python -m scripts.graph_icc studies/study129/aal3_search/sweep-dd7bea33-934c-4aca-9c0e-c11ce051c2ea/optimize \
  -o "$TMPDIR/graph_icc_check"
```

(use any scratch directory for `-o`; never write into `studies/`.)
Expected: `graph_icc.csv` with rows for AAL3 count/fa/qa. The count candidate `bootstrap_qa_wave_1/sweep_0001` should show `global_efficiency(binary)` ≈ 0.62, n = 6, low_confidence True, matching the spec's probe table within rounding. If it differs by more than 0.02, stop and report: the probe and the shipped code disagree.

- [ ] **Step 7: Full suite and commit**

Run: `python -m pytest tests -q`
Expected: all pass.

```bash
git add scripts/graph_icc.py tests/test_graph_icc.py scripts/cross_validation_bootstrap_optimizer.py
git commit -m "feat: graph-measure ICC report per candidate (reported, never ranked)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Documentation, paper and specs

**Files:**
- Modify: `docs/methods.md`, `docs/user_guide.md`, `docs/cli_steps.md:143-146`, `README.md:196-199` and `:470-473`
- Modify: `paper.md`, `paper.bib`
- Modify: `docs/superpowers/specs/2026-09-18-discriminability-noise-floor-design.md`, `docs/superpowers/specs/2026-09-21-cross-sectional-graph-icc-design.md` (status line)

**Interfaces:** none (documentation only). Every flag, file name and default mentioned must match Tasks 1–5.

In this task's find/replace instructions, `\`` inside quoted text stands for a literal backtick. Write plain backticks in the files.

- [ ] **Step 1: `docs/methods.md`**

1. In the saturation table, delete the row `| Same subject, different session, same parameters | 0.889 (median 0.900) | 0.111 |`.
2. Replace the sentence starting `The noise floor is roughly 2.5% of the between-session effect and 2% of the parameter effect.` with `The noise floor is roughly 2% of the parameter effect and 1% of the between-subject difference.` Keep the rest of that paragraph.
3. In "How to read this as a user", replace `the same table shows a modest parameter change moving the connectome slightly *more* than a real between-session change does` with `the same table shows a modest parameter change moving the connectome about 0.43x as far as the difference between two people`.
4. Replace the whole `### Session-aware wave staging` section (heading plus its two paragraphs) with:

```markdown
### Cross-sectional by design

OptiConn uses one scan per subject: the first session in natural order (`ses-2` before `ses-10`), for both `tune-grid` staging and `tune-bayes` sampling. Datasets with repeat DWI almost always acquired it to measure change, often an intervention effect, so a later session is never used to choose parameters, neither as a repeat nor as a benchmark. When `exclude_scans` removes a subject's baseline scan, the subject drops out rather than falling back to a later session. `--subjects` counts subjects (default 10). Configs that still set `data_selection.sessions_per_subject` fail at load with an explanation.

### Graph-measure reliability (ICC)

Discriminability works on whole edge vectors and saturates. The graph measures a study analyses do not necessarily: after a two-wave sweep, `graph_icc.csv` reports, per candidate and per global measure (density, global efficiency and clustering, binary and weighted, small-worldness, Louvain modularity), a one-way ICC(1,1) of subjects against tracking repeats with a 95% confidence interval. On the study-129 AAL3 sweep, two candidates tied on discriminability (1.000), margin (0.163 vs 0.158) and repeatability (0.951 vs 0.950), yet their binary global-efficiency ICC was 0.62 vs 0.92 while binary clustering favoured the other candidate (0.84 vs 0.71). At n = 5–6 those intervals overlap, so this is suggestive only. Below 10 subjects every ICC is flagged as low confidence; below 3 none is reported.

ICC is reported, never ranked: measures can disagree on the better candidate, which measures matter is a study decision, and a setting that flattens individual differences can still score well on some measures. Modularity's ICC includes the Louvain algorithm's own variability.

Reliability is not validity. A setting can reproducibly produce false-positive connections, which distort graph measures more than missed connections do (Zalesky et al., 2016). Treat a high ICC as necessary, not sufficient.
```

- [ ] **Step 2: `docs/user_guide.md`**

1. Replace the example line `  --subjects 5 --sessions-per-subject 2 \` with `  --subjects 10 \`.
2. Replace the two bullets starting `- \`--subjects\` counts subjects, not scans.` and `- The \`--sessions-per-subject\` flag only applies` with:

```markdown
- `--subjects` counts subjects (default 10; 3 with `--quick`). OptiConn uses one scan per subject, the first session, because later sessions usually carry the effect you are studying. Fewer than 10 subjects gives noisy reliability estimates, and the reports flag it.
```

3. In section 4, after the `exclude_scans` JSON example, add: `If you exclude a subject's first-session scan, that subject is dropped; OptiConn does not fall back to a later session.`
4. In section 5, delete the table row for `between_session`. Replace the paragraph starting `The headline ratio is \`parameter / between_session\`.` with:

```markdown
The headline ratio is `parameter / between_subject`: how far the parameter choice moves a connectome, relative to the difference between two people. A value near 0.5 means switching between reasonable settings moves a connectome half as far as swapping in a different person, which is large for any group analysis. Strata with fewer than 10 pairs are flagged as low confidence. The decomposition is diagnostic only; it never influences ranking, because minimising parameter sensitivity would reward settings that flatten real differences.

- `graph_icc.csv` and `graph_icc_summary.txt`: for each candidate and each graph measure, how reliably that measure ranks subjects despite tracking noise (ICC with a 95% confidence interval). Look at the measures you plan to analyse: candidates that tie on discriminability often differ here, and different measures can favour different candidates. ICC is reported, not ranked; see [Methods](methods.md) for why a high ICC is necessary but not sufficient.
```

5. In section 8, replace the bullet `- **\`between_session\` is "not available".** ...` with:

```markdown
- **"data_selection.sessions_per_subject was removed".** Delete that key from your wave configs; OptiConn now always uses one first-session scan per subject.
```

- [ ] **Step 3: `README.md` and `docs/cli_steps.md`**

In `README.md`, replace both occurrences (around lines 198–199 and 472–473) of the two lines `--subjects N ... (default: 3)` and `--sessions-per-subject N ...` with this single line (keep each block's existing wording style for `--subjects`):

```markdown
- `--subjects N`: Subjects per wave, one first-session scan each (default: 10; 3 with `--quick`).
```

In `docs/cli_steps.md`, replace lines 145–146 (`--subjects <int>` and `--sessions-per-subject <int>`) with:

```markdown
- `--subjects <int>`: subjects per wave, one first-session scan each (default: `10`; `3` with `--quick`)
```

Check: `grep -rn "sessions-per-subject\|sessions_per_subject\|between_session" README.md docs --include=*.md | grep -v docs/superpowers` prints nothing.

- [ ] **Step 4: `paper.bib` — add the Zalesky reference**

Append:

```bibtex
@article{Zalesky2016,
  title   = {Connectome sensitivity or specificity: which is more important?},
  author  = {Zalesky, Andrew and Fornito, Alex and Cocchi, Luca and Gollo, Leonardo L. and van den Heuvel, Martijn P. and Breakspear, Michael},
  journal = {NeuroImage},
  volume  = {142},
  pages   = {407--420},
  year    = {2016},
  doi     = {10.1016/j.neuroimage.2016.06.035}
}
```

Before committing, check the DOI resolves to this title (e.g. `curl -sI https://doi.org/10.1016/j.neuroimage.2016.06.035 | head -3`, expecting a redirect). If it doesn't match, stop and report instead of guessing.

- [ ] **Step 5: `paper.md`**

1. **Summary.** Replace `places the parameter effect on the same scale as tracking noise, between-session change and between-subject anatomy, so a user can see whether the choice rivals their effect of interest` with `places the parameter effect on the same scale as tracking noise and between-subject differences, and an ICC report of the graph measures a study will analyse, so a user can see how much the choice matters for their analysis`. Insert this sentence before `Candidates are proposed either by`: `*OptiConn* is deliberately cross-sectional: it uses one scan per subject, because repeat sessions in real datasets usually carry the very effect under study.`
2. **Statement of need, second paragraph.** Replace from `On a 150-subject longitudinal cohort` through `the longitudinal effect being reported.` with: `On a 150-subject cohort (AAL3 parcellation, edge-vector correlations), a modest parameter change — raising the FA threshold from 0 to 0.1 and the turning angle to 45° — moved a subject's connectome about 0.43 times as far as the difference between two people (dissimilarity 0.136 versus 0.316), while tracking stochasticity accounted for 0.003. And candidates that tie on every connectome-level criterion can still differ in how reliably they preserve the graph measures a study analyses.` These figures are replaced by the fresh study-129 sweep before submission (see the plan's "After implementation").
3. **State of the field, item (4).** Replace `a variance decomposition that reports the parameter effect on the same scale as the biological effects a study measures` with `a variance decomposition that reports the parameter effect on the scale of between-subject differences, together with the per-measure reliability of downstream graph measures`.
4. **Component 5.** Replace its text from `The strata are` through `never as a zero effect, otherwise.` with: `The strata are \`tracking_noise\` (same scan, same candidate, different random seed), \`parameter\` (same scan, different candidate) and \`between_subject\` (different subjects, same candidate). Each is reported as a distribution (n, mean, median, IQR) rather than a point estimate, and strata with fewer than ten pairs are flagged as low-confidence. The headline ratios are \`parameter / between_subject\` — how far the parameter choice moves a connectome relative to the difference between two people — and \`tracking_noise / parameter\` — how much of the apparent parameter effect is merely stochastic.` Keep the component's final sentences about not being wired into ranking.
5. **New component**, inserted right after component 5, renumbering the later ones and every in-text `(6)`/`(7)` cross-reference (`grep -n "([0-9])" paper.md`): `**Graph-measure reliability**: For each candidate and each global graph measure (density, global efficiency and clustering in binary and weighted form, small-worldness, Louvain modularity), a one-way ICC(1,1) of subjects against tracking repeats, with a 95% confidence interval from the F distribution. Discriminability saturates for plausible candidates because it compares whole edge vectors against a very small tracking-noise floor; a scalar graph measure compresses the connectome, so candidates that tie on discriminability can differ markedly in how reliably they preserve a given measure, and different measures can favour different candidates. ICC is therefore reported per measure and never ranked, and is flagged as low-confidence below ten subjects.`
6. **Cohort handling component.** Replace the sentence starting `Wave staging is session-aware:` with `Staging is cross-sectional: each subject contributes one scan, its first session, in both the sweep and the Bayesian proposer, and a subject whose first-session scan fails QC is dropped rather than replaced by a later session.`
7. **Limitations.** Add a paragraph at the end of "Design and implementation": `**Limitations.** Every criterion *OptiConn* reports measures reliability, and reliability is not validity: dense tractography can reproducibly generate false-positive connections, which distort graph measures more than missed connections do [@Zalesky2016]. The density gate is the only safeguard for specificity; a high ICC is necessary for a trustworthy graph analysis, not sufficient. Because repeats are tracking re-runs of a single scan, they carry algorithmic but not measurement noise.`
8. **Test count.** Replace `156 tests` with the count from `python -m pytest tests -q | tail -1`.

Check: `grep -n "between_session\|between-session\|longitudinal\|sessions-per-subject" paper.md` prints nothing.

- [ ] **Step 6: Spec status lines**

In `docs/superpowers/specs/2026-09-18-discriminability-noise-floor-design.md`, directly under the `Status:` line add: `Note (2026-09-21): OptiConn became cross-sectional only; the between_session stratum and session-aware staging described here were removed. See 2026-09-21-cross-sectional-graph-icc-design.md.`

In `docs/superpowers/specs/2026-09-21-cross-sectional-graph-icc-design.md`, change `Status: Approved design, spec awaiting review` to `Status: Implemented`.

- [ ] **Step 7: Verify and commit**

Run: `python -m pytest tests -q`
Expected: all pass (docs changes can't break tests, but confirm the tree is green).

```bash
git add docs/methods.md docs/user_guide.md docs/cli_steps.md README.md paper.md paper.bib docs/superpowers/specs/2026-09-18-discriminability-noise-floor-design.md docs/superpowers/specs/2026-09-21-cross-sectional-graph-icc-design.md
git commit -m "docs: cross-sectional scope, graph-measure ICC, reliability-vs-validity limitation

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## After implementation (not part of this plan)

1. One fresh cross-sectional `tune-grid` on study 129: AAL3, 10 subjects per wave, `exclude_scans` from `python -m scripts.qc_gate`.
2. Replace the study-129 figures in `paper.md`, `docs/methods.md` and `docs/user_guide.md` with that run's numbers only.
3. Remove the force-added `.pyc` files from history, push, submit to JOSS.
