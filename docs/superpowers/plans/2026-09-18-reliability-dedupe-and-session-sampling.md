# Reliability De-duplication and Session-Aware Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two problems found while finishing the variance-decomposition branch: (1) `reliability.collect_matrices` loads every repeat of a DSI Studio scan twice, corrupting `n_repeats`, repeatability and the decomposition's tracking-noise stratum; (2) wave staging samples random *scans*, so `between_session` (the decomposition's headline stratum) is almost never available.

**Architecture:** (1) `collect_matrices` keeps one matrix per (atlas, metric, scan) per repeat, with the combined `.mat` taking priority over the per-metric `.csv` that is just a conversion of it. (2) A pure, unit-testable `select_scans` in `scripts/utils/discovery.py` replaces the inline `random.sample` in wave staging; with `sessions_per_subject >= 2` it samples subjects (not scans) and stages several sessions of each, falling back to the old scan-level sampling when no subject has enough sessions. A CLI flag defaults it to 2, so multi-session cohorts work out of the box while single-session cohorts behave exactly as before.

**Tech Stack:** Python 3.10+, numpy, scipy, pandas, pytest. No new dependencies.

## Global Constraints

- Selection stays diagnostic-safe: nothing here may change how candidates are *ranked* other than by correcting the repeat count and repeatability values that were inflated by duplicate loading.
- Backward compatibility: `select_scans(pool, n, seed, sessions_per_subject=0 or 1)` must return exactly what the legacy code returned for the same inputs, i.e. what `random.seed(seed); random.sample(pool, n)` returns (or the whole pool when `n >= len(pool)`). Wave configs that lack the `sessions_per_subject` key must keep the legacy behaviour.
- `sessions_per_subject` semantics: the number of sessions staged for each sampled subject. `--subjects N` then means N *subjects*, so a session-aware wave stages up to `N * sessions_per_subject` scans (more tracking compute). This must be stated in the CLI help.
- Selection is deterministic for a given seed and cohort (sessions of a subject are taken by natural sort order of the session id: `ses-2` before `ses-10`).
- Use percent-style arguments for every new `logging` call (never f-strings), matching the rest of this branch.
- Work only in the worktree `/data/local/software/opticonn/.worktrees/feature/variance-decomposition`; activate the env with `source braingraph_pipeline/bin/activate`. Baseline: 119 tests pass.

---

## File Structure

- Modify: `scripts/reliability.py` (`collect_matrices`), `tests/test_reliability.py`
- Modify: `scripts/utils/discovery.py` (add `select_scans`), `tests/test_discovery.py`
- Modify: `scripts/cross_validation_bootstrap_optimizer.py` (staging call site, config generators, CLI flag), plus a dry-run integration test in `tests/test_cross_validation_repeats.py` or a new `tests/test_session_selection.py`
- Modify: `docs/methods.md`, `docs/superpowers/specs/2026-09-18-discriminability-noise-floor-design.md`, and whichever doc lists the sweep CLI flags

---

### Task 1: One matrix per (atlas, metric, scan) per repeat

**Files:**
- Modify: `scripts/reliability.py` (`collect_matrices`, currently the loop starting `for rep_dir in sorted(Path(combo_dir).glob("rep_*")):`)
- Test: `tests/test_reliability.py`

**Interfaces:**
- Consumes: existing helpers in `tests/test_reliability.py`: `_dataset(subject_specific, subjects, repeats, seed)`, `_write_combined_dsi_mat(root, rep, subject, atlas, count_matrix, fa_matrix)`, `_write_dsi_mat(root, rep, subject, atlas, metric, matrix)`, `_write_mrtrix_csv(root, rep, subject, atlas, metric, matrix)`.
- Produces: `collect_matrices(combo_dir)` keeps its signature and return shape `{(atlas, metric): {subject: [matrix per repeat]}}`, but the list now has exactly one entry per repeat.

**Background (verified on real data):** in a real DSI Studio sweep each repeat directory holds, for one scan, the combined `<atlas>.connectivity.mat` (which `_load_combined` expands into count/fa/qa) *and* per-metric CSV conversions such as `<scan>_AAL3.tt.gz.AAL3.count.connectivity.csv`. `collect_matrices` loads both, so `count` and `qa` get 4 matrices for 2 repeats (the `.mat` and `.csv` copies are identical, edge-vector distance exactly 0), `n_repeats` reads 4, repeatability is inflated (0.99727 reported vs 0.99590 true on study 129), and a scan with a single surviving repeat still has 2 entries so it slips past the "fewer than 2 repeats" gate. `fa` has no CSV so it is correct. The MRtrix3 backend writes CSV only and must be unaffected.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_reliability.py`)

```python
def test_collect_dedupes_combined_mat_and_per_metric_csv(tmp_path):
    for subject, reps in _dataset(True, subjects=3).items():
        for k, m in enumerate(reps, 1):
            _write_combined_dsi_mat(tmp_path, k, subject, "AAL3", m, m + 1.0)
            # DSI Studio also converts the count matrix to a per-metric CSV
            _write_mrtrix_csv(tmp_path, k, subject, "AAL3", "count", m)
    got = collect_matrices(tmp_path)
    assert len(got[("AAL3", "count")]["sub0"]) == 2  # was 4: .mat + .csv per repeat
    assert len(got[("AAL3", "fa")]["sub0"]) == 2
    [count_row] = [r for r in score_combo(tmp_path, {}) if r["connectivity_metric"] == "count"]
    assert count_row["n_repeats"] == 2


def test_collect_csv_still_supplies_metrics_the_mat_lacks(tmp_path):
    for subject, reps in _dataset(True, subjects=3).items():
        for k, m in enumerate(reps, 1):
            _write_combined_dsi_mat(tmp_path, k, subject, "AAL3", m, m + 1.0)
            _write_mrtrix_csv(tmp_path, k, subject, "AAL3", "ncount2", m)  # not a combined-mat metric
    got = collect_matrices(tmp_path)
    assert len(got[("AAL3", "ncount2")]["sub0"]) == 2
    assert len(got[("AAL3", "count")]["sub0"]) == 2


def test_single_surviving_repeat_is_not_padded_to_two_by_its_csv_copy(tmp_path):
    for subject, reps in _dataset(True, subjects=3).items():
        n_reps = 1 if subject == "sub0" else 2
        for k, m in enumerate(reps[:n_reps], 1):
            _write_combined_dsi_mat(tmp_path, k, subject, "AAL3", m, m + 1.0)
            _write_mrtrix_csv(tmp_path, k, subject, "AAL3", "count", m)
    [count_row] = [
        r for r in score_combo(tmp_path, {"density_range": [0.02, 1.0]})
        if r["connectivity_metric"] == "count"
    ]
    assert "fewer than 2 repeats" in count_row["rejected"]


def test_same_scan_written_twice_in_one_repeat_counts_once(tmp_path):
    rng_data = _dataset(True, subjects=3)
    for subject, reps in rng_data.items():
        for k, m in enumerate(reps, 1):
            _write_dsi_mat(tmp_path, k, subject, "AAL3", "count", m)
    # a second timestamped output dir for the same scan in repeat 1
    extra = tmp_path / "rep_1" / "01_connectivity" / "sub0.gqi_20250102" / "tracks_100k" / "results" / "AAL3"
    extra.mkdir(parents=True)
    scipy.io.savemat(
        str(extra / "sub0.gqi_AAL3.tt.gz.AAL3.count..pass.connectivity.mat"),
        {"connectivity": rng_data["sub0"][0]},
    )
    got = collect_matrices(tmp_path)
    assert len(got[("AAL3", "count")]["sub0"]) == 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /data/local/software/opticonn/.worktrees/feature/variance-decomposition && source braingraph_pipeline/bin/activate && python3 -m pytest tests/test_reliability.py -q -k "dedupes or supplies_metrics or single_surviving or written_twice"`
Expected: `test_collect_dedupes...`, `test_single_surviving...` and `test_same_scan_written_twice...` FAIL (counts of 4/3 instead of 2, and the gate not firing); `test_collect_csv_still_supplies_metrics_the_mat_lacks` may already pass for `ncount2` but must still pass after the change.

- [ ] **Step 3: Implement** — in `collect_matrices`, replace the per-rep loop body so it tracks what each repeat has already supplied and processes `.mat` before `.csv`:

```python
    found: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}
    for rep_dir in sorted(Path(combo_dir).glob("rep_*")):
        # One matrix per (atlas, metric, scan) per repeat. DSI Studio writes each
        # metric twice per scan -- inside the combined <atlas>.connectivity.mat and
        # again as a converted per-metric .connectivity.csv -- and loading both
        # made every repeat count twice (n_repeats 4 instead of 2, repeatability
        # inflated, and a scan with one surviving repeat still passing the >=2
        # repeats gate). The .mat is the source of truth so it wins; a .csv only
        # fills metrics no .mat supplied (the MRtrix3 backend writes CSV only).
        supplied: set[tuple[str, str, str]] = set()
        paths = sorted(rep_dir.rglob("*.connectivity.mat")) + sorted(rep_dir.rglob("*.connectivity.csv"))
        for path in paths:
            atlas = path.parent.name
            subject = path.name.split(f"_{atlas}.")[0].split(".")[0]
            if path.suffix == ".csv":
                match = _CSV_MATRIX_NAME.search(path.name)
                if not match:
                    continue
                metrics = {match.group(1): _load_csv_matrix(path)}
            else:
                match = _MATRIX_NAME.search(path.name)
                metrics = {match.group(1): load_matrix(path)} if match else _load_combined(path)
            for metric, matrix in metrics.items():
                if (atlas, metric, subject) in supplied:
                    continue
                supplied.add((atlas, metric, subject))
                found.setdefault((atlas, metric), {}).setdefault(subject, []).append(matrix)
    return found
```

Note the old code re-sorted the combined list (`for path in sorted(paths)`), which interleaved `.csv` among `.mat` by path; the new order is deliberately all `.mat` first. Update the function docstring to say one matrix per (atlas, metric, scan) per repeat is returned.

- [ ] **Step 4: Run the new tests, then the full suite**

Run: `python3 -m pytest tests/test_reliability.py -q` then `python3 -m pytest tests/ -q`
Expected: all pass (119 + 4 new = 123). If an *existing* test fails because it encoded the duplicate loading, do not edit its assertion to fit: report it, since that would be a real behavioural finding.

- [ ] **Step 5: Commit**

```bash
git add scripts/reliability.py tests/test_reliability.py
git commit -m "$(cat <<'EOF'
fix: load each repeat once in collect_matrices, not once per file format

DSI Studio writes count/qa twice per scan (combined .mat plus a converted
per-metric .csv); collect_matrices loaded both, so n_repeats read 4 for 2
repeats, repeatability was inflated (0.99727 vs 0.99590 true on study 129),
and a scan with one surviving repeat passed the >=2-repeats gate. The .mat
now wins; a .csv only supplies metrics the .mat lacks (MRtrix3 is CSV-only).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Session-aware wave staging

**Files:**
- Modify: `scripts/utils/discovery.py` (add `select_scans`; add `import logging, random, re`)
- Test: `tests/test_discovery.py`
- Modify: `scripts/cross_validation_bootstrap_optimizer.py`
- Test: new `tests/test_session_selection.py` (dry-run integration; model it on how `tests/test_cross_validation_repeats.py` drives `run_wave_pipeline`)

**Interfaces:**
- Consumes: `parse_subject_session(path) -> (subject | None, session | None)` from `scripts/utils/discovery.py` (filename-first).
- Produces: `select_scans(pool: list[Path], n_subjects: int, seed: int, sessions_per_subject: int = 0) -> list[Path]`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_discovery.py`; add `import logging, random` and `from scripts.utils.discovery import select_scans` at the top). This exact test set was run against a prototype of the implementation below and passes.

```python
def _scan(sub, ses, ext="fz"):
    return Path(f"/d/sub-{sub}/fib/sub-{sub}_ses-{ses}.odf.qsdr.{ext}")


_POOL = (
    [_scan("A", 1), _scan("A", 2), _scan("A", 3)]
    + [_scan("B", 1), _scan("B", 2)]
    + [_scan("C", 1)]
    + [_scan("D", 1), _scan("D", 2)]
)


def test_select_scans_matches_legacy_sampling_when_not_session_aware():
    pool = [_scan(str(i), 1) for i in range(10)]
    random.seed(42)
    legacy = random.sample(pool, 3)
    assert select_scans(pool, 3, 42, 0) == legacy
    assert select_scans(pool, 3, 42, 1) == legacy


def test_select_scans_uses_whole_pool_when_asked_for_more_than_exists():
    pool = [_scan(str(i), 1) for i in range(4)]
    assert select_scans(pool, 10, 42, 0) == pool


def test_select_scans_session_aware_takes_k_sessions_from_eligible_subjects_only():
    got = select_scans(_POOL, 2, 42, 2)
    assert len(got) == 4
    subjects = {p.name.split("_")[0] for p in got}
    assert len(subjects) == 2 and "sub-C" not in subjects  # C has a single session
    for s in subjects:
        assert sorted(p.name for p in got if p.name.startswith(s)) == [
            f"{s}_ses-1.odf.qsdr.fz",
            f"{s}_ses-2.odf.qsdr.fz",
        ]


def test_select_scans_is_deterministic():
    assert select_scans(_POOL, 2, 7, 2) == select_scans(_POOL, 2, 7, 2)


def test_select_scans_uses_all_eligible_subjects_when_n_exceeds_them():
    got = select_scans(_POOL, 99, 1, 2)
    assert {p.name.split("_")[0] for p in got} == {"sub-A", "sub-B", "sub-D"}
    assert len(got) == 6


def test_select_scans_falls_back_to_scan_sampling_without_multisession_subjects(caplog):
    pool = [_scan(str(i), 1) for i in range(10)]
    with caplog.at_level(logging.INFO):
        got = select_scans(pool, 3, 42, 2)
    random.seed(42)
    assert got == random.sample(pool, 3)
    assert "falling back" in caplog.text


def test_select_scans_skips_unparseable_names_with_a_warning(caplog):
    pool = _POOL + [Path("/d/.git/MD5E-abc.qsdr.fz")]
    with caplog.at_level(logging.WARNING):
        got = select_scans(pool, 99, 1, 2)
    assert all("MD5E" not in p.name for p in got)
    assert "1 of 9 scans" in caplog.text


def test_select_scans_prefers_fz_over_fib_gz_copy_of_the_same_scan():
    pool = [_scan("A", 1), _scan("A", 2), _scan("A", 1, "fib.gz"), _scan("A", 2, "fib.gz")]
    assert select_scans(pool, 1, 1, 2) == [_scan("A", 1), _scan("A", 2)]


def test_select_scans_orders_sessions_naturally():
    pool = [_scan("A", 1), _scan("A", 10), _scan("A", 2)]
    assert select_scans(pool, 1, 1, 2) == [_scan("A", 1), _scan("A", 2)]
```

- [ ] **Step 2: Run to verify they fail** — `python3 -m pytest tests/test_discovery.py -q` → FAIL with `ImportError: cannot import name 'select_scans'`.

- [ ] **Step 3: Implement `select_scans`** (append to `scripts/utils/discovery.py`; add `import logging`, `import random` next to the existing `import re`). Prototype-verified:

```python
def _natural_key(text: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def select_scans(
    pool: list[Path], n_subjects: int, seed: int, sessions_per_subject: int = 0
) -> list[Path]:
    """Choose the scans a wave stages.

    sessions_per_subject < 2 reproduces the legacy behaviour exactly: sample
    n_subjects individual scans (the whole pool when asked for at least as many).
    With sessions_per_subject >= 2, sample n_subjects *subjects* that have at
    least that many sessions and stage their first sessions_per_subject sessions
    (natural order of the session id), so within-subject between-session
    comparisons exist. Falls back to scan-level sampling when no subject has
    enough sessions, so single-session cohorts behave exactly as before.
    """

    def scan_level() -> list[Path]:
        if n_subjects >= len(pool):
            return list(pool)
        return random.Random(seed).sample(pool, n_subjects)

    if sessions_per_subject < 2:
        return scan_level()

    by_subject: dict[str, dict[str, Path]] = {}
    unparseable: list[Path] = []
    for path in pool:
        subject, session = parse_subject_session(path)
        if not subject or not session:
            unparseable.append(path)
            continue
        by_subject.setdefault(subject, {}).setdefault(session, path)  # first (.fz) wins
    if unparseable:
        logging.warning(
            "%d of %d scans had no parseable sub-<id>/ses-<id> and cannot be used "
            "for session-aware selection (e.g. %s)",
            len(unparseable),
            len(pool),
            ", ".join(p.name for p in unparseable[:3]),
        )
    eligible = sorted(s for s, sessions in by_subject.items() if len(sessions) >= sessions_per_subject)
    if not eligible:
        logging.info(
            "No subject has >=%d sessions; falling back to sampling individual scans",
            sessions_per_subject,
        )
        return scan_level()
    chosen = (
        eligible
        if n_subjects >= len(eligible)
        else sorted(random.Random(seed).sample(eligible, n_subjects))
    )
    selected: list[Path] = []
    for subject in chosen:
        sessions = sorted(by_subject[subject], key=_natural_key)[:sessions_per_subject]
        selected.extend(by_subject[subject][s] for s in sessions)
    return selected
```

- [ ] **Step 4: Run `python3 -m pytest tests/test_discovery.py -q`** → PASS.

- [ ] **Step 5: Wire it into wave staging** in `scripts/cross_validation_bootstrap_optimizer.py`.

(a) Import: `from scripts.utils.discovery import find_subject_files, select_scans`.

(b) At the selection block (currently `n_subjects = ...`, `seed = ...`, `random.seed(seed)`, the fz/fib pool, then `if n_subjects >= len(pool): selected = pool / else: selected = random.sample(pool, n_subjects)`), keep the `random.seed(seed)` line and the pool construction unchanged, and replace only the `if/else` selection with:

```python
    sessions_per_subject = int(wave_config["data_selection"].get("sessions_per_subject") or 0)
    selected = select_scans(pool, n_subjects, seed, sessions_per_subject)
    if sessions_per_subject >= 2:
        logging.info(
            " Session-aware selection: %d scans (%d subjects requested, up to %d sessions each)",
            len(selected), n_subjects, sessions_per_subject,
        )
```
Before removing the old `random.sample`, grep the rest of `run_wave_pipeline` for any later use of the *global* `random` module (`random.` calls) that could depend on the RNG state having been advanced by the old `random.sample`; report what you find. (`select_scans` uses its own `random.Random(seed)`, so the global state is no longer advanced; the selection itself is identical.)

(c) Config generators: add a parameter `sessions_per_subject: int = 2` to `generate_wave_configs` and `generate_single_wave_config`, and write `"sessions_per_subject": int(sessions_per_subject)` into every `data_selection` dict they build (both waves in `generate_wave_configs`, the single one in `generate_single_wave_config`).

(d) CLI: next to `--subjects`, add
```python
    parser.add_argument(
        "--sessions-per-subject",
        type=int,
        default=2,
        help=(
            "Sessions staged per sampled subject (default: 2). With >=2, --subjects counts "
            "SUBJECTS rather than scans, so a wave stages up to subjects x sessions scans "
            "(more tracking compute) but within-subject between-session comparisons become "
            "available to the variance decomposition. Falls back to sampling individual scans "
            "when no subject has enough sessions. Use 1 (or 0) for the legacy scan-level sampling."
        ),
    )
```
and pass `sessions_per_subject=args.sessions_per_subject` at each call to `generate_wave_configs` / `generate_single_wave_config` in `main()`.

- [ ] **Step 6: Integration test** (`tests/test_session_selection.py`). Model it on how `tests/test_cross_validation_repeats.py` calls `run_wave_pipeline(..., dry_run=True)`. Build a temp dataset of empty files `sub-00X/fib/sub-00X_ses-Y.odf.qsdr.fz` (6 subjects; give some 3 sessions, some 1), a wave config with `n_subjects=2`, `sessions_per_subject=2`, then assert the written `selected_files.txt` has 4 lines, two subjects with two sessions each, no single-session subject. Add a second case with `sessions_per_subject` absent (legacy config) asserting it selects 2 scans exactly as `select_scans(pool, 2, seed, 0)` would. Add a third asserting `generate_wave_configs(...)` and `generate_single_wave_config(...)` write `data_selection.sessions_per_subject == 2` by default and honour an explicit value.

- [ ] **Step 7: Full suite** — `python3 -m pytest tests/ -q` → all pass.

- [ ] **Step 8: Commit**

```bash
git add scripts/utils/discovery.py scripts/cross_validation_bootstrap_optimizer.py tests/test_discovery.py tests/test_session_selection.py
git commit -m "$(cat <<'EOF'
feat: session-aware wave staging so between_session is available by default

Wave staging sampled random scans, so with 3 requested subjects on a
171-subject x 2-session cohort the chance two scans shared a subject was ~1%
and the variance decomposition's headline between_session stratum was almost
never available. select_scans (pure, unit-tested) now samples subjects and
stages sessions_per_subject sessions of each (natural session order), falling
back to the legacy scan-level sampling when no subject has enough sessions.
--sessions-per-subject defaults to 2; configs without the key keep legacy
behaviour and identical seeds select identical scans.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Correct the documented numbers and document the new behaviour

**Files:**
- Modify: `docs/methods.md`, `docs/superpowers/specs/2026-09-18-discriminability-noise-floor-design.md`, and the doc that lists sweep CLI flags (grep `docs/` for `--subjects`)

**Background:** `docs/methods.md` and the spec quote "only repeatability separated them, across a span of 0.0021" for four study-129 candidates (fa in {0, 0.1} x turning angle in {0, 45}, AAL3, 5M tracts, 3 subjects, 2 repeats). That figure came from repeatability inflated by the duplicate loading fixed in Task 1, so it must be recomputed, not guessed.

- [ ] **Step 1: Recompute from the real sweep output.** The output tree is `/data/local/software/opticonn/studies/study129/verify_production/optimize/verify_production_params/combos/sweep_0001` ... `sweep_0004` (in the main checkout; read-only for you). For each combo dir run, from the worktree with the env active:

```python
from pathlib import Path
from scripts.reliability import score_combo, rank_with_fallback
root = Path("/data/local/software/opticonn/studies/study129/verify_production/optimize/verify_production_params/combos")
for d in sorted(root.glob("sweep_*")):
    rows = score_combo(d, {})
    best = rank_with_fallback(rows)
    print(d.name, {r["connectivity_metric"]: (round(r["repeatability"], 5), r["n_repeats"], round(r["discriminability"], 3)) for r in rows}, "best:", best["connectivity_metric"] if best else None)
```
Record the corrected per-candidate repeatability (on the metric the sweep ranked by, and `count`), the corrected `n_repeats` (should now be 2), the corrected span across the four candidates, and confirm discriminability is still 1.0 for all four. Also confirm the previously quoted ordering of candidates by repeatability is unchanged (report it either way).

- [ ] **Step 2: Edit the docs with the recomputed values.** In `docs/methods.md` (Known limitation section) and the spec's Problem section, replace the 0.0021 span with the recomputed span, keeping every other statement only if it still holds (if the candidate ordering changed, say so). In the spec add one short paragraph under Problem noting the duplicate-loading defect, its measured effect (repeatability 0.99727 reported vs 0.99590 true on one candidate; `n_repeats` 4 vs 2), and that it was fixed.

- [ ] **Step 3: Document the new sampling behaviour.** In `docs/methods.md` add a short subsection describing session-aware staging: `--sessions-per-subject` (default 2), that `--subjects` then counts subjects, the compute cost (subjects x sessions scans per wave), the single-session fallback, and this caveat about discriminability: because discriminability's unit is the scan, a subject's second session is compared against the first as a "different subject", which makes the test harder than before but is not a test-retest reliability estimate. Add the flag to the CLI flag doc found by the grep.

- [ ] **Step 4:** Run `python3 -m pytest tests/ -q` (docs only, so unchanged), then commit:

```bash
git add docs/
git commit -m "$(cat <<'EOF'
docs: correct repeatability figures and document session-aware sampling

The quoted repeatability span came from values inflated by duplicate matrix
loading (fixed in collect_matrices); recomputed from the real study 129
sweep. Documents --sessions-per-subject, its compute cost, the single-session
fallback, and what discriminability's scan-level unit means once a subject's
sessions are staged together.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

- **Spec/goal coverage:** duplicate loading -> Task 1; between_session unavailability -> Task 2; the wrong quoted numbers and the new behaviour's documentation -> Task 3. The decomposition's tracking-noise bias is corrected automatically by Task 1 because `compute_strata` consumes `collect_matrices`.
- **Backward compatibility** is pinned by two tests: legacy-equivalence of `select_scans` for `sessions_per_subject` in {0, 1}, and the dry-run test with the key absent from the wave config.
- **Known follow-up, not in this plan:** `tune-bayes`'s own subject sampling is unaffected and is not session-aware.
