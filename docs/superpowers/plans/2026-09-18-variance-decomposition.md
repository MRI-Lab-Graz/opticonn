# Variance Decomposition (Component 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Report, for a completed sweep, how much the connectome moves under four sources of variation (tracking noise, parameter choice, between-session, between-subject) as comparable dissimilarity distributions, so a user can judge whether their parameter choice matters relative to the effect they study.

**Architecture:** A new pure-Python module (`scripts/variance_decomposition.py`) walks a sweep's `optimize/` output tree, reuses `reliability.py`'s existing matrix loading and distance function (no new distance metric), groups matrix pairs into four strata, and writes a CSV + human-readable summary. It never touches `rank()` — this is diagnostic reporting, not a selection criterion. A prerequisite fix (`scripts/utils/discovery.py`) removes a live duplicate-file bug in both optimizer backends' subject discovery, and supplies the subject/session parsing the decomposition needs.

**Tech Stack:** Python 3.10+, numpy, pandas (already project dependencies). No new dependencies.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-18-discriminability-noise-floor-design.md` (Component 0 and Component 1 sections; Component 2 — the margin statistic — is out of scope for this plan).
- Reuse `scripts/reliability.py`'s `edge_vector` and `_distance` for all distance computation — no new distance metric.
- Every stratum is reported as a distribution (n, mean, median, IQR), never a point estimate.
- A stratum with zero pairs is reported `available: False` with a `reason` string — never a silent NaN.
- A stratum with fewer than `MIN_PAIRS_FOR_CONFIDENCE = 10` pairs is flagged `low_confidence: True`, not hidden.
- The decomposition must never be wired into `score_combo`, `rank`, or `rank_with_fallback` — diagnostic only.
- Any call into this module from the main sweep run must be wrapped so a failure in the decomposition can never fail the sweep itself.

---

## File Structure

- Create: `scripts/utils/discovery.py` — shared file discovery, replaces three drifted copies of the same logic.
- Create: `tests/test_discovery.py`
- Modify: `scripts/bayesian_optimizer.py` — two call sites use the new helper instead of raw `rglob`.
- Modify: `scripts/cross_validation_bootstrap_optimizer.py` — one call site uses the new helper; end-of-sweep hook calls the decomposition.
- Create: `scripts/variance_decomposition.py` — the decomposition module and CLI.
- Create: `tests/test_variance_decomposition.py`

---

### Task 1: Discovery helpers

**Files:**
- Create: `scripts/utils/discovery.py`
- Test: `tests/test_discovery.py`

**Interfaces:**
- Produces: `find_subject_files(root: Path, patterns: list[str]) -> list[Path]` — union of `root.rglob(pattern)` for each pattern in `patterns`, excluding any path with a `.git` component, deduplicated by `Path.resolve()` (first occurrence wins, original non-resolved path is what's returned).
- Produces: `parse_subject_session(path: str | Path) -> tuple[str | None, str | None]` — searches the string form of `path` for `sub-<id>` and `ses-<id>` (case-sensitive, alphanumeric id), returns `(f"sub-{id}", f"ses-{id}")` per match found, `None` for whichever isn't found.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_discovery.py
from pathlib import Path

import pytest

from scripts.utils.discovery import find_subject_files, parse_subject_session


def test_find_subject_files_excludes_git_paths(tmp_path):
    real = tmp_path / "sub-001" / "fib"
    real.mkdir(parents=True)
    real_file = real / "sub-001_ses-1.odf.qsdr.fz"
    real_file.write_bytes(b"data")

    annex_dir = tmp_path / "sub-001" / ".git" / "annex" / "objects" / "xx" / "yy" / "KEY.fz"
    annex_dir.mkdir(parents=True)
    (annex_dir / "KEY.fz").write_bytes(b"data")

    found = find_subject_files(tmp_path, ["*.fz"])

    assert found == [real_file]


def test_find_subject_files_dedupes_symlink_and_target(tmp_path):
    target_dir = tmp_path / "store"
    target_dir.mkdir()
    target = target_dir / "content.fz"
    target.write_bytes(b"data")

    friendly = tmp_path / "sub-001_ses-1.fz"
    friendly.symlink_to(target)

    found = find_subject_files(tmp_path, ["*.fz"])

    assert len(found) == 1


def test_find_subject_files_unions_multiple_patterns(tmp_path):
    (tmp_path / "a.fz").write_bytes(b"1")
    (tmp_path / "b.fib.gz").write_bytes(b"1")

    found = find_subject_files(tmp_path, ["*.fz", "*.fib.gz"])

    assert {p.name for p in found} == {"a.fz", "b.fib.gz"}


def test_find_subject_files_empty_dir_returns_empty_list(tmp_path):
    assert find_subject_files(tmp_path, ["*.fz"]) == []


@pytest.mark.parametrize(
    "path,expected",
    [
        ("sub-001_ses-1.odf.qsdr", ("sub-001", "ses-1")),
        ("/data/x/sub-079/fib/sub-079_ses-2.odf.qsdr.fz", ("sub-079", "ses-2")),
        ("sub-042", ("sub-042", None)),
        ("MD5E-s67517482--3e7da19b35854439806ebb6d2abc409b", (None, None)),
    ],
)
def test_parse_subject_session(path, expected):
    assert parse_subject_session(path) == expected


def test_parse_subject_session_accepts_path_object(tmp_path):
    p = tmp_path / "sub-005_ses-3.fz"
    assert parse_subject_session(p) == ("sub-005", "ses-3")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/local/software/opticonn && source braingraph_pipeline/bin/activate && python3 -m pytest tests/test_discovery.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.utils.discovery'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/utils/discovery.py
"""Shared file discovery for the DSI Studio and Bayesian optimizer backends.

Both backends used to run their own `rglob` calls directly, which drifted:
one gained an `is_file()` guard the other didn't, neither excluded `.git`.
Under a DataLad/git-annex dataset, `rglob("*.fz")` returns both the friendly
symlink (`sub-079/fib/sub-079_ses-1.odf.qsdr.fz`) and the annex object it
resolves to (`.git/annex/objects/xx/yy/KEY.fz`, itself a directory containing
a same-named file one level deeper) -- both satisfy `is_file()`, so every
scan was discovered twice under two different, non-obviously-related names.
"""

from __future__ import annotations

import re
from pathlib import Path


def find_subject_files(root: Path, patterns: list[str]) -> list[Path]:
    """Union of `root.rglob(pattern)` for each pattern, excluding VCS internals
    and de-duplicated by resolved path (first occurrence, in pattern order,
    then sorted order, wins and is what's returned -- not the resolved path).
    """
    seen: set[Path] = set()
    out: list[Path] = []
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if ".git" in path.parts:
                continue
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(path)
    return out


_SUBJECT_RE = re.compile(r"sub-([A-Za-z0-9]+)")
_SESSION_RE = re.compile(r"ses-([A-Za-z0-9]+)")


def parse_subject_session(path: str | Path) -> tuple[str | None, str | None]:
    """Parse `sub-<id>` and `ses-<id>` from the string form of `path`.

    Returns (None, None) for names with no `sub-` match at all, such as a
    git-annex content hash (`MD5E-s6751...`).
    """
    text = str(path)
    subject_match = _SUBJECT_RE.search(text)
    session_match = _SESSION_RE.search(text)
    subject = f"sub-{subject_match.group(1)}" if subject_match else None
    session = f"ses-{session_match.group(1)}" if session_match else None
    return subject, session
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/local/software/opticonn && source braingraph_pipeline/bin/activate && python3 -m pytest tests/test_discovery.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
cd /data/local/software/opticonn
git add scripts/utils/discovery.py tests/test_discovery.py
git commit -m "$(cat <<'EOF'
feat: add shared file discovery, dedupes git-annex symlink/object pairs

Both bayesian_optimizer.py and cross_validation_bootstrap_optimizer.py ran
their own rglob logic, which drifted. Under a DataLad dataset, rglob("*.fz")
returns each scan twice: the friendly symlink and the annex object it
resolves to. One consolidated helper for both backends, plus subject/session
parsing that variance_decomposition.py depends on.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Wire discovery helper into both optimizer backends

**Files:**
- Modify: `scripts/bayesian_optimizer.py:249-266` (`_get_all_subjects`)
- Modify: `scripts/bayesian_optimizer.py:1392-1396` (CLI pre-check in `main()`)
- Modify: `scripts/cross_validation_bootstrap_optimizer.py:478-489` (wave file listing)

**Interfaces:**
- Consumes: `find_subject_files(root: Path, patterns: list[str]) -> list[Path]` from Task 1.

- [ ] **Step 1: Replace `_get_all_subjects` in `scripts/bayesian_optimizer.py`**

Current code (lines 249-266):
```python
    def _get_all_subjects(self) -> List[Path]:
        """Get list of all subject files in data directory.

        Recursive (rglob), matching cross_validation_bootstrap_optimizer.py --
        a bare glob() only sees data_dir's top level, which for a nested
        sub-*/fib/*.fz layout finds zero real subjects. Prefers .fz over
        .fib.gz (only falls back when no .fz exist) since study-level
        .fib.gz files here are typically longitudinal diffs (e.g.
        "longitudinal_ses-2_minus_ses-1.db.fib.gz"), not per-subject data.
        """
        # is_file() guards against git-annex object paths, where an
        # intermediate directory can share the leaf file's exact name
        # (.git/annex/objects/xx/yy/KEY.qsdr.fz/KEY.qsdr.fz).
        fz_files = sorted(p for p in self.data_dir.rglob("*.fz") if p.is_file())
        all_files = fz_files or sorted(
            p for p in self.data_dir.rglob("*.fib.gz") if p.is_file()
        )
        if not all_files:
            logger.warning(f"  No .fz or .fib.gz files found in {self.data_dir}")
        return all_files
```

Replace with:
```python
    def _get_all_subjects(self) -> List[Path]:
        """Get list of all subject files in data directory.

        Recursive, matching cross_validation_bootstrap_optimizer.py -- a bare
        glob() only sees data_dir's top level, which for a nested
        sub-*/fib/*.fz layout finds zero real subjects. Prefers .fz over
        .fib.gz (only falls back when no .fz exist) since study-level
        .fib.gz files here are typically longitudinal diffs (e.g.
        "longitudinal_ses-2_minus_ses-1.db.fib.gz"), not per-subject data.
        """
        fz_files = find_subject_files(self.data_dir, ["*.fz"])
        all_files = fz_files or find_subject_files(self.data_dir, ["*.fib.gz"])
        if not all_files:
            logger.warning(f"  No .fz or .fib.gz files found in {self.data_dir}")
        return all_files
```

- [ ] **Step 2: Replace the CLI pre-check in the same file**

Current code (lines 1392-1396):
```python
    # Check for .fz or .fib.gz files (recursive -- data_dir is typically a
    # nested sub-*/fib/*.fz layout, not flat; see _get_all_subjects())
    fz_files = [p for p in data_path.rglob("*.fz") if p.is_file()]
    fib_gz_files = [p for p in data_path.rglob("*.fib.gz") if p.is_file()]
    all_data_files = fz_files + fib_gz_files
```

Replace with:
```python
    # Check for .fz or .fib.gz files (recursive -- data_dir is typically a
    # nested sub-*/fib/*.fz layout, not flat; see _get_all_subjects())
    fz_files = find_subject_files(data_path, ["*.fz"])
    fib_gz_files = find_subject_files(data_path, ["*.fib.gz"])
    all_data_files = fz_files + fib_gz_files
```

- [ ] **Step 3: Add the import**

Find the existing import block near the top of `scripts/bayesian_optimizer.py` (look for `from pathlib import Path` or similar early imports) and add:

```python
from scripts.utils.discovery import find_subject_files
```

- [ ] **Step 4: Replace file listing in `scripts/cross_validation_bootstrap_optimizer.py`**

Current code (lines 478-489):
```python
    # List all available files in source and save manifest
    try:
        src_dir = Path(wave_config["data_selection"]["source_dir"])
        patterns = [wave_config["data_selection"].get("file_pattern", "*.fz")]
        files = []
        # is_file() guards against git-annex object paths, where an intermediate
        # directory can share the leaf file's exact name
        # (.git/annex/objects/xx/yy/KEY.qsdr.fz/KEY.qsdr.fz).
        for pat in patterns:
            files.extend(sorted([p for p in src_dir.rglob(pat) if p.is_file()]))
        # Also include .fib.gz if not already covered
        files.extend(sorted([p for p in src_dir.rglob("*.fib.gz") if p.is_file()]))
        # Deduplicate
        seen = set()
        uniq = []
```

Replace with:
```python
    # List all available files in source and save manifest
    try:
        src_dir = Path(wave_config["data_selection"]["source_dir"])
        patterns = [wave_config["data_selection"].get("file_pattern", "*.fz")]
        files = []
        for pat in patterns:
            files.extend(find_subject_files(src_dir, [pat]))
        # Also include .fib.gz if not already covered
        files.extend(find_subject_files(src_dir, ["*.fib.gz"]))
        # Deduplicate (patterns list and the .fib.gz fallback can overlap;
        # find_subject_files only dedupes within its own single call)
        seen = set()
        uniq = []
```

- [ ] **Step 5: Add the import to `cross_validation_bootstrap_optimizer.py`**

Add next to the existing `from scripts.reliability import ...` line (around line 26):

```python
from scripts.utils.discovery import find_subject_files
```

- [ ] **Step 6: Run the full test suite**

Run: `cd /data/local/software/opticonn && source braingraph_pipeline/bin/activate && python3 -m pytest tests/ -q`
Expected: PASS, same count as before this task (87 or more, none fewer).

- [ ] **Step 7: Manual smoke check against the real dataset**

Run:
```bash
cd /data/local/software/opticonn && source braingraph_pipeline/bin/activate && python3 -c "
from pathlib import Path
from scripts.utils.discovery import find_subject_files
root = Path('/data/local/129_PK01/derivatives/dsi_newqsiprep/dsistudio')
files = find_subject_files(root, ['*.fz'])
print('found:', len(files))
assert len(files) == 342, f'expected 342 real scans, got {len(files)}'
print('OK')
"
```
Expected: `found: 342` then `OK` (342 is the known real-scan count on this dataset; 684 would mean the git-annex duplicate is back).

- [ ] **Step 8: Commit**

```bash
cd /data/local/software/opticonn
git add scripts/bayesian_optimizer.py scripts/cross_validation_bootstrap_optimizer.py
git commit -m "$(cat <<'EOF'
fix: use shared discovery helper in both optimizer backends

Both call sites duplicated rglob + is_file() filtering logic that never
excluded .git -- on a DataLad dataset this found every scan twice (once as
the friendly symlink, once as the git-annex object). Switches both backends
to scripts.utils.discovery.find_subject_files, which also excludes .git.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Collect matrices across a sweep

**Files:**
- Create: `scripts/variance_decomposition.py`
- Test: `tests/test_variance_decomposition.py`

**Interfaces:**
- Consumes: `scripts.reliability.collect_matrices(combo_dir: Path) -> dict[tuple[str, str], dict[str, list[np.ndarray]]]` (existing function; keys are `(atlas, metric)`, values are `{subj_sess_key: [matrix, ...]}` where `subj_sess_key` is e.g. `"sub-001_ses-1"`).
- Produces: `collect_sweep_matrices(sweep_optimize_dir: Path) -> dict[tuple[str, str], dict[str, dict[str, list[np.ndarray]]]]` — keys are `(atlas, metric)`; values are `{combo_id: {subj_sess_key: [matrix, ...]}}` where `combo_id` is `"<wave_name>/<sweep_id>"`.

This task's fixture (a small tree of real-shaped `.connectivity.mat` files) is reused by Task 4 and Task 5's tests, so build it as a shared pytest fixture function, not inline in one test.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_variance_decomposition.py
from pathlib import Path

import numpy as np
import scipy.io

from scripts.variance_decomposition import collect_sweep_matrices

N_NODES = 10


def _make_matrix(rng, scale=100.0):
    upper = np.triu(rng.random((N_NODES, N_NODES)) * scale, 1)
    return upper + upper.T


def _write_combo(
    combo_dir: Path,
    atlas: str,
    metric: str,
    subj_sess_reps: dict[str, list[np.ndarray]],
) -> None:
    """subj_sess_reps: {"sub-001_ses-1": [matrix_rep1, matrix_rep2, ...]}"""
    for subj_sess, reps in subj_sess_reps.items():
        for rep_idx, matrix in enumerate(reps, start=1):
            rep_dir = combo_dir / f"rep_{rep_idx}" / "results" / atlas
            rep_dir.mkdir(parents=True, exist_ok=True)
            out = rep_dir / f"{subj_sess}_{atlas}.tt.gz.{atlas}..pass.connectivity.mat"
            scipy.io.savemat(str(out), {"connectivity": matrix})


def build_sweep_fixture(tmp_path: Path) -> Path:
    """A 2-wave, 2-combo sweep: wave1/sweep_0001, wave1/sweep_0002.
    3 subjects, 2 of which (sub-001, sub-002) have 2 sessions each; sub-003
    has one session; 2 reps per subj_sess per combo.
    """
    rng = np.random.default_rng(0)
    optimize_dir = tmp_path / "optimize"

    subj_sess_keys = [
        "sub-001_ses-1",
        "sub-001_ses-2",
        "sub-002_ses-1",
        "sub-002_ses-2",
        "sub-003_ses-1",
    ]

    for combo_id in ("sweep_0001", "sweep_0002"):
        combo_dir = optimize_dir / "wave1" / "combos" / combo_id
        reps = {key: [_make_matrix(rng), _make_matrix(rng)] for key in subj_sess_keys}
        _write_combo(combo_dir, "AAL3", "count", reps)

    return optimize_dir


def test_collect_sweep_matrices_groups_by_atlas_metric_combo_subject(tmp_path):
    optimize_dir = build_sweep_fixture(tmp_path)

    grouped = collect_sweep_matrices(optimize_dir)

    assert ("AAL3", "count") in grouped
    combo_matrices = grouped[("AAL3", "count")]
    assert set(combo_matrices) == {"wave1/sweep_0001", "wave1/sweep_0002"}
    one_combo = combo_matrices["wave1/sweep_0001"]
    assert set(one_combo) == {
        "sub-001_ses-1",
        "sub-001_ses-2",
        "sub-002_ses-1",
        "sub-002_ses-2",
        "sub-003_ses-1",
    }
    assert len(one_combo["sub-001_ses-1"]) == 2  # 2 reps


def test_collect_sweep_matrices_empty_tree_returns_empty_dict(tmp_path):
    optimize_dir = tmp_path / "optimize"
    optimize_dir.mkdir()

    assert collect_sweep_matrices(optimize_dir) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/local/software/opticonn && source braingraph_pipeline/bin/activate && python3 -m pytest tests/test_variance_decomposition.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.variance_decomposition'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/variance_decomposition.py
"""Variance decomposition across a completed OptiConn sweep.

Reports how much the connectome moves under four sources of variation --
tracking noise, parameter choice, between-session, and between-subject -- as
comparable dissimilarity distributions, so a user can judge whether their
parameter choice matters relative to the effect they study.

This is diagnostic reporting, not a selection criterion: nothing here is
consumed by scripts.reliability.rank() or rank_with_fallback(). See
docs/superpowers/specs/2026-09-18-discriminability-noise-floor-design.md
for why discriminability alone saturates and what this adds.
"""

from __future__ import annotations

from pathlib import Path

from scripts.reliability import collect_matrices

MIN_PAIRS_FOR_CONFIDENCE = 10


def collect_sweep_matrices(
    sweep_optimize_dir: Path,
) -> dict[tuple[str, str], dict[str, dict[str, list]]]:
    """{(atlas, metric): {combo_id: {subj_sess_key: [matrix, ...]}}}

    combo_id is "<wave_dir_name>/<sweep_dir_name>", e.g. "wave1/sweep_0001" --
    scoped to the wave so two waves that happen to both have a "sweep_0001"
    (independent grid sampling per wave) are never merged into one candidate.
    """
    result: dict[tuple[str, str], dict[str, dict[str, list]]] = {}
    combo_dirs = sorted(Path(sweep_optimize_dir).glob("*/combos/sweep_*"))
    for combo_dir in combo_dirs:
        combo_id = f"{combo_dir.parent.parent.name}/{combo_dir.name}"
        found = collect_matrices(combo_dir)
        for (atlas, metric), subj_map in found.items():
            result.setdefault((atlas, metric), {})[combo_id] = subj_map
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /data/local/software/opticonn && source braingraph_pipeline/bin/activate && python3 -m pytest tests/test_variance_decomposition.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
cd /data/local/software/opticonn
git add scripts/variance_decomposition.py tests/test_variance_decomposition.py
git commit -m "$(cat <<'EOF'
feat: start variance_decomposition module -- collect matrices across a sweep

First piece of Component 1 (see the 2026-09-18 discriminability spec):
walks a sweep's optimize/ tree and groups every evaluated connectivity
matrix by (atlas, metric, combo, subject-session), reusing
reliability.collect_matrices per combo rather than reimplementing matrix
loading.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Compute the four strata

**Files:**
- Modify: `scripts/variance_decomposition.py`
- Test: `tests/test_variance_decomposition.py`

**Interfaces:**
- Consumes: `collect_sweep_matrices(...)` output, narrowed to one `(atlas, metric)`'s `dict[combo_id, dict[subj_sess_key, list[np.ndarray]]]` (Task 3). `scripts.reliability.edge_vector(matrix) -> np.ndarray` and `scripts.reliability._distance(a, b) -> float` (existing; `_distance` already returns `1.0 - corrcoef`, i.e. dissimilarity directly).
- Produces: `compute_strata(combo_matrices: dict[str, dict[str, list]]) -> dict[str, dict]` — keys are `"tracking_noise"`, `"parameter"`, `"between_session"`, `"between_subject"`; each value is `{"dissimilarities": list[float], "available": bool, "reason": str | None}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_variance_decomposition.py`:

```python
from scripts.variance_decomposition import compute_strata


def test_compute_strata_tracking_noise_is_within_subject_within_combo(tmp_path):
    optimize_dir = build_sweep_fixture(tmp_path)
    grouped = collect_sweep_matrices(optimize_dir)
    combo_matrices = grouped[("AAL3", "count")]

    strata = compute_strata(combo_matrices)

    # 5 subj_sess keys x 2 combos x 1 pair (2 reps -> C(2,2)=1 pair) = 10
    assert strata["tracking_noise"]["available"] is True
    assert len(strata["tracking_noise"]["dissimilarities"]) == 10


def test_compute_strata_parameter_compares_same_subject_across_combos(tmp_path):
    optimize_dir = build_sweep_fixture(tmp_path)
    grouped = collect_sweep_matrices(optimize_dir)
    combo_matrices = grouped[("AAL3", "count")]

    strata = compute_strata(combo_matrices)

    # 5 subj_sess keys, each present in both combos -> C(2,2)=1 pair each = 5
    assert strata["parameter"]["available"] is True
    assert len(strata["parameter"]["dissimilarities"]) == 5


def test_compute_strata_between_session_only_pairs_same_subject(tmp_path):
    optimize_dir = build_sweep_fixture(tmp_path)
    grouped = collect_sweep_matrices(optimize_dir)
    combo_matrices = grouped[("AAL3", "count")]

    strata = compute_strata(combo_matrices)

    # sub-001 (2 sessions) + sub-002 (2 sessions) -> 1 pair each, x 2 combos = 2
    # sub-003 has 1 session, contributes nothing.
    assert strata["between_session"]["available"] is True
    assert len(strata["between_session"]["dissimilarities"]) == 2


def test_compute_strata_between_subject_excludes_same_subject_pairs(tmp_path):
    optimize_dir = build_sweep_fixture(tmp_path)
    grouped = collect_sweep_matrices(optimize_dir)
    combo_matrices = grouped[("AAL3", "count")]

    strata = compute_strata(combo_matrices)

    # 5 keys -> C(5,2)=10 raw pairs per combo, minus the 2 same-subject pairs
    # (sub-001 ses1/ses2, sub-002 ses1/ses2) = 8 per combo, x 2 combos = 16
    assert strata["between_subject"]["available"] is True
    assert len(strata["between_subject"]["dissimilarities"]) == 16


def test_compute_strata_between_session_unavailable_for_single_session_cohort(tmp_path):
    rng = np.random.default_rng(1)
    optimize_dir = tmp_path / "optimize"
    combo_dir = optimize_dir / "wave1" / "combos" / "sweep_0001"
    reps = {
        "sub-001_ses-1": [_make_matrix(rng), _make_matrix(rng)],
        "sub-002_ses-1": [_make_matrix(rng), _make_matrix(rng)],
    }
    _write_combo(combo_dir, "AAL3", "count", reps)

    grouped = collect_sweep_matrices(optimize_dir)
    strata = compute_strata(grouped[("AAL3", "count")])

    assert strata["between_session"]["available"] is False
    assert strata["between_session"]["dissimilarities"] == []
    assert strata["between_session"]["reason"] is not None


def test_compute_strata_parameter_unavailable_for_single_combo(tmp_path):
    rng = np.random.default_rng(2)
    optimize_dir = tmp_path / "optimize"
    combo_dir = optimize_dir / "wave1" / "combos" / "sweep_0001"
    reps = {"sub-001_ses-1": [_make_matrix(rng), _make_matrix(rng)]}
    _write_combo(combo_dir, "AAL3", "count", reps)

    grouped = collect_sweep_matrices(optimize_dir)
    strata = compute_strata(grouped[("AAL3", "count")])

    assert strata["parameter"]["available"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/local/software/opticonn && source braingraph_pipeline/bin/activate && python3 -m pytest tests/test_variance_decomposition.py -v -k compute_strata`
Expected: FAIL with `ImportError: cannot import name 'compute_strata'`

- [ ] **Step 3: Write the implementation**

Append to `scripts/variance_decomposition.py` (add the import at the top alongside the existing one):

```python
from scripts.reliability import edge_vector, _distance
from scripts.utils.discovery import parse_subject_session


def _entry(dissimilarities: list[float], reason_if_empty: str) -> dict:
    return {
        "dissimilarities": dissimilarities,
        "available": len(dissimilarities) > 0,
        "reason": None if dissimilarities else reason_if_empty,
    }


def compute_strata(combo_matrices: dict[str, dict[str, list]]) -> dict[str, dict]:
    """Four dissimilarity-pair strata for one (atlas, metric)'s sweep matrices.

    combo_matrices: {combo_id: {subj_sess_key: [matrix, ...]}}, as produced by
    one value of collect_sweep_matrices()'s return dict.
    """
    combo_ids = sorted(combo_matrices)

    vecs_all_reps: dict[tuple[str, str], list] = {}
    vec_rep0: dict[tuple[str, str], object] = {}
    subject_of: dict[str, str | None] = {}
    session_of: dict[str, str | None] = {}

    for combo_id in combo_ids:
        for key, matrices in combo_matrices[combo_id].items():
            vecs = [edge_vector(m) for m in matrices]
            vecs_all_reps[(combo_id, key)] = vecs
            vec_rep0[(combo_id, key)] = vecs[0]
            if key not in subject_of:
                subject, session = parse_subject_session(key)
                subject_of[key] = subject
                session_of[key] = session

    tracking_noise: list[float] = []
    for vecs in vecs_all_reps.values():
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                tracking_noise.append(_distance(vecs[i], vecs[j]))

    key_to_combos: dict[str, list[str]] = {}
    for combo_id, key in vec_rep0:
        key_to_combos.setdefault(key, []).append(combo_id)

    parameter: list[float] = []
    for key, combos in key_to_combos.items():
        combos = sorted(combos)
        for i in range(len(combos)):
            for j in range(i + 1, len(combos)):
                a = vec_rep0[(combos[i], key)]
                b = vec_rep0[(combos[j], key)]
                parameter.append(_distance(a, b))

    between_session: list[float] = []
    between_subject: list[float] = []
    for combo_id in combo_ids:
        keys = sorted(combo_matrices[combo_id])

        subject_to_keys: dict[str, list[str]] = {}
        for key in keys:
            subject = subject_of.get(key)
            session = session_of.get(key)
            if subject and session:
                subject_to_keys.setdefault(subject, []).append(key)
        for subject, sess_keys in subject_to_keys.items():
            sess_keys = sorted(sess_keys)
            for i in range(len(sess_keys)):
                for j in range(i + 1, len(sess_keys)):
                    a = vec_rep0[(combo_id, sess_keys[i])]
                    b = vec_rep0[(combo_id, sess_keys[j])]
                    between_session.append(_distance(a, b))

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                subject_i = subject_of.get(keys[i]) or keys[i]
                subject_j = subject_of.get(keys[j]) or keys[j]
                if subject_i == subject_j:
                    continue
                a = vec_rep0[(combo_id, keys[i])]
                b = vec_rep0[(combo_id, keys[j])]
                between_subject.append(_distance(a, b))

    return {
        "tracking_noise": _entry(
            tracking_noise, "no combo had >=2 tracking repeats for any subject"
        ),
        "parameter": _entry(
            parameter,
            "no subject/session was evaluated under >=2 candidate parameter sets",
        ),
        "between_session": _entry(
            between_session,
            "not available: no subject had >=2 parsed sessions under one candidate "
            "(single-session cohort, or subject/session identifiers did not parse)",
        ),
        "between_subject": _entry(between_subject, "fewer than 2 distinct subjects found"),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/local/software/opticonn && source braingraph_pipeline/bin/activate && python3 -m pytest tests/test_variance_decomposition.py -v`
Expected: PASS, all tests including Task 3's.

- [ ] **Step 5: Commit**

```bash
cd /data/local/software/opticonn
git add scripts/variance_decomposition.py tests/test_variance_decomposition.py
git commit -m "$(cat <<'EOF'
feat: compute the four variance-decomposition strata

tracking_noise (within subject-session, within combo, across reps),
parameter (same subject-session, across combos), between_session (same
subject, across sessions, within one combo), between_subject (different
subjects, within one combo). Each stratum is a pooled list of dissimilarities
(1 - correlation, via reliability's existing edge_vector/_distance), reported
available=False with a reason rather than an empty/NaN result when a stratum
has no data -- e.g. between_session on a single-session cohort.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Summarize strata and compute ratios

**Files:**
- Modify: `scripts/variance_decomposition.py`
- Test: `tests/test_variance_decomposition.py`

**Interfaces:**
- Consumes: one value from `compute_strata(...)`'s return dict (Task 4): `{"dissimilarities": list[float], "available": bool, "reason": str | None}`.
- Produces: `summarize_stratum(entry: dict) -> dict` — `{"n": int, "mean": float | None, "median": float | None, "iqr_low": float | None, "iqr_high": float | None, "available": bool, "low_confidence": bool, "reason": str | None}`.
- Produces: `compute_ratios(summaries: dict[str, dict]) -> dict[str, float | None]` — keys `"parameter_over_between_session"`, `"tracking_noise_over_parameter"`, `"tracking_noise_over_between_subject"`, `"between_session_over_between_subject"`, `"parameter_over_between_subject"`; `summaries` maps stratum name to a `summarize_stratum(...)` result.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_variance_decomposition.py`:

```python
from scripts.variance_decomposition import (
    MIN_PAIRS_FOR_CONFIDENCE,
    compute_ratios,
    summarize_stratum,
)


def test_summarize_stratum_computes_distribution_stats():
    entry = {"dissimilarities": [0.1, 0.2, 0.3, 0.4], "available": True, "reason": None}

    summary = summarize_stratum(entry)

    assert summary["n"] == 4
    assert summary["mean"] == pytest.approx(0.25)
    assert summary["median"] == pytest.approx(0.25)
    assert summary["available"] is True
    assert summary["low_confidence"] is True  # 4 < MIN_PAIRS_FOR_CONFIDENCE


def test_summarize_stratum_not_low_confidence_above_threshold():
    entry = {
        "dissimilarities": [0.1] * MIN_PAIRS_FOR_CONFIDENCE,
        "available": True,
        "reason": None,
    }

    summary = summarize_stratum(entry)

    assert summary["low_confidence"] is False


def test_summarize_stratum_unavailable_passes_through_reason():
    entry = {"dissimilarities": [], "available": False, "reason": "single-session cohort"}

    summary = summarize_stratum(entry)

    assert summary["available"] is False
    assert summary["n"] == 0
    assert summary["mean"] is None
    assert summary["reason"] == "single-session cohort"


def test_compute_ratios_divides_means():
    summaries = {
        "parameter": {"available": True, "mean": 0.2, "n": 20},
        "between_session": {"available": True, "mean": 0.1, "n": 20},
        "tracking_noise": {"available": True, "mean": 0.02, "n": 20},
        "between_subject": {"available": True, "mean": 0.4, "n": 20},
    }

    ratios = compute_ratios(summaries)

    assert ratios["parameter_over_between_session"] == pytest.approx(2.0)
    assert ratios["tracking_noise_over_parameter"] == pytest.approx(0.1)


def test_compute_ratios_none_when_a_stratum_unavailable():
    summaries = {
        "parameter": {"available": True, "mean": 0.2, "n": 20},
        "between_session": {"available": False, "mean": None, "n": 0},
        "tracking_noise": {"available": True, "mean": 0.02, "n": 20},
        "between_subject": {"available": True, "mean": 0.4, "n": 20},
    }

    ratios = compute_ratios(summaries)

    assert ratios["parameter_over_between_session"] is None
    assert ratios["tracking_noise_over_parameter"] is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/local/software/opticonn && source braingraph_pipeline/bin/activate && python3 -m pytest tests/test_variance_decomposition.py -v -k "summarize_stratum or compute_ratios"`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**

Append to `scripts/variance_decomposition.py` (add `import numpy as np` at the top alongside the existing imports):

```python
import numpy as np


def summarize_stratum(entry: dict) -> dict:
    if not entry["available"]:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "iqr_low": None,
            "iqr_high": None,
            "available": False,
            "low_confidence": False,
            "reason": entry["reason"],
        }
    values = np.asarray(entry["dissimilarities"], dtype=float)
    n = len(values)
    return {
        "n": n,
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "iqr_low": float(np.percentile(values, 25)),
        "iqr_high": float(np.percentile(values, 75)),
        "available": True,
        "low_confidence": n < MIN_PAIRS_FOR_CONFIDENCE,
        "reason": None,
    }


_RATIO_PAIRS = [
    ("parameter_over_between_session", "parameter", "between_session"),
    ("tracking_noise_over_parameter", "tracking_noise", "parameter"),
    ("tracking_noise_over_between_subject", "tracking_noise", "between_subject"),
    ("between_session_over_between_subject", "between_session", "between_subject"),
    ("parameter_over_between_subject", "parameter", "between_subject"),
]


def compute_ratios(summaries: dict[str, dict]) -> dict[str, float | None]:
    ratios: dict[str, float | None] = {}
    for ratio_name, numerator_key, denominator_key in _RATIO_PAIRS:
        numerator = summaries.get(numerator_key)
        denominator = summaries.get(denominator_key)
        if (
            not numerator
            or not denominator
            or not numerator["available"]
            or not denominator["available"]
            or denominator["mean"] == 0
        ):
            ratios[ratio_name] = None
        else:
            ratios[ratio_name] = numerator["mean"] / denominator["mean"]
    return ratios
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/local/software/opticonn && source braingraph_pipeline/bin/activate && python3 -m pytest tests/test_variance_decomposition.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
cd /data/local/software/opticonn
git add scripts/variance_decomposition.py tests/test_variance_decomposition.py
git commit -m "$(cat <<'EOF'
feat: summarize strata as distributions and compute headline ratios

summarize_stratum turns a pooled dissimilarity list into n/mean/median/IQR,
flagging low_confidence below MIN_PAIRS_FOR_CONFIDENCE=10 rather than hiding
a thin sample. compute_ratios derives the decision-relevant comparisons
(parameter effect vs between-session effect, tracking noise as a fraction of
the parameter effect, etc.), returning None rather than dividing by an
unavailable or zero denominator.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Output writer, headline text, orchestration, and CLI

**Files:**
- Modify: `scripts/variance_decomposition.py`
- Test: `tests/test_variance_decomposition.py`

**Interfaces:**
- Consumes: `collect_sweep_matrices`, `compute_strata`, `summarize_stratum`, `compute_ratios` (Tasks 3-5).
- Produces: `headline_text(atlas: str, metric: str, ratios: dict) -> str`.
- Produces: `write_decomposition(output_dir: Path, atlas: str, metric: str, summaries: dict, ratios: dict) -> None` — appends to `output_dir/variance_decomposition.csv` (writes header on first call, i.e. when the file doesn't yet exist) and `output_dir/variance_decomposition_summary.txt`.
- Produces: `run(sweep_optimize_dir: Path, output_dir: Path) -> dict[tuple[str, str], dict]` — top-level orchestration; return value maps `(atlas, metric)` to `{"summaries": dict, "ratios": dict}`. This is what Task 7's integration hook calls.
- Produces: CLI `python scripts/variance_decomposition.py <sweep_optimize_dir> [-o OUTPUT_DIR]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_variance_decomposition.py`:

```python
import csv

from scripts.variance_decomposition import headline_text, run, write_decomposition


def test_headline_text_reports_ratio_and_noise_fraction():
    ratios = {"parameter_over_between_session": 1.23, "tracking_noise_over_parameter": 0.025}

    text = headline_text("AAL3", "count", ratios)

    assert "AAL3" in text
    assert "count" in text
    assert "1.2" in text  # 1.23x, allow for rounding


def test_headline_text_handles_missing_ratio():
    text = headline_text("AAL3", "count", {"parameter_over_between_session": None})

    assert "not available" in text.lower()


def test_write_decomposition_creates_csv_with_header_once(tmp_path):
    summaries = {
        "tracking_noise": summarize_stratum({"dissimilarities": [0.01, 0.02], "available": True, "reason": None}),
        "parameter": summarize_stratum({"dissimilarities": [0.1] * 12, "available": True, "reason": None}),
        "between_session": summarize_stratum({"dissimilarities": [], "available": False, "reason": "no repeats"}),
        "between_subject": summarize_stratum({"dissimilarities": [0.3, 0.4, 0.5], "available": True, "reason": None}),
    }
    ratios = compute_ratios(summaries)

    write_decomposition(tmp_path, "AAL3", "count", summaries, ratios)
    write_decomposition(tmp_path, "AAL3", "qa", summaries, ratios)

    csv_path = tmp_path / "variance_decomposition.csv"
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 8  # 4 strata x 2 metrics
    assert rows[0]["atlas"] == "AAL3"
    assert rows[0]["metric"] == "count"
    assert {r["stratum"] for r in rows[:4]} == {
        "tracking_noise", "parameter", "between_session", "between_subject",
    }

    summary_path = tmp_path / "variance_decomposition_summary.txt"
    assert summary_path.exists()
    text = summary_path.read_text()
    assert "AAL3 / count" in text
    assert "AAL3 / qa" in text


def test_run_end_to_end_writes_output_for_every_atlas_metric(tmp_path):
    optimize_dir = build_sweep_fixture(tmp_path)
    output_dir = tmp_path / "optimization_results"

    results = run(optimize_dir, output_dir)

    assert ("AAL3", "count") in results
    assert results[("AAL3", "count")]["summaries"]["tracking_noise"]["available"] is True
    assert (output_dir / "variance_decomposition.csv").exists()
    assert (output_dir / "variance_decomposition_summary.txt").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/local/software/opticonn && source braingraph_pipeline/bin/activate && python3 -m pytest tests/test_variance_decomposition.py -v -k "headline_text or write_decomposition or run_end_to_end"`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**

Append to `scripts/variance_decomposition.py` (add `import argparse`, `import csv`, `import logging`, `import sys` at the top alongside the existing imports):

```python
import argparse
import csv
import logging
import sys


def headline_text(atlas: str, metric: str, ratios: dict) -> str:
    parameter_ratio = ratios.get("parameter_over_between_session")
    if parameter_ratio is None:
        return (
            f"{atlas}/{metric}: parameter-vs-session ratio not available "
            "(needs a cohort with >=2 sessions per subject)"
        )
    noise_note = ""
    noise_ratio = ratios.get("tracking_noise_over_parameter")
    if noise_ratio is not None:
        noise_note = f" (tracking noise is {noise_ratio * 100:.1f}% of the parameter effect)"
    return (
        f"{atlas}/{metric}: parameter choice moves the connectome "
        f"{parameter_ratio:.2f}x the size of the between-session effect{noise_note}"
    )


_CSV_COLUMNS = [
    "atlas", "metric", "stratum", "n", "mean_dissimilarity", "median_dissimilarity",
    "iqr_low", "iqr_high", "available", "low_confidence", "reason",
]

_STRATUM_ORDER = ["tracking_noise", "parameter", "between_session", "between_subject"]


def write_decomposition(
    output_dir: Path, atlas: str, metric: str, summaries: dict, ratios: dict
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "variance_decomposition.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for stratum in _STRATUM_ORDER:
            s = summaries[stratum]
            writer.writerow(
                {
                    "atlas": atlas,
                    "metric": metric,
                    "stratum": stratum,
                    "n": s["n"],
                    "mean_dissimilarity": s["mean"],
                    "median_dissimilarity": s["median"],
                    "iqr_low": s["iqr_low"],
                    "iqr_high": s["iqr_high"],
                    "available": s["available"],
                    "low_confidence": s["low_confidence"],
                    "reason": s["reason"] or "",
                }
            )

    summary_path = output_dir / "variance_decomposition_summary.txt"
    with summary_path.open("a") as f:
        f.write(f"\n=== {atlas} / {metric} ===\n")
        for stratum in _STRATUM_ORDER:
            s = summaries[stratum]
            if not s["available"]:
                f.write(f"  {stratum}: not available ({s['reason']})\n")
                continue
            flag = " [low confidence]" if s["low_confidence"] else ""
            f.write(
                f"  {stratum}: n={s['n']} mean={s['mean']:.4f} median={s['median']:.4f} "
                f"IQR=[{s['iqr_low']:.4f}, {s['iqr_high']:.4f}]{flag}\n"
            )
        f.write(f"  {headline_text(atlas, metric, ratios)}\n")


def run(sweep_optimize_dir: Path, output_dir: Path) -> dict[tuple[str, str], dict]:
    grouped = collect_sweep_matrices(sweep_optimize_dir)
    results: dict[tuple[str, str], dict] = {}
    for atlas, metric in sorted(grouped):
        combo_matrices = grouped[(atlas, metric)]
        strata = compute_strata(combo_matrices)
        summaries = {name: summarize_stratum(entry) for name, entry in strata.items()}
        ratios = compute_ratios(summaries)
        write_decomposition(output_dir, atlas, metric, summaries, ratios)
        results[(atlas, metric)] = {"summaries": summaries, "ratios": ratios}
        logging.info(headline_text(atlas, metric, ratios))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Variance decomposition across a completed OptiConn sweep"
    )
    parser.add_argument(
        "sweep_optimize_dir", help="Path to a sweep's optimize/ directory"
    )
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/local/software/opticonn && source braingraph_pipeline/bin/activate && python3 -m pytest tests/test_variance_decomposition.py -v`
Expected: PASS, all tests in the file.

- [ ] **Step 5: Run the full test suite**

Run: `cd /data/local/software/opticonn && source braingraph_pipeline/bin/activate && python3 -m pytest tests/ -q`
Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
cd /data/local/software/opticonn
git add scripts/variance_decomposition.py tests/test_variance_decomposition.py
git commit -m "$(cat <<'EOF'
feat: variance_decomposition CSV/summary output, orchestration, and CLI

write_decomposition appends per-stratum rows to variance_decomposition.csv
(one header, written once) and a human-readable block to
variance_decomposition_summary.txt. run() orchestrates collection through
writing for every (atlas, metric) a sweep evaluated and returns the results
for programmatic use. CLI:
  python scripts/variance_decomposition.py <sweep>/optimize -o <output_dir>

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Log the headline at the end of a sweep

**Files:**
- Modify: `scripts/cross_validation_bootstrap_optimizer.py:1384-1400` (final summary section)
- Test: `tests/test_variance_decomposition.py`

**Interfaces:**
- Consumes: `scripts.variance_decomposition.run(sweep_optimize_dir: Path, output_dir: Path) -> dict` (Task 6).

Per the Global Constraints, this call must never be able to fail the sweep: wrap it in `try/except Exception`, log a warning on failure, and continue. This mirrors the existing pattern in `scripts/opticonn_hub.py` around the post-sweep quick-quality-check subprocess call, which is similarly optional and non-fatal.

- [ ] **Step 1: Write the failing test**

This exercises the integration point directly (not by running a full sweep, which needs DSI Studio) by calling the same code path the hook will call, against the fixture from Task 3, and checking the sweep's own `output_dir` layout matches what `variance_decomposition.run` expects. Append to `tests/test_variance_decomposition.py`:

```python
def test_run_accepts_a_sweep_shaped_output_dir_directly(tmp_path):
    """The real call site passes output_dir/"optimize" (see cross_validation_
    bootstrap_optimizer.py's own `output_dir` variable) -- confirm run()
    against that exact shape, not just the "optimize_dir already given" case
    tested in Task 6.
    """
    output_dir = tmp_path  # what cross_validation_bootstrap_optimizer.py calls output_dir
    optimize_dir = output_dir / "optimize"
    build_sweep_fixture(output_dir)  # writes into output_dir / "optimize"
    assert optimize_dir.exists()

    results = run(optimize_dir, optimize_dir / "optimization_results")

    assert ("AAL3", "count") in results
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/local/software/opticonn && source braingraph_pipeline/bin/activate && python3 -m pytest tests/test_variance_decomposition.py -v -k sweep_shaped`
Expected: This test actually passes already, since `run()` doesn't care what the directory is called -- it just needs to exist with the right internal shape. Confirm it PASSES as a check that Task 6's `run()` is directly usable from the real call site's directory layout before wiring in the hook. If it fails, fix `build_sweep_fixture` (in this test file) to write into `tmp_path / "optimize"` rather than assuming `tmp_path` itself is the optimize dir -- check the earlier fixture's structure and adjust the assertion, not `run()`.

- [ ] **Step 3: Add the hook in `cross_validation_bootstrap_optimizer.py`**

Current code (lines 1384-1400):
```python
    # Final summary
    total_duration = time.time() - start_time
    logging.info("\\n" + "=" * 60)

    if wave1_success and wave2_success:
        if args.single_wave:
            logging.info(" COMPREHENSIVE OPTIMIZATION COMPLETED SUCCESSFULLY")
            logging.info(f" Results saved in: {output_dir}")
            logging.info(f"  Total runtime: {total_duration:.1f} seconds")
            # For single wave, copy results directly to optimization_results
            # (No further action required here; results are already saved in output_dir)
        else:
            logging.info(" CROSS-VALIDATION COMPLETED SUCCESSFULLY")
            logging.info(f" Results saved in: {output_dir}")
            logging.info(f"  Total runtime: {total_duration:.1f} seconds")
            logging.info(f"   • Wave 1: {wave1_duration:.1f}s")
            logging.info(f"   • Wave 2: {wave2_duration:.1f}s")
```

Replace with:
```python
    # Final summary
    total_duration = time.time() - start_time
    logging.info("\\n" + "=" * 60)

    if wave1_success and wave2_success:
        if args.single_wave:
            logging.info(" COMPREHENSIVE OPTIMIZATION COMPLETED SUCCESSFULLY")
            logging.info(f" Results saved in: {output_dir}")
            logging.info(f"  Total runtime: {total_duration:.1f} seconds")
            # For single wave, copy results directly to optimization_results
            # (No further action required here; results are already saved in output_dir)
        else:
            logging.info(" CROSS-VALIDATION COMPLETED SUCCESSFULLY")
            logging.info(f" Results saved in: {output_dir}")
            logging.info(f"  Total runtime: {total_duration:.1f} seconds")
            logging.info(f"   • Wave 1: {wave1_duration:.1f}s")
            logging.info(f"   • Wave 2: {wave2_duration:.1f}s")

        try:
            from scripts.variance_decomposition import run as run_variance_decomposition

            run_variance_decomposition(
                Path(output_dir) / "optimize", Path(output_dir) / "optimize" / "optimization_results"
            )
        except Exception as exc:
            logging.warning(f"  Variance decomposition skipped: {exc}")
```

- [ ] **Step 4: Run tests to verify everything still passes**

Run: `cd /data/local/software/opticonn && source braingraph_pipeline/bin/activate && python3 -m pytest tests/ -q`
Expected: PASS, no regressions.

- [ ] **Step 5: Manual check of the hook against real fixture data**

Run:
```bash
cd /data/local/software/opticonn && source braingraph_pipeline/bin/activate && python3 -c "
from pathlib import Path
from scripts.variance_decomposition import run

# reuse an already-produced real sweep from earlier today if present, else skip
sweep = Path('studies/study129/verify_production/optimize')
if sweep.exists():
    results = run(sweep, sweep / 'optimization_results')
    print('atlas/metric pairs found:', list(results.keys()))
else:
    print('no real sweep output present locally; unit tests already cover this path')
"
```
Expected: Either prints found `(atlas, metric)` pairs from a real sweep, or the skip message. Either is fine — this step is a sanity check, not a required assertion, since the fixture-based tests already cover correctness.

- [ ] **Step 6: Commit**

```bash
cd /data/local/software/opticonn
git add scripts/cross_validation_bootstrap_optimizer.py tests/test_variance_decomposition.py
git commit -m "$(cat <<'EOF'
feat: run variance decomposition at the end of a two-wave sweep

Wrapped in try/except so a failure here can never fail the sweep itself --
matches the existing non-fatal pattern for the post-sweep quick-quality-check
subprocess in opticonn_hub.py. Single-wave (--single-wave) runs are left
alone for now since their output_dir layout differs; only the two-wave path
(tune-grid's default) is wired in.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** Component 0 (discovery fix, subject/session parsing) — Tasks 1-2. Component 1 (four strata, distributions not point estimates, ratios, CSV + summary, headline logged at end of sweep) — Tasks 3-7. Component 1's "never wired into rank()" constraint — satisfied by construction: `variance_decomposition.py` never imports from or is imported by `reliability.py`'s `rank`/`rank_with_fallback`/`score_combo`. Component 2 (margin statistic) is explicitly out of scope per the Global Constraints and is a separate future plan.
- **Type consistency:** `collect_sweep_matrices` → `compute_strata` → `summarize_stratum` → `compute_ratios` → `write_decomposition` — checked that each function's output dict shape matches what the next consumes (stratum names, key names inside summary dicts) across all task boundaries.
- **combo_id scoping:** deliberately `"<wave>/<sweep_id>"`, not bare `sweep_id` — documented inline in Task 3's implementation, so `parameter`-stratum comparisons never conflate two different waves' independently-sampled grid candidates that happen to share a `sweep_0001` name.
