# DSI Studio Execution Verification Implementation Plan (Plan A of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every DSI Studio tracking run OptiConn makes is proven to have executed exactly the parameters OptiConn sent; a run that cannot be proven is a failed run whose matrices are deleted.

**Architecture:** A new pure module `scripts/dsi_verify.py` checks three independent records DSI Studio leaves behind — its stdout echo (positive confirmation of every sent flag), the execution `report` in the tract file (each parameter's expected executed statement), and the tract file's `parameter_id` fingerprint. `scripts/extract_connectivity_matrices.py`, the one chokepoint every tracking call passes through, sends only explicit (non-`null`) parameters, runs the checks after every run, writes a per-run execution record, and deletes all outputs of a run that fails. A new `scripts/dsi_preflight.py` proves each specification once on one subject — including streamline geometry and differential output — and writes the expected fingerprints production runs are checked against.

**Tech Stack:** Python 3.10, numpy, scipy (`scipy.io.loadmat`), nibabel (preflight only; already installed, declared by this plan), pytest.

**Plan B** (`opticonn-multiverse-battery`: config, container pin, QC, pinned upstreams, preprocessing moderator) follows this plan and depends on it.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-27-battery-readiness-design.md` §0. Read it first.
- **Mandatory, fail closed.** Anything that cannot be verified is a failure. Never downgrade a verification error to a warning, never add a switch that disables verification.
- **`null` means "DSI Studio's own default"**: the flag is omitted, and the value DSI Studio applied is recorded from its execution report, not compared. Any non-`null` tracking value is always sent and must be executed as sent.
- Pinned DSI Studio build: `/data/local/software/apptainer_images/dsi_studio/dsi_studio_hou-2026-09-27.sif`. The expected report statements are those of this build; the fixtures in `tests/fixtures/dsi_studio_echo/` were captured from it.
- Established facts on this build (do not re-derive; the fixtures prove them): `parameter_id` is identical across subjects for the same parameters and differs with `random_seed`; the report does not distinguish Euler from Runge–Kutta; a `.tt.gz` converted to `.trk` is quantised to ~0.03 mm, a direct `.trk` export is exact; DSI Studio converts length limits to whole steps: floor(min_length/step) ≤ points ≤ floor(max_length/step) (established by the 2026-09-27 preflight, both bounds hit exactly; the earlier "points × step" model coincided only at integral step sizes).
- No new runtime dependency other than declaring `nibabel` (already installed), imported only by the preflight.
- Tests never need DSI Studio or network. Run the suite exactly like this, or two unrelated tests fail spuriously:
  `DSI_STUDIO_PATH=/data/local/software/dsistuido/installation/apptainer/run_dsi_studio.sh OPTICONN_SKIP_VENV=1 /data/local/software/opticonn/braingraph_pipeline/bin/python -m pytest tests/ -q`
- Baseline before this plan: **208 tests passing**.
- Every commit message ends with: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

---

### Task 1: Echo verification — positive confirmation of every sent flag

**Files:**
- Create: `scripts/dsi_verify.py`
- Test: `tests/test_dsi_verify.py`
- Fixtures (exist): `tests/fixtures/dsi_studio_echo/sweep_0001_stdout.txt`, `tests/fixtures/dsi_studio_echo/sweep_0001_command.txt`

**Interfaces:**
- Produces: `parse_command_flags(cmd: list[str]) -> dict[str, str]`, `parse_echo(stdout: str) -> dict[str, str]`, `check_echo(sent: dict[str, str], stdout: str) -> list[str]`, constants `COMPARED: frozenset[str]`, `PATH_FLAGS: frozenset[str]`, private `_same(a, b) -> bool`.

The fixture `sweep_0001_stdout.txt` is DSI Studio's real stdout for the command in `sweep_0001_command.txt`. That command sent `--connectivity_threshold=0.001`, which this build does not recognise: it prints `❗--connectivity_threshold is not used/recognized` and exits 0. Echoed lines look like `│  │  ├──\x1b[0;32mfa_threshold\x1b[0m=0` (ANSI colour codes around the key).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dsi_verify.py`:

```python
from pathlib import Path

import pytest

from scripts import dsi_verify

FIX = Path(__file__).parent / "fixtures" / "dsi_studio_echo"
WARNING = "--connectivity_threshold is not used/recognized"


def _stdout(name="sweep_0001"):
    return (FIX / f"{name}_stdout.txt").read_text()


def _sent(name="sweep_0001"):
    return dsi_verify.parse_command_flags((FIX / f"{name}_command.txt").read_text().split())


def _without_warning(text):
    return "\n".join(line for line in text.splitlines() if WARNING not in line)


def test_parse_command_flags_reads_every_flag():
    sent = _sent()
    assert sent["action"] == "trk"
    assert sent["turning_angle"] == "35"
    assert sent["connectivity_threshold"] == "0.001"
    assert all(not k.startswith("-") for k in sent)


def test_parse_echo_strips_ansi_and_tree_prefixes():
    echo = dsi_verify.parse_echo(_stdout())
    assert echo["action"] == "trk"
    assert echo["fa_threshold"] == "0"
    assert echo["turning_angle"] == "35"
    assert echo["track_voxel_ratio"] == "0.738525"   # derived by DSI Studio, never sent
    assert "connectivity_threshold" not in echo


def test_unrecognised_flag_fails_twice_over():
    errors = dsi_verify.check_echo(_sent(), _stdout())
    assert any("connectivity_threshold" in e and "not used/recognized" in e for e in errors)
    assert any("connectivity_threshold" in e and "not echoed back" in e for e in errors)
    assert all("connectivity_threshold" in e for e in errors), errors


def test_sent_flag_missing_from_echo_fails_even_without_the_warning():
    # A build that stops printing the warning must still be caught.
    errors = dsi_verify.check_echo(_sent(), _without_warning(_stdout()))
    assert errors == [
        "--connectivity_threshold=0.001: sent but not echoed back, so not executed"]


def test_faithful_echo_passes():
    sent = _sent()
    del sent["connectivity_threshold"]
    assert dsi_verify.check_echo(sent, _without_warning(_stdout())) == []


def test_changed_value_fails():
    sent = _sent()
    del sent["connectivity_threshold"]
    sent["turning_angle"] = "40"
    errors = dsi_verify.check_echo(sent, _without_warning(_stdout()))
    assert errors == ["--turning_angle: sent 40, DSI Studio parsed 35"]


def test_numeric_formatting_is_not_a_mismatch():
    sent = _sent()
    del sent["connectivity_threshold"]
    sent["step_size"] = "1"          # echo says 1.0
    assert dsi_verify.check_echo(sent, _without_warning(_stdout())) == []


def test_paths_are_confirmed_present_but_not_compared():
    sent = _sent()
    del sent["connectivity_threshold"]
    sent["source"] = "/somewhere/else.fz"
    assert dsi_verify.check_echo(sent, _without_warning(_stdout())) == []


def test_no_echo_at_all_fails():
    assert dsi_verify.check_echo({"action": "trk"}, "") == [
        "no parameter echo found in DSI Studio output; cannot confirm anything was executed"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `… -m pytest tests/test_dsi_verify.py -q`
Expected: collection error, `ImportError: cannot import name 'dsi_verify'`.

- [ ] **Step 3: Implement**

Create `scripts/dsi_verify.py`:

```python
"""Prove that DSI Studio executed the tracking parameters OptiConn sent.

Mandatory for every tracking run (docs/superpowers/specs/
2026-09-27-battery-readiness-design.md, section 0). DSI Studio accepts unknown
options and exits 0, and resolves some values into strategies (fa_threshold=0 with
otsu_threshold=0.6 executes as a threshold randomised over 0.5-0.7 x Otsu), so an
exit code and an output file prove nothing. Three records are checked, each
failing closed:

1. the stdout echo -- what DSI Studio parsed. Every sent flag must be echoed back;
   a flag that is not is treated as not executed, whatever warning is printed.
2. the tract file's `report` -- DSI Studio's own prose record of what it executed.
   Each sent parameter must map to its expected executed statement.
3. the tract file's `parameter_id` -- an encoded fingerprint of the executed
   parameters (subject-independent, seed-dependent). Must equal the preflight's.

Streamline geometry (`check_geometry`) is a fourth, physical check used by the
preflight. Everything here is pure: nothing runs DSI Studio or writes files.
"""

from __future__ import annotations

import re

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_ECHO = re.compile(r"^[│\s]*(?:[├└]──)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_UNRECOGNIZED = re.compile(r"--([A-Za-z_][A-Za-z0-9_]*) is not used/recognized")
_TOL = 1e-6

# Echoed values that must equal what was sent. Paths and connectivity options come
# back rewritten, so they are only confirmed present.
COMPARED = frozenset({
    "method", "otsu_threshold", "fa_threshold", "turning_angle", "step_size",
    "smoothing", "min_length", "max_length", "check_ending", "tip_iteration",
    "threshold_index", "random_seed", "tract_count", "thread_count",
})
# Flags naming files rather than parameters: excluded from the fingerprint key.
PATH_FLAGS = frozenset({"source", "output", "connectivity"})


def parse_command_flags(cmd: list[str]) -> dict[str, str]:
    """{flag: value} for every `--flag=value` token of a DSI Studio command."""
    flags: dict[str, str] = {}
    for token in map(str, cmd):
        if token.startswith("--") and "=" in token:
            key, value = token[2:].split("=", 1)
            flags[key] = value
    return flags


def parse_echo(stdout: str) -> dict[str, str]:
    """{key: value} DSI Studio echoed as parsed; the first occurrence wins."""
    echo: dict[str, str] = {}
    for line in _ANSI.sub("", stdout).splitlines():
        m = _ECHO.match(line.rstrip())
        if m:
            echo.setdefault(m[1], m[2].strip())
    return echo


def _same(a, b) -> bool:
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()
    return abs(fa - fb) <= _TOL * max(1.0, abs(fa))


def check_echo(sent: dict[str, str], stdout: str) -> list[str]:
    """Errors if a sent flag was reported unrecognised, not echoed back, or changed."""
    echo = parse_echo(stdout)
    if not echo:
        return ["no parameter echo found in DSI Studio output; cannot confirm anything was executed"]
    errors = [f"--{flag}: DSI Studio reports it as not used/recognized"
              for flag in _UNRECOGNIZED.findall(_ANSI.sub("", stdout))]
    for key, value in sent.items():
        if key not in echo:
            errors.append(f"--{key}={value}: sent but not echoed back, so not executed")
        elif key in COMPARED and not _same(value, echo[key]):
            errors.append(f"--{key}: sent {value}, DSI Studio parsed {echo[key]}")
    return errors
```

- [ ] **Step 4: Run to verify they pass**

Run: `… -m pytest tests/test_dsi_verify.py -q`
Expected: 9 passed. If `test_faithful_echo_passes` fails naming a flag, print `sorted(set(_sent()) - set(dsi_verify.parse_echo(_stdout())))` — every flag in the fixture command other than `connectivity_threshold` is echoed on this build, so a difference is a parser bug, not a fixture problem.

- [ ] **Step 5: Commit**

```bash
git add scripts/dsi_verify.py tests/test_dsi_verify.py
git commit -m "$(cat <<'MSG'
feat(verify): every sent DSI Studio flag must be echoed back

Positive confirmation rather than warning detection: a flag DSI Studio does
not echo is treated as not executed, so a build that rewords or drops its
'not used/recognized' warning is still caught.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 2: Execution-report verification

**Files:**
- Modify: `scripts/dsi_verify.py`
- Test: `tests/test_dsi_verify.py`
- Fixtures (exist): `tests/fixtures/dsi_studio_echo/exec_{grid,otsu08,fa01,rk4,tip2,reference}_report.txt`

**Interfaces:**
- Consumes: `_same` from Task 1.
- Produces: `parse_report(report: str) -> dict`, `expected_execution(sent: dict[str, str]) -> dict`, `check_report(sent: dict[str, str], report: str) -> tuple[dict, list[str]]`.

Executed-parameter representation (JSON-serialisable; used by Tasks 3, 5, 6):

```python
{
  "anisotropy":    {"kind": "otsu_window", "low": 0.5, "high": 0.7} | {"kind": "fixed", "value": 0.1},
  "turning_angle": {"kind": "fixed", "value": 35.0} | {"kind": "random", "low": 45.0, "high": 90.0},
  "step_size":     {"kind": "fixed", "value": 1.0} | {"kind": "voxel_spacing"},
  "smoothing": 0.1,            # fraction; 0.0 when DSI Studio reports no smoothing
  "min_length": 10.0, "max_length": 250.0,
  "tract_count": 50000,
  "tip_iteration": 0,          # 0 when DSI Studio reports no pruning
}
```

The report sentences on the pinned build (from the fixtures):
- "The anisotropy threshold was randomly selected between 0.5 and 0.7 otsu threshold." / "The anisotropy threshold was 0.1."
- "The angular threshold was 35 degrees." / "The angular threshold was randomly selected from 45 degrees to 90 degrees."
- "The step size was 1.00 mm." / "The step size was set to voxel spacing."
- "…smoothed by averaging the propagation direction with 10% of the previous direction." (absent when there is no smoothing)
- "Tracks with length shorter than 10.00 or longer than 250.00 mm were discarded." (the reference prints `30.0`/`200.0`)
- "A total of 50000 tracts were tracked."
- "Topology-informed pruning (…) was applied to the tractography with 2 iteration(s) to remove false connections." (absent without pruning; the citation contains a line break)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dsi_verify.py`:

```python
def _report(name):
    return (FIX / f"exec_{name}_report.txt").read_text()


GRID_SENT = {"turning_angle": "35", "step_size": "1.0", "smoothing": "0.1",
             "min_length": "10", "max_length": "250", "tract_count": "50000",
             "otsu_threshold": "0.6", "fa_threshold": "0.0"}


def test_parse_report_grid():
    ex = dsi_verify.parse_report(_report("grid"))
    assert ex["anisotropy"] == {"kind": "otsu_window", "low": 0.5, "high": 0.7}
    assert ex["turning_angle"] == {"kind": "fixed", "value": 35.0}
    assert ex["step_size"] == {"kind": "fixed", "value": 1.0}
    assert ex["smoothing"] == pytest.approx(0.1)
    assert (ex["min_length"], ex["max_length"]) == (10.0, 250.0)
    assert ex["tract_count"] == 50000
    assert ex["tip_iteration"] == 0


def test_parse_report_reference_resolves_defaults():
    ex = dsi_verify.parse_report(_report("reference"))
    assert ex["anisotropy"] == {"kind": "otsu_window", "low": 0.5, "high": 0.7}
    assert ex["turning_angle"] == {"kind": "random", "low": 45.0, "high": 90.0}
    assert ex["step_size"] == {"kind": "voxel_spacing"}
    assert ex["smoothing"] == 0.0
    assert (ex["min_length"], ex["max_length"]) == (30.0, 200.0)


def test_parse_report_fixed_fa_otsu_window_and_pruning():
    assert dsi_verify.parse_report(_report("fa01"))["anisotropy"] == {"kind": "fixed", "value": 0.1}
    assert dsi_verify.parse_report(_report("otsu08"))["anisotropy"] == {
        "kind": "otsu_window", "low": 0.7, "high": 0.9}
    assert dsi_verify.parse_report(_report("tip2"))["tip_iteration"] == 2


def test_grid_report_matches_what_was_sent():
    executed, errors = dsi_verify.check_report(GRID_SENT, _report("grid"))
    assert errors == []
    assert executed["tract_count"] == 50000


def test_otsu_window_is_derived_from_the_sent_centre():
    sent = {**GRID_SENT, "otsu_threshold": "0.8"}
    _, errors = dsi_verify.check_report(sent, _report("grid"))
    assert len(errors) == 1 and errors[0].startswith("anisotropy:")


def test_fixed_fa_threshold_takes_precedence_over_otsu():
    sent = {**GRID_SENT, "fa_threshold": "0.1"}          # otsu 0.6 still sent, inert
    assert dsi_verify.check_report(sent, _report("fa01"))[1] == []


def test_zero_angle_means_the_documented_random_window():
    sent = {"turning_angle": "0"}
    assert dsi_verify.check_report(sent, _report("reference"))[1] == []
    assert dsi_verify.check_report(sent, _report("grid"))[1] != []


def test_zero_step_means_voxel_spacing():
    assert dsi_verify.check_report({"step_size": "0"}, _report("reference"))[1] == []


def test_smoothing_sent_but_not_executed_fails():
    _, errors = dsi_verify.check_report({"smoothing": "0.1"}, _report("reference"))
    assert errors == ["smoothing: expected 0.1, DSI Studio executed 0.0"]


def test_pruning_is_verified():
    assert dsi_verify.check_report({"tip_iteration": "2"}, _report("tip2"))[1] == []
    assert dsi_verify.check_report({"tip_iteration": "2"}, _report("grid"))[1] != []


def test_changed_length_fails():
    _, errors = dsi_verify.check_report({**GRID_SENT, "min_length": "30"}, _report("grid"))
    assert errors == ["min_length: expected 30.0, DSI Studio executed 10.0"]


def test_nothing_sent_means_nothing_compared_but_everything_recorded():
    executed, errors = dsi_verify.check_report({}, _report("reference"))
    assert errors == []
    assert executed["step_size"] == {"kind": "voxel_spacing"}


def test_empty_or_reworded_report_fails_closed():
    _, errors = dsi_verify.check_report(GRID_SENT, "")
    assert {e.split(" for ")[-1] for e in errors} >= {
        "anisotropy", "turning_angle", "step_size", "min_length", "tract_count"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `… -m pytest tests/test_dsi_verify.py -q`
Expected: the new tests fail with `AttributeError: module 'scripts.dsi_verify' has no attribute 'parse_report'`; Task 1's still pass.

- [ ] **Step 3: Implement**

Append to `scripts/dsi_verify.py`:

```python
_N = r"([0-9]+(?:\.[0-9]+)?)"
_REPORT = {
    "otsu_window": re.compile(
        rf"anisotropy threshold was randomly selected between {_N} and {_N} otsu threshold"),
    "fa_fixed": re.compile(rf"anisotropy threshold was {_N}\."),
    "angle_random": re.compile(
        rf"angular threshold was randomly selected from {_N} degrees to {_N} degrees"),
    "angle_fixed": re.compile(rf"angular threshold was {_N} degrees"),
    "step_voxel": re.compile(r"step size was set to voxel spacing"),
    "step_fixed": re.compile(rf"step size was {_N} mm"),
    "smoothing": re.compile(rf"propagation direction with {_N}% of the previous direction"),
    "length": re.compile(rf"length shorter than {_N} or longer than {_N} mm were discarded"),
    "tract_count": re.compile(r"A total of ([0-9]+) tracts were tracked"),
    "pruning": re.compile(r"with ([0-9]+) iteration\(s\) to remove false connections"),
}
# A report without these statements cannot prove anything: fail closed.
_MANDATORY = ("anisotropy", "turning_angle", "step_size", "min_length", "tract_count")


def parse_report(report: str) -> dict:
    """What DSI Studio says it executed, from a tract file's `report`.

    A key is present only when its statement was found. Smoothing and pruning are
    reported only when active, so their absence means 0.
    """
    r = " ".join((report or "").split())
    ex: dict = {"smoothing": 0.0, "tip_iteration": 0}
    if m := _REPORT["otsu_window"].search(r):
        ex["anisotropy"] = {"kind": "otsu_window", "low": float(m[1]), "high": float(m[2])}
    elif m := _REPORT["fa_fixed"].search(r):
        ex["anisotropy"] = {"kind": "fixed", "value": float(m[1])}
    if m := _REPORT["angle_random"].search(r):
        ex["turning_angle"] = {"kind": "random", "low": float(m[1]), "high": float(m[2])}
    elif m := _REPORT["angle_fixed"].search(r):
        ex["turning_angle"] = {"kind": "fixed", "value": float(m[1])}
    if _REPORT["step_voxel"].search(r):
        ex["step_size"] = {"kind": "voxel_spacing"}
    elif m := _REPORT["step_fixed"].search(r):
        ex["step_size"] = {"kind": "fixed", "value": float(m[1])}
    if m := _REPORT["smoothing"].search(r):
        ex["smoothing"] = float(m[1]) / 100
    if m := _REPORT["length"].search(r):
        ex["min_length"], ex["max_length"] = float(m[1]), float(m[2])
    if m := _REPORT["tract_count"].search(r):
        ex["tract_count"] = int(m[1])
    if m := _REPORT["pruning"].search(r):
        ex["tip_iteration"] = int(m[1])
    return ex


def expected_execution(sent: dict[str, str]) -> dict:
    """What DSI Studio should execute for the sent flags (pinned build, spec section 0).

    Only sent parameters carry an expectation. Omitted ones (null in the config) are
    DSI Studio's own defaults, recorded as executed rather than compared. The
    otsu window (centre +/- 0.1) and the 45-90 degree random angle are DSI Studio's
    documented behaviour for fa_threshold=0 and turning_angle=0.
    """
    f = lambda k: float(sent[k])
    exp: dict = {}
    if "fa_threshold" in sent and f("fa_threshold") > 0:
        exp["anisotropy"] = {"kind": "fixed", "value": f("fa_threshold")}
    elif "otsu_threshold" in sent:
        t = f("otsu_threshold")
        exp["anisotropy"] = {"kind": "otsu_window", "low": t - 0.1, "high": t + 0.1}
    if "turning_angle" in sent:
        a = f("turning_angle")
        exp["turning_angle"] = ({"kind": "fixed", "value": a} if a > 0
                                else {"kind": "random", "low": 45.0, "high": 90.0})
    if "step_size" in sent:
        s = f("step_size")
        exp["step_size"] = {"kind": "fixed", "value": s} if s > 0 else {"kind": "voxel_spacing"}
    for key in ("smoothing", "min_length", "max_length"):
        if key in sent:
            exp[key] = f(key)
    for key in ("tract_count", "tip_iteration"):
        if key in sent:
            exp[key] = int(float(sent[key]))
    return exp


def _match(want, got) -> bool:
    if isinstance(want, dict) or isinstance(got, dict):
        return (isinstance(want, dict) and isinstance(got, dict) and want.keys() == got.keys()
                and all(_match(want[k], got[k]) for k in want))
    return _same(want, got)


def check_report(sent: dict[str, str], report: str) -> tuple[dict, list[str]]:
    """(executed parameters, errors): missing statements and mismatches are errors."""
    executed = parse_report(report)
    errors = [f"execution report has no statement for {k}" for k in _MANDATORY if k not in executed]
    for key, want in expected_execution(sent).items():
        got = executed.get(key)
        if got is not None and not _match(want, got):
            errors.append(f"{key}: expected {want}, DSI Studio executed {got}")
    return executed, errors
```

Note on `test_empty_or_reworded_report_fails_closed`: an empty report yields `{"smoothing": 0.0, "tip_iteration": 0}`, so the five mandatory keys are missing and each produces "execution report has no statement for <key>". `max_length` shares `min_length`'s sentence, so it is not listed separately.

- [ ] **Step 4: Run to verify they pass**

Run: `… -m pytest tests/test_dsi_verify.py -q`
Expected: 22 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/dsi_verify.py tests/test_dsi_verify.py
git commit -m "$(cat <<'MSG'
feat(verify): check DSI Studio's execution report against what was sent

The echo shows what was parsed; the tract file's report shows what ran --
e.g. otsu_threshold=0.6 executes as a threshold randomised over 0.5-0.7 x
Otsu. Each sent parameter must map to its expected executed statement; a
report missing a mandatory statement fails closed.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 3: Tract record, fingerprint and geometry checks

**Files:**
- Modify: `scripts/dsi_verify.py`
- Test: `tests/test_dsi_verify.py`
- Fixtures (exist): `tests/fixtures/dsi_studio_echo/exec_{grid,rk4,tip2,otsu08,fa01,reference}_parameter_id.txt`

**Interfaces:**
- Consumes: `PATH_FLAGS` (Task 1), the executed-parameter dict (Task 2).
- Produces: `read_tract_record(path: Path) -> dict` returning `{"report": str|None, "parameter_id": str|None, "track_sha256": str|None}`; `fingerprint_key(sent: dict[str, str]) -> str`; `check_fingerprint(sent: dict[str, str], parameter_id: str|None, expected: dict[str, str]) -> list[str]`; `check_geometry(streamlines, executed: dict, voxel_size: float) -> list[str]`; `same_streamlines(a, b, tol_mm: float = 0.05) -> list[str]`.

A DSI Studio `.tt.gz` is a gzip-compressed MATLAB file with `uint8` arrays `report`, `parameter_id` and `track`, readable with `scipy.io.loadmat`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dsi_verify.py`:

```python
import gzip
import io

import numpy as np
import scipy.io


def _pid(name):
    return (FIX / f"exec_{name}_parameter_id.txt").read_text().strip()


def _write_tract(path, report, parameter_id, track=b"\x01\x02\x03"):
    buf = io.BytesIO()
    as_u8 = lambda s: np.frombuffer(s.encode() if isinstance(s, str) else s, dtype=np.uint8)
    scipy.io.savemat(buf, {"report": as_u8(report), "parameter_id": as_u8(parameter_id),
                           "track": as_u8(track)})
    path.write_bytes(gzip.compress(buf.getvalue()))


def test_read_tract_record_roundtrip(tmp_path):
    p = tmp_path / "x.tt.gz"
    _write_tract(p, _report("grid"), _pid("grid"))
    rec = dsi_verify.read_tract_record(p)
    assert rec["parameter_id"] == _pid("grid")
    assert "angular threshold was 35 degrees" in rec["report"]
    assert len(rec["track_sha256"]) == 64


def test_fingerprints_differ_for_every_varied_parameter():
    ids = {n: _pid(n) for n in ["grid", "rk4", "tip2", "otsu08", "fa01", "reference"]}
    assert len(set(ids.values())) == len(ids), ids


def test_fingerprint_key_ignores_paths_and_order():
    a = {"source": "/a.fz", "output": "/o.tt.gz", "connectivity": "/atlas.nii.gz",
         "turning_angle": "35", "random_seed": "1"}
    b = {"random_seed": "1", "turning_angle": "35", "source": "/b.fz",
         "output": "/p.tt.gz", "connectivity": "/other.nii.gz"}
    assert dsi_verify.fingerprint_key(a) == dsi_verify.fingerprint_key(b)
    assert dsi_verify.fingerprint_key({**a, "random_seed": "2"}) != dsi_verify.fingerprint_key(a)


def test_check_fingerprint():
    sent = {"turning_angle": "35", "random_seed": "1"}
    key = dsi_verify.fingerprint_key(sent)
    assert dsi_verify.check_fingerprint(sent, _pid("grid"), {key: _pid("grid")}) == []
    assert dsi_verify.check_fingerprint(sent, _pid("rk4"), {key: _pid("grid")})[0].startswith(
        "parameter_id")
    assert "run the preflight" in dsi_verify.check_fingerprint(sent, _pid("grid"), {})[0]


# Geometry: synthetic streamlines in mm, executed parameters as parsed from a report.
EXEC = {"step_size": {"kind": "fixed", "value": 1.0},
        "turning_angle": {"kind": "fixed", "value": 35.0},
        "min_length": 10.0, "max_length": 250.0, "tract_count": 2}


def _line(n_points, step=1.0):
    return np.column_stack([np.arange(n_points) * step, np.zeros(n_points), np.zeros(n_points)])


def test_geometry_passes_on_compliant_streamlines():
    assert dsi_verify.check_geometry([_line(10), _line(40)], EXEC, voxel_size=1.7) == []


def test_geometry_uses_dsi_studio_length_convention():
    # 10 points at 1 mm: segment sum 9 mm, DSI Studio counts 10 mm -> kept at min 10.
    assert dsi_verify.check_geometry([_line(10), _line(11)], EXEC, voxel_size=1.7) == []
    assert dsi_verify.check_geometry([_line(9), _line(11)], EXEC, voxel_size=1.7) != []


def test_geometry_catches_wrong_step_count_and_turn():
    assert any("step" in e for e in dsi_verify.check_geometry(
        [_line(20, step=0.5), _line(20)], EXEC, voxel_size=1.7))
    assert any("streamlines" in e for e in dsi_verify.check_geometry(
        [_line(20)], EXEC, voxel_size=1.7))
    turned = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0],
                       [2 + np.cos(np.radians(40)), np.sin(np.radians(40)), 0]])
    turned = np.vstack([turned, turned[-1] + np.arange(1, 10)[:, None] * [np.cos(np.radians(40)),
                                                                         np.sin(np.radians(40)), 0]])
    assert any("turn" in e for e in dsi_verify.check_geometry(
        [turned, _line(20)], EXEC, voxel_size=1.7))


def test_geometry_voxel_spacing_step_and_random_angle():
    ex = {**EXEC, "step_size": {"kind": "voxel_spacing"},
          "turning_angle": {"kind": "random", "low": 45.0, "high": 90.0},
          "min_length": 30.0, "max_length": 200.0}
    assert dsi_verify.check_geometry([_line(30, 1.7), _line(40, 1.7)], ex, voxel_size=1.7) == []


def test_same_streamlines_tolerates_quantisation_only():
    a = [_line(10), _line(20)]
    assert dsi_verify.same_streamlines(a, [s + 0.02 for s in a]) == []
    assert dsi_verify.same_streamlines(a, [s + 0.2 for s in a]) != []
    assert dsi_verify.same_streamlines(a, a[:1]) != []
```

- [ ] **Step 2: Run to verify they fail**

Run: `… -m pytest tests/test_dsi_verify.py -q`
Expected: the new tests fail with `AttributeError … 'read_tract_record'`.

- [ ] **Step 3: Implement**

Add to the imports of `scripts/dsi_verify.py`:

```python
import gzip
import hashlib
import io
from pathlib import Path

import numpy as np
import scipy.io
```

Append:

```python
def read_tract_record(path: Path) -> dict:
    """`report`, `parameter_id` and a digest of the streamline data from a .tt.gz."""
    m = scipy.io.loadmat(io.BytesIO(gzip.decompress(Path(path).read_bytes())))

    def text(key):
        if key not in m:
            return None
        return bytes(np.asarray(m[key]).ravel().astype(np.uint8)).decode("utf-8", "replace").strip()

    track = m.get("track")
    return {
        "report": text("report"),
        "parameter_id": text("parameter_id"),
        "track_sha256": (hashlib.sha256(np.asarray(track).tobytes()).hexdigest()
                         if track is not None else None),
    }


def fingerprint_key(sent: dict[str, str]) -> str:
    """Identity of a specification-and-repeat: every sent flag except file paths."""
    return " ".join(f"--{k}={sent[k]}" for k in sorted(sent) if k not in PATH_FLAGS)


def check_fingerprint(sent: dict[str, str], parameter_id: str | None,
                      expected: dict[str, str]) -> list[str]:
    """The run's parameter_id must equal the one the preflight recorded for it."""
    key = fingerprint_key(sent)
    if key not in expected:
        return [f"no preflight fingerprint for this specification ({key}); run the preflight"]
    if expected[key] != parameter_id:
        return [f"parameter_id {parameter_id} differs from the preflight's {expected[key]}"]
    return []


def _max_turn(s: np.ndarray) -> float:
    d = np.diff(np.asarray(s, dtype=float), axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return float(np.degrees(np.arccos(np.clip((d[1:] * d[:-1]).sum(1), -1, 1))).max())


def check_geometry(streamlines, executed: dict, voxel_size: float) -> list[str]:
    """Streamlines obey the executed step, turning angle, length bounds and count.

    `streamlines`: (n_i, 3) arrays in mm from a direct .trk export -- a converted
    .tt.gz is quantised (~0.03 mm) and cannot prove an exact step. Length follows
    DSI Studio's convention, points x step (one step more than the segment sum).
    """
    errors = []
    if len(streamlines) != executed.get("tract_count"):
        errors.append(f"{len(streamlines)} streamlines, executed tract_count "
                      f"{executed.get('tract_count')}")
    spec = executed["step_size"]
    step = spec["value"] if spec["kind"] == "fixed" else voxel_size
    segs = [np.linalg.norm(np.diff(np.asarray(s, float), axis=0), axis=1)
            for s in streamlines if len(s) > 1]
    if segs:
        seg = np.concatenate(segs)
        if np.abs(seg - step).max() > 1e-3:
            errors.append(f"step lengths {seg.min():.4f}-{seg.max():.4f} mm, executed {step} mm")
    angle = executed["turning_angle"]
    limit = angle["value"] if angle["kind"] == "fixed" else angle["high"]
    worst = max((_max_turn(s) for s in streamlines if len(s) > 2), default=0.0)
    if worst > limit + 1e-3:
        errors.append(f"turn of {worst:.3f} deg exceeds the executed limit of {limit} deg")
    if len(streamlines):
        lengths = np.array([len(s) * step for s in streamlines])
        lo, hi = executed["min_length"], executed["max_length"]
        if lengths.min() < lo - 1e-3 or lengths.max() > hi + 1e-3:
            errors.append(f"lengths {lengths.min():.2f}-{lengths.max():.2f} mm outside "
                          f"the executed {lo}-{hi} mm")
    return errors


def same_streamlines(a, b, tol_mm: float = 0.05) -> list[str]:
    """Two exports of one tracking run agree within the .tt.gz quantisation."""
    if len(a) != len(b):
        return [f"{len(a)} vs {len(b)} streamlines"]
    for i, (x, y) in enumerate(zip(a, b)):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape or np.abs(x - y).max() > tol_mm:
            return [f"streamline {i} differs between the direct export and the verified tract file"]
    return []
```

- [ ] **Step 4: Run to verify they pass**

Run: `… -m pytest tests/test_dsi_verify.py -q`
Expected: 31 passed (22 from Tasks 1-2 plus 9 new), zero failures.

- [ ] **Step 5: Commit**

```bash
git add scripts/dsi_verify.py tests/test_dsi_verify.py
git commit -m "$(cat <<'MSG'
feat(verify): tract record, parameter_id fingerprint and geometry checks

parameter_id changes with every parameter varied -- including the algorithm,
which the prose report cannot show -- and is identical across subjects, so one
preflight subject can vouch for every production run. Geometry proves step,
turn, length and count from the streamlines themselves.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 4: Explicit intent — the extractor sends exactly the non-null parameters

**Files:**
- Modify: `scripts/extract_connectivity_matrices.py` — `DEFAULT_CONFIG` (lines 86-107), `__init__` (line ~151), `validate_configuration` (lines 472-484), `create_output_structure` (lines 575-589), the command builder inside `extract_connectivity_matrix` (lines 650-704), CLI `--track_voxel_ratio` (argument definition and lines 2153-2154)
- Modify: `scripts/json_validator.py` — `validate_param_value` and the `min_length`/`max_length` check (lines ~290-362)
- Modify: `scripts/sweep_utils.py` — `build_param_grid_from_config`
- Modify: every `configs/*.json` except `dsi_studio_config_schema.json` (22 files use a rejected key)
- Test: `tests/test_extract_command.py` (create)

**Interfaces:**
- Consumes: nothing from Tasks 1-3.
- Produces: module-level `build_track_command(config: dict, dsi_cmd: str, source: str, output: str, atlas: str) -> list[str]`; `tracking_config_errors(config: dict) -> list[str]`; `REJECTED_KEYS: dict[str, str]`; `DEFAULT_CONFIG["tracking_parameters"]` whose every value is `None`. Task 5 calls `build_track_command`; Task 6 calls it for the `.trk` export.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_extract_command.py`:

```python
import pytest

from scripts import extract_connectivity_matrices as ecm
from scripts.extract_connectivity_matrices import (
    ConnectivityExtractor, build_track_command, tracking_config_errors)


def _cfg(**tracking):
    return {"tract_count": 50000, "thread_count": 1, "connectivity_values": ["count", "qa"],
            "tracking_parameters": tracking}


def _flags(cmd):
    return {t.split("=", 1)[0] for t in cmd if t.startswith("--")}


def test_null_is_omitted_and_every_literal_is_sent_even_zero():
    cmd = build_track_command(_cfg(turning_angle=0, step_size=None, fa_threshold=0.0,
                                   otsu_threshold=0.6, random_seed=1),
                              "dsi", "s.fz", "o.tt.gz", "AAL3")
    assert "--turning_angle=0" in cmd
    assert "--fa_threshold=0.0" in cmd
    assert "--otsu_threshold=0.6" in cmd       # formerly dropped as an 'assumed default'
    assert "--step_size" not in _flags(cmd)


def test_all_null_sends_no_tracking_flag():
    cmd = build_track_command(_cfg(**dict.fromkeys(ecm.DEFAULT_CONFIG["tracking_parameters"])),
                              "dsi", "s.fz", "o.tt.gz", "AAL3")
    tracking = set(ecm.DEFAULT_CONFIG["tracking_parameters"])
    assert not {f[2:] for f in _flags(cmd)} & tracking


def test_connectivity_threshold_is_never_sent():
    cfg = {**_cfg(), "connectivity_options": {"connectivity_type": "pass"}}
    assert "--connectivity_threshold" not in _flags(
        build_track_command(cfg, "dsi", "s.fz", "o.tt.gz", "AAL3"))


def test_defaults_carry_no_assumed_values():
    assert all(v is None for v in ecm.DEFAULT_CONFIG["tracking_parameters"].values())
    assert "connectivity_threshold" not in ecm.DEFAULT_CONFIG["connectivity_options"]


@pytest.mark.parametrize("key", ["track_voxel_ratio", "dt_threshold"])
def test_parameters_dsi_studio_would_not_apply_are_errors(key):
    errors = tracking_config_errors(_cfg(**{key: 1.0}))
    assert len(errors) == 1 and key in errors[0]


def test_unknown_tracking_key_is_an_error():
    assert "unknown" in tracking_config_errors(_cfg(fa_treshold=0.1))[0]


def test_connectivity_threshold_is_an_error():
    cfg = {**_cfg(), "connectivity_options": {"connectivity_threshold": 0.001}}
    assert "connectivity_threshold" in tracking_config_errors(cfg)[0]


def test_extractor_refuses_a_config_it_cannot_execute_faithfully():
    with pytest.raises(ValueError, match="dt_threshold"):
        ConnectivityExtractor(_cfg(dt_threshold=0.2))


def test_null_tracking_values_do_not_crash_validation_or_naming(tmp_path):
    ex = ConnectivityExtractor(_cfg(**dict.fromkeys(ecm.DEFAULT_CONFIG["tracking_parameters"])))
    run_dir = ex.create_output_structure(str(tmp_path), "sub01")
    assert run_dir.name == "tracks_50k_streamline"


def test_json_validator_accepts_null_tracking_values():
    # The reference combo's derived config reaches this validator via
    # run_pipeline.load_test_configuration; before the fix, None < 0.0 raised TypeError.
    from scripts.json_validator import JSONValidator
    cfg = {"atlases": ["AAL3"], "connectivity_values": ["count"], "tract_count": 50000,
           "thread_count": 1,
           "tracking_parameters": {"fa_threshold": None, "turning_angle": None,
                                   "otsu_threshold": None, "min_length": None,
                                   "max_length": None}}
    errors = JSONValidator()._validate_dsi_studio_config(cfg, dry_run=True)
    assert not [e for e in errors if any(k in e for k in (
        "fa_threshold", "turning_angle", "otsu_threshold", "min_length", "max_length"))], errors


@pytest.mark.parametrize("axis", ["connectivity_threshold_range", "track_voxel_ratio_range",
                                  "dt_threshold_range"])
def test_sweeping_an_unapplied_parameter_is_refused(axis):
    from scripts.sweep_utils import build_param_grid_from_config
    with pytest.raises(ValueError, match=axis):
        build_param_grid_from_config({"sweep_parameters": {axis: [1, 2]}})


# The schema file is not valid JSON (pre-existing; out of scope) and is not a config.
CONFIGS = sorted(p for p in (Path(__file__).parent.parent / "configs").glob("*.json")
                 if p.name != "dsi_studio_config_schema.json")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_shipped_config_is_executable_as_written(path):
    from scripts.sweep_utils import build_param_grid_from_config
    cfg = json.loads(path.read_text())
    assert tracking_config_errors(cfg) == []
    build_param_grid_from_config(cfg)   # raises on a sweep axis DSI Studio would not apply
```

Add `import json` and `from pathlib import Path` to the top of `tests/test_extract_command.py`.

- [ ] **Step 2: Run to verify they fail**

Run: `… -m pytest tests/test_extract_command.py -q`
Expected: collection error, `ImportError: cannot import name 'build_track_command'`.

- [ ] **Step 3: Implement**

**3a.** In `scripts/extract_connectivity_matrices.py`, replace the `"tracking_parameters"` and `"connectivity_options"` blocks of `DEFAULT_CONFIG` with:

```python
    # null means "DSI Studio's own default": the flag is omitted and the value DSI
    # Studio actually applied is recorded from its execution report. Any other value
    # is always sent and must be executed as sent (scripts/dsi_verify.py). There is
    # no table of assumed defaults: DSI Studio is a rolling release and its defaults
    # are template-dependent, so assuming them lets a change pass silently.
    "tracking_parameters": {
        "method": None,           # 0 Euler, 1 Runge-Kutta
        "otsu_threshold": None,   # centre of the Otsu window when fa_threshold is 0
        "fa_threshold": None,     # 0: Otsu window (centre +/- 0.1); > 0: fixed
        "turning_angle": None,    # degrees; 0: randomised 45-90
        "step_size": None,        # mm; 0: voxel spacing
        "smoothing": None,        # fraction of the previous direction
        "min_length": None,       # mm
        "max_length": None,       # mm
        "check_ending": None,
        "tip_iteration": None,    # topology-informed pruning iterations
        "threshold_index": None,
        "random_seed": None,
    },
    "connectivity_options": {
        "connectivity_type": "pass",  # 'pass' or 'end'
        "connectivity_output": "matrix,connectogram,measure",
    },
```

**3b.** Directly after `DEFAULT_CONFIG` (before `_deep_merge_dict`), add:

```python
# Settings DSI Studio would not execute as written. Each is a configuration error,
# never silently dropped.
REJECTED_KEYS = {
    "track_voxel_ratio": "OptiConn always fixes tract_count, and DSI Studio then "
                         "derives the track/voxel ratio itself",
    "dt_threshold": "differential tractography is not supported "
                    "(DSI Studio needs dt_metric1/dt_metric2 for it)",
}


def tracking_config_errors(config: dict) -> list[str]:
    """Settings in `config` that DSI Studio would not execute as written."""
    errors = []
    for key in config.get("tracking_parameters") or {}:
        if key in REJECTED_KEYS:
            errors.append(f"tracking_parameters.{key}: not applied -- {REJECTED_KEYS[key]}")
        elif key not in DEFAULT_CONFIG["tracking_parameters"]:
            errors.append(f"tracking_parameters.{key}: unknown to OptiConn, so it would "
                          "never reach DSI Studio")
    if "connectivity_threshold" in (config.get("connectivity_options") or {}):
        errors.append("connectivity_options.connectivity_threshold: not a DSI Studio option "
                      "(it is ignored as 'not used/recognized'); matrices are unthresholded")
    return errors


def build_track_command(config: dict, dsi_cmd: str, source: str, output: str,
                        atlas: str) -> list[str]:
    """DSI Studio tracking command: a tracking parameter is sent iff it is not None."""
    conn = {**DEFAULT_CONFIG["connectivity_options"],
            **(config.get("connectivity_options") or {})}
    tract_count = config.get("tract_count", config.get("track_count", 100000))
    cmd = [
        dsi_cmd,
        "--action=trk",
        f"--source={source}",
        f"--tract_count={tract_count}",
        f"--connectivity={atlas}",
        f"--connectivity_value={','.join(config['connectivity_values'])}",
        f"--connectivity_type={conn['connectivity_type']}",
        f"--connectivity_output={conn['connectivity_output']}",
        f"--thread_count={config['thread_count']}",
        f"--output={output}",
        "--export=stat",
    ]
    for key, value in (config.get("tracking_parameters") or {}).items():
        if value is not None:
            cmd.append(f"--{key}={value}")
    return cmd
```

**3c.** In `ConnectivityExtractor.__init__`, directly after `self.config = _deep_merge_dict(DEFAULT_CONFIG, config or {})`:

```python
        errors = tracking_config_errors(self.config)
        if errors:
            raise ValueError("configuration cannot be executed faithfully by DSI Studio: "
                             + "; ".join(errors))
```

**3d.** In `validate_configuration`, replace the FA-threshold and turning-angle checks with:

```python
        # Check FA threshold (None = DSI Studio's default, nothing to range-check)
        fa_threshold = tracking_params.get("fa_threshold")
        if fa_threshold is not None and not 0 <= fa_threshold <= 1:
            validation_result["warnings"].append(
                f"FA threshold {fa_threshold} outside normal range [0-1]"
            )

        # Check turning angle
        turning_angle = tracking_params.get("turning_angle")
        if turning_angle is not None and turning_angle > 180:
            validation_result["warnings"].append(
                f"Turning angle {turning_angle}° seems too large"
            )
```

**3e.** In `create_output_structure`, replace the method lookup and the two `param_dir +=` conditions with:

```python
        method_name = {0: "streamline", 1: "rk4", 2: "voxel"}.get(
            tracking_params.get("method") or 0, "streamline"
        )
```
and
```python
        if tracking_params.get("turning_angle"):   # None and 0 both mean "not fixed"
            param_dir += f"_angle{int(tracking_params['turning_angle'])}"
        if tracking_params.get("fa_threshold"):
            param_dir += f"_fa{tracking_params['fa_threshold']:.2f}"
```

**3f.** In `extract_connectivity_matrix`, delete the whole block from `# Resolve connectivity option defaults safely` through the last `cmd.append(f"--random_seed=...")` line (the `_conn_opts` dict, the inline `cmd = [...]` list and every `if tracking_params.get(...)` append), and put in its place:

```python
        cmd = build_track_command(self.config, dsi_cmd_arg, source_arg, output_arg, atlas_arg)
```

Keep the existing `if self.debug_dsi:` logging line that follows.

**3g.** Delete the CLI argument `--track_voxel_ratio` (its `add_argument` call) and the two lines in `main()` that copy `args.track_voxel_ratio` into `tracking_params`.

**3h.** In `scripts/json_validator.py`, make `validate_param_value` return early on `None`, as the first statement of its body:

```python
                if value is None:          # null = DSI Studio's default; nothing to check
                    return []
```

and replace the length check with:

```python
            if (params.get("min_length") is not None and params.get("max_length") is not None
                    and params["min_length"] >= params["max_length"]):
                errors.append("min_length must be less than max_length")
```

**3i.** In `scripts/sweep_utils.py`, at the top of `build_param_grid_from_config` (after `sp = cfg.get("sweep_parameters", {}) or {}`), add:

```python
    for axis in ("connectivity_threshold_range", "track_voxel_ratio_range", "dt_threshold_range"):
        if sp.get(axis) is not None:
            raise ValueError(
                f"sweep_parameters.{axis}: DSI Studio would not apply this parameter as "
                "specified, so sweeping it enumerates identical runs "
                "(see extract_connectivity_matrices.REJECTED_KEYS)")
```

and delete the `add(...)` calls for `track_voxel_ratio`, `dt_threshold` and `connectivity_threshold`.

**3j.** Clean the shipped configs. 22 files in `configs/` set `track_voxel_ratio`, `dt_threshold` or `connectivity_threshold`, or sweep one of them; after this task every one of them would refuse to start. Run this one-off from the repo root (do not commit the script):

```bash
OPTICONN_SKIP_VENV=1 /data/local/software/opticonn/braingraph_pipeline/bin/python - <<'EOF'
import json
from pathlib import Path
DROP_TRACKING = ("track_voxel_ratio", "dt_threshold")
DROP_SWEEP = ("track_voxel_ratio_range", "dt_threshold_range", "connectivity_threshold_range")
for p in sorted(Path("configs").glob("*.json")):
    if p.name == "dsi_studio_config_schema.json":
        continue
    c = json.loads(p.read_text())
    tp = c.get("tracking_parameters") or {}
    for k in DROP_TRACKING:
        tp.pop(k, None)
    if "comment" in tp:   # would otherwise be sent to DSI Studio as --comment=...
        c["tracking_parameters_comment"] = tp.pop("comment")
    (c.get("connectivity_options") or {}).pop("connectivity_threshold", None)
    c.pop("connectivity_threshold", None)
    sp = c.get("sweep_parameters") or {}
    for k in DROP_SWEEP:
        sp.pop(k, None)
    p.write_text(json.dumps(c, indent=2) + "\n")
    print("cleaned", p)
EOF
```

Removing a sweep axis shrinks that config's grid. That is correct -- the axis never changed what DSI Studio executed -- and belongs in the commit message.

- [ ] **Step 4: Run the new tests, then the full suite**

Run: `… -m pytest tests/test_extract_command.py -q` — expected: all pass, including one `test_every_shipped_config_is_executable_as_written` case per config.

Run the full suite. `tests/test_extract_connectivity_matrix_success_path.py` is expected to still pass at this point (verification is wired in Task 5). Any other failure is a test that encoded the old contract — a test asserting an assumed default is omitted, or that `connectivity_threshold`/`track_voxel_ratio` is accepted. Update such a test to assert the new contract (the value is sent explicitly / the key is rejected); never delete an assertion without replacing it with the new-contract equivalent, and list every test you changed, with the reason, in your report.

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_connectivity_matrices.py scripts/json_validator.py scripts/sweep_utils.py configs/ tests/
git commit -m "$(cat <<'MSG'
feat(extract): send exactly the non-null parameters, reject unapplied ones

Deletes the table of assumed DSI Studio defaults: null now means DSI Studio's
own default and is omitted, any other value is always sent. Parameters DSI
Studio would not apply as written -- track_voxel_ratio with a fixed
tract_count, dt_threshold without dt metrics, connectivity_threshold (not an
option) -- are configuration errors instead of silent no-ops. The 22 shipped configs
that set or swept them are cleaned; their grids shrink by axes that never
changed what DSI Studio executed.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 5: Verify every run and fail closed

**Files:**
- Modify: `scripts/extract_connectivity_matrices.py` — `extract_connectivity_matrix`, from `result = subprocess.run(` to the end of the success/failure handling (lines ~709-784 before Task 4)
- Delete: `tests/test_extract_connectivity_matrix_success_path.py` (superseded; its regression is covered by `test_verified_run_succeeds_and_is_recorded` below — move its module docstring into the new test's docstring)
- Test: `tests/test_extract_verified_run.py` (create)

**Interfaces:**
- Consumes: `dsi_verify.parse_command_flags`, `check_echo`, `check_report`, `read_tract_record`, `check_fingerprint` (Tasks 1-3); `build_track_command` (Task 4).
- Produces: per run, `<prefix>.dsi_command.txt` and `<prefix>.dsi_execution.json` in the atlas results directory, where `<prefix>` is `{base_name}_{atlas}`. The JSON has keys `verified: bool`, `errors: list[str]`, `sent: dict`, `echo: dict`, `executed: dict`, `parameter_id: str|None`, `track_sha256: str|None`, `report: str|None`, `returncode: int`. Config key `verification.expected_fingerprints` (path to a preflight JSON) enables the fingerprint check; config key `verification.keep_tract` (bool) keeps the `.tt.gz` for the preflight. The result dict gains `verification_errors: list[str]`. Task 6 relies on all of these.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_extract_verified_run.py`:

```python
"""A tracking run counts only if DSI Studio provably executed what was sent.

Supersedes test_extract_connectivity_matrix_success_path.py, which guarded a dead
call left in the success branch by an incomplete rename (commit 205a1e5): the
success path must still return without raising.
"""

import gzip
import io
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.io

from scripts.extract_connectivity_matrices import ConnectivityExtractor

FIX = Path(__file__).parent / "fixtures" / "dsi_studio_echo"
WARNING = "--connectivity_threshold is not used/recognized"
# The configuration whose DSI Studio run produced the sweep_0001 / exec_grid fixtures.
CONFIG = {
    "tract_count": 50000, "thread_count": 1, "connectivity_values": ["count", "qa"],
    "tracking_parameters": {"turning_angle": 35, "step_size": 1.0, "smoothing": 0.1,
                            "min_length": 10, "max_length": 250, "random_seed": 1},
}


def _clean_stdout():
    return "\n".join(l for l in (FIX / "sweep_0001_stdout.txt").read_text().splitlines()
                     if WARNING not in l)


def _fake_dsi(stdout, report_name="grid", write_tract=True):
    """subprocess.run stand-in: writes what DSI Studio would, returns its stdout."""
    def run(cmd, **_):
        out = Path(next(t.split("=", 1)[1] for t in cmd if t.startswith("--output=")))
        atlas = next(t.split("=", 1)[1] for t in cmd if t.startswith("--connectivity="))
        if write_tract:
            u8 = lambda s: np.frombuffer(s.encode(), dtype=np.uint8)
            buf = io.BytesIO()
            scipy.io.savemat(buf, {
                "report": u8((FIX / f"exec_{report_name}_report.txt").read_text()),
                "parameter_id": u8((FIX / f"exec_{report_name}_parameter_id.txt").read_text()),
                "track": u8("streamlines")})
            out.write_bytes(gzip.compress(buf.getvalue()))
        scipy.io.savemat(str(out) + f".{atlas}.connectivity.mat",
                         {"number of tracts r2r": np.ones((3, 3))})
        (out.parent / (out.name + f".{atlas}.count.connectogram.txt")).write_text("x")
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
    return run


def _run(tmp_path, config=CONFIG, **fake):
    ex = ConnectivityExtractor(config)
    with patch("scripts.extract_connectivity_matrices.subprocess.run", side_effect=_fake_dsi(**fake)):
        result = ex.extract_connectivity_matrix("sub01.qsdr.fz", tmp_path, "AAL3", "sub01")
    atlas_dir = tmp_path / "results" / "AAL3"
    record = json.loads((atlas_dir / "sub01_AAL3.dsi_execution.json").read_text())
    return result, atlas_dir, record


def test_verified_run_succeeds_and_is_recorded(tmp_path):
    result, atlas_dir, record = _run(tmp_path, stdout=_clean_stdout())
    assert result["success"] is True
    assert result["verification_errors"] == []
    assert record["verified"] is True
    assert record["executed"]["turning_angle"] == {"kind": "fixed", "value": 35.0}
    assert record["parameter_id"] == (FIX / "exec_grid_parameter_id.txt").read_text().strip()
    assert "--turning_angle=35" in (atlas_dir / "sub01_AAL3.dsi_command.txt").read_text()
    assert list(atlas_dir.glob("*.connectivity.mat"))
    assert not list(atlas_dir.glob("*.tt.gz")), "tract file is deleted once verified"


def _assert_failed_closed(result, atlas_dir, record, fragment):
    assert result["success"] is False
    assert any(fragment in e for e in result["verification_errors"]), result["verification_errors"]
    assert record["verified"] is False
    left = sorted(p.name for p in atlas_dir.iterdir())
    assert left == ["sub01_AAL3.dsi_command.txt", "sub01_AAL3.dsi_execution.json"], left


def test_unrecognised_flag_fails_closed(tmp_path):
    _assert_failed_closed(*_run(tmp_path, stdout=(FIX / "sweep_0001_stdout.txt").read_text()),
                          "connectivity_threshold")


def test_report_contradicting_what_was_sent_fails_closed(tmp_path):
    cfg = {**CONFIG, "tracking_parameters": {**CONFIG["tracking_parameters"],
                                             "otsu_threshold": 0.6, "fa_threshold": 0.0}}
    _assert_failed_closed(*_run(tmp_path, config=cfg, stdout=_clean_stdout(),
                                report_name="otsu08"), "anisotropy")


def test_missing_tract_file_fails_closed(tmp_path):
    _assert_failed_closed(*_run(tmp_path, stdout=_clean_stdout(), write_tract=False),
                          "tract file")


def test_fingerprint_mismatch_fails_closed(tmp_path):
    pre = tmp_path / "preflight.json"
    pre.write_text(json.dumps({"passed": True, "expected_fingerprints": {}}))
    cfg = {**CONFIG, "verification": {"expected_fingerprints": str(pre)}}
    _assert_failed_closed(*_run(tmp_path, config=cfg, stdout=_clean_stdout()),
                          "preflight fingerprint")


def test_a_failed_preflight_cannot_vouch_for_anything(tmp_path):
    pre = tmp_path / "preflight.json"
    pre.write_text(json.dumps({"passed": False, "expected_fingerprints": {}}))
    cfg = {**CONFIG, "verification": {"expected_fingerprints": str(pre)}}
    _assert_failed_closed(*_run(tmp_path, config=cfg, stdout=_clean_stdout()),
                          "preflight did not pass")


def test_keep_tract_leaves_the_tract_file_for_the_preflight(tmp_path):
    cfg = {**CONFIG, "verification": {"keep_tract": True}}
    _, atlas_dir, _ = _run(tmp_path, config=cfg, stdout=_clean_stdout())
    assert list(atlas_dir.glob("*.tt.gz"))
```

- [ ] **Step 2: Run to verify they fail**

Run: `… -m pytest tests/test_extract_verified_run.py -q`
Expected: failures — `FileNotFoundError` for `sub01_AAL3.dsi_execution.json`.

- [ ] **Step 3: Implement**

At the top of `scripts/extract_connectivity_matrices.py`, add `from scripts import dsi_verify` beside the other imports (check how the file imports sibling modules and follow that pattern; if it uses a `sys.path` fallback for script execution, add the import inside the same guard).

Add this method to `ConnectivityExtractor` (above `extract_connectivity_matrix`):

```python
    def _verify_run(self, cmd: list, result, output_file: Path) -> tuple[dict, list[str]]:
        """Prove DSI Studio executed what `cmd` sent (scripts/dsi_verify.py)."""
        sent = dsi_verify.parse_command_flags(cmd)
        record = dsi_verify.read_tract_record(output_file) if output_file.exists() else {}
        errors = [] if record else [
            "tract file missing, so DSI Studio's execution record is unavailable"]
        errors += dsi_verify.check_echo(sent, result.stdout or "")
        executed, report_errors = dsi_verify.check_report(sent, record.get("report") or "")
        errors += report_errors
        expected_path = (self.config.get("verification") or {}).get("expected_fingerprints")
        if expected_path:
            preflight = json.loads(Path(expected_path).read_text())
            if preflight.get("passed") is not True:
                errors.append(f"preflight did not pass ({expected_path}); it cannot vouch "
                              "for this run")
            else:
                errors += dsi_verify.check_fingerprint(
                    sent, record.get("parameter_id"), preflight["expected_fingerprints"])
        info = {
            "verified": not errors, "errors": errors, "sent": sent,
            "echo": dsi_verify.parse_echo(result.stdout or ""), "executed": executed,
            "parameter_id": record.get("parameter_id"),
            "track_sha256": record.get("track_sha256"), "report": record.get("report"),
            "returncode": result.returncode,
        }
        return info, errors
```

In `extract_connectivity_matrix`, replace everything from `command_success = result.returncode == 0` up to (not including) the `return {` of the success/failure result with:

```python
            command_success = result.returncode == 0
            info, verification_errors = self._verify_run(cmd, result, output_file)
            prefix = output_prefix.name
            (atlas_dir / f"{prefix}.dsi_command.txt").write_text(" ".join(map(str, cmd)) + "\n")
            (atlas_dir / f"{prefix}.dsi_execution.json").write_text(json.dumps(info, indent=2))

            if verification_errors:
                # Fail closed: nothing from an unverifiable run may be scored downstream.
                keep = {f"{prefix}.dsi_command.txt", f"{prefix}.dsi_execution.json"}
                for f in atlas_dir.glob(f"{prefix}*"):
                    if f.name not in keep and f.is_file():
                        f.unlink()
                self.logger.error(f" {atlas}: DSI Studio execution not verified:")
                for e in verification_errors:
                    self.logger.error(f"   - {e}")
                success = False
            else:
                expected_files_created = self._check_connectivity_files_created(
                    output_dir, atlas, base_name)
                success = command_success and expected_files_created
                if not (self.config.get("verification") or {}).get("keep_tract") \
                        and output_file.exists():
                    output_file.unlink()

            if success:
                self.logger.info(f"[Atlas] {atlas} -> verified, done in {duration:.1f}s")
            elif not verification_errors:
                self.logger.error(f" Failed to process {atlas}: return code "
                                  f"{result.returncode}, expected matrices missing")
                if result.stderr:
                    self.logger.error(f"DSI Studio stderr: {result.stderr}")
```

In the `return {...}` that follows, add the key `"verification_errors": verification_errors,`. In the `except subprocess.TimeoutExpired` return, add `"verification_errors": ["timeout: DSI Studio did not finish"],`.

Ensure `json` is imported at the top of the module (add `import json` if absent).

Then delete `tests/test_extract_connectivity_matrix_success_path.py` (its docstring now lives in the new test module).

- [ ] **Step 4: Run the new tests, then the full suite**

Run: `… -m pytest tests/test_extract_verified_run.py -q` — expected: 7 passed.
Run the full suite — expected: all pass. Report the final count.

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_connectivity_matrices.py tests/test_extract_verified_run.py tests/test_extract_connectivity_matrix_success_path.py
git commit -m "$(cat <<'MSG'
feat(extract): verify every DSI Studio run and fail closed

Success used to mean 'exit 0 and the matrix file exists', stdout was
discarded and the tract file -- DSI Studio's only record of what it executed
-- was deleted. Now each run is checked against the echo, the execution
report and (when configured) the preflight fingerprint; a run that cannot be
proven loses every output except its execution record.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 6: Preflight gate

**Files:**
- Create: `scripts/dsi_preflight.py`
- Modify: `pyproject.toml` (add `"nibabel"` to `[project] dependencies`)
- Modify: `docs/cli_steps.md` (short `### Preflight` section)
- Test: `tests/test_dsi_preflight.py` (create)

**Interfaces:**
- Consumes: `build_track_command`, `ConnectivityExtractor` with `verification.keep_tract` (Tasks 4-5); `dsi_verify.read_tract_record`, `fingerprint_key`, `parse_command_flags`, `check_geometry`, `same_streamlines` (Tasks 1-3); from `scripts.cross_validation_bootstrap_optimizer`: `build_combos(sp, candidate_combos) -> (combos, method, reference_index)`, `apply_param_choice_to_config(cfg, choice, mapping)`, `apply_unmapped_params(cfg, choice, mapping)`; from `scripts.sweep_utils`: `build_param_grid_from_config`.
- Produces: `enumerate_specs(config: dict) -> list[dict]` (each `{"spec": int, "repeat": int, "choice": dict, "config": dict}`); `single_axis_pairs(choices: list[dict]) -> list[tuple[int, int, str]]`; `summarize(runs: list[dict], pairs, digests: dict) -> dict`; CLI `python -m scripts.dsi_preflight --config C --subject S --out DIR [--max-parallel N]` writing `DIR/preflight.json` with keys `passed: bool`, `failures: list[str]`, `expected_fingerprints: {fingerprint_key: parameter_id}`, `runs: list`, `determinism: {spec: bool}`, `dsi_apptainer_image: str|None`. Plan B points `verification.expected_fingerprints` at that file.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dsi_preflight.py`:

```python
import pytest

from scripts import dsi_preflight as pf

SWEEP = {
    "tract_count": 50000, "thread_count": 1, "connectivity_values": ["count"],
    "reliability": {"repeats": 2},
    "tracking_parameters": {"step_size": 1.0, "smoothing": 0.1, "max_length": 250},
    "sweep_parameters": {
        "fa_threshold_range": [0.0, 0.1], "otsu_range": [0.6, 0.8],
        "turning_angle_range": [35, 50],
        "reference_candidate": {"fa_threshold": None, "turning_angle": None,
                                "step_size": None, "smoothing": None, "max_length": None},
    },
}


def test_enumerates_every_distinct_spec_and_repeat():
    specs = pf.enumerate_specs(SWEEP)
    # (fa 0/otsu 0.6, fa 0/otsu 0.8, fa 0.1) x 2 angles = 6 grid + 1 reference, x 2 repeats
    assert len(specs) == 14
    assert {s["repeat"] for s in specs} == {1, 2}
    assert all(s["config"]["tracking_parameters"]["random_seed"] == s["repeat"] for s in specs)


def test_reference_spec_carries_nulls_through_to_its_config():
    ref = [s for s in pf.enumerate_specs(SWEEP) if s["choice"].get("turning_angle") is None]
    assert ref and all(s["config"]["tracking_parameters"]["turning_angle"] is None for s in ref)


def test_preflight_requires_single_threaded_determinism():
    with pytest.raises(ValueError, match="thread_count"):
        pf.enumerate_specs({**SWEEP, "thread_count": 4})


def test_single_axis_pairs():
    choices = [{"a": 1, "b": 1}, {"a": 2, "b": 1}, {"a": 1, "b": 2}, {"a": 2, "b": 2}]
    assert sorted(pf.single_axis_pairs(choices)) == [
        (0, 1, "a"), (0, 2, "b"), (1, 3, "b"), (2, 3, "a")]


def _run(spec, repeat, key, pid="PID", errors=()):
    return {"spec": spec, "repeat": repeat, "fingerprint_key": key, "parameter_id": pid,
            "errors": list(errors)}


def test_summary_passes_only_when_everything_passed():
    runs = [_run(0, 1, "k0", "p0"), _run(1, 1, "k1", "p1")]
    s = pf.summarize(runs, [(0, 1, "a")], {0: "d0", 1: "d1"})
    assert s["passed"] is True
    assert s["expected_fingerprints"] == {"k0": "p0", "k1": "p1"}


def test_any_run_error_fails_the_preflight():
    s = pf.summarize([_run(0, 1, "k0", errors=["boom"])], [], {0: "d0"})
    assert s["passed"] is False and "boom" in s["failures"][0]


def test_an_axis_that_changes_nothing_fails_the_preflight():
    s = pf.summarize([_run(0, 1, "k0", "p0"), _run(1, 1, "k1", "p1")],
                     [(0, 1, "turning_angle")], {0: "same", 1: "same"})
    assert s["passed"] is False
    assert "turning_angle" in s["failures"][0] and "did not change" in s["failures"][0]


def test_two_specs_with_one_fingerprint_fail_the_preflight():
    s = pf.summarize([_run(0, 1, "k0", "p"), _run(1, 1, "k1", "p")], [], {0: "a", 1: "b"})
    assert s["passed"] is False and "same parameter_id" in s["failures"][0]
```

- [ ] **Step 2: Run to verify they fail**

Run: `… -m pytest tests/test_dsi_preflight.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'scripts.dsi_preflight'`.

- [ ] **Step 3: Implement**

Create `scripts/dsi_preflight.py`:

```python
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
        # Same command as the verified run except the output file, so the direct
        # export is the same execution (deterministic at one thread).
        cmd = build_track_command(cfg, _dsi_cmd(), subject, str(direct), _atlas_arg(cfg))
        subprocess.run(cmd, capture_output=True, text=True, check=False)
        subprocess.run([_dsi_cmd(), "--action=ana", f"--source={subject}", f"--tract={tract}",
                        f"--output={converted}"], capture_output=True, text=True, check=False)
        if not (direct.exists() and converted.exists()):
            errors.append("geometry export failed (direct or converted .trk missing)")
        else:
            a = nib.streamlines.load(str(direct))
            b = nib.streamlines.load(str(converted))
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
```

Note on `single_axis_pairs(choices)`: `choices` is indexed by the order of `sorted(first)`, which equals the spec index because `enumerate_specs` numbers specs from 0 consecutively; `summarize` receives digests keyed by spec index, so the indices line up.

In `pyproject.toml`, add `"nibabel",` to the `[project] dependencies` list.

In `docs/cli_steps.md`, after the `### view` section, add:

```markdown
### Preflight (before any battery run)

Proves, once per pinned DSI Studio build, that every specification in a sweep
config executes exactly as sent: DSI Studio's echo, its execution report,
streamline geometry from a direct `.trk` export, and that every swept parameter
changes the output. Writes the expected `parameter_id` fingerprints that every
production run is then checked against (`verification.expected_fingerprints`).

```bash
DSI_APPTAINER_IMAGE=/path/to/dsi_studio_<date>.sif \
python -m scripts.dsi_preflight --config configs/battery.json \
  --subject staged/<ds>/<subject>.qsdr.fz --out preflight/ --max-parallel 8
```

Exits non-zero unless every check passed. Requires `thread_count: 1`.
```

- [ ] **Step 4: Run the tests, then the full suite**

Run: `… -m pytest tests/test_dsi_preflight.py -q` — expected: 8 passed.
Run the full suite — expected: all pass.

- [ ] **Step 5: End-to-end check on real data (DSI Studio required)**

This is the acceptance test for the whole plan: the preflight must pass on a small real sweep. Write `/tmp/claude-1002/preflight_smoke.json`:

```json
{
  "tract_count": 50000, "thread_count": 1, "connectivity_values": ["count", "qa"],
  "atlas_dir": "/data/local/software/dsi_studio_atlases/human",
  "reliability": {"repeats": 2},
  "tracking_parameters": {"step_size": 1.0, "smoothing": 0.1, "min_length": 10, "max_length": 250},
  "sweep_parameters": {
    "fa_threshold_range": [0.0], "otsu_range": [0.6], "turning_angle_range": [35, 50],
    "reference_candidate": {"method": null, "otsu_threshold": null, "fa_threshold": null,
      "turning_angle": null, "step_size": null, "smoothing": null, "min_length": null,
      "max_length": null, "tip_iteration": null}
  }
}
```

Run:

```bash
DSI_STUDIO_PATH=/data/local/software/dsistuido/installation/apptainer/run_dsi_studio.sh \
DSI_APPTAINER_IMAGE=/data/local/software/apptainer_images/dsi_studio/dsi_studio_hou-2026-09-27.sif \
OPTICONN_SKIP_VENV=1 /data/local/software/opticonn/braingraph_pipeline/bin/python -m scripts.dsi_preflight \
  --config /tmp/claude-1002/preflight_smoke.json \
  --subject /data/local/software/opticonn-multiverse-battery/data/ds000221/sub-010104_ses-01_dwi.qsdr.fz \
  --out /tmp/claude-1002/preflight_smoke --max-parallel 6
```

Expected: `preflight PASSED: 6 runs, 0 failure(s)` (2 angles + reference = 3 specs x 2 repeats). Paste the `failures` list, the three repeat-1 `executed` blocks and the `determinism` map from `preflight.json` into your report. If it fails, do not adjust a check to make it pass: report the failure verbatim with the run's `pf_AAL3.dsi_execution.json`, and stop.

- [ ] **Step 6: Commit**

```bash
git add scripts/dsi_preflight.py tests/test_dsi_preflight.py pyproject.toml docs/cli_steps.md
git commit -m "$(cat <<'MSG'
feat(preflight): prove every specification executes before a battery runs

One run per specification and repeat on one subject, through the verified
path, plus streamline geometry from an exact .trk export and a differential
check that every swept parameter changes the output. Writes the expected
parameter_id fingerprints production runs are checked against; exits
non-zero unless everything passed.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

## After this plan

Plan B (`opticonn-multiverse-battery`): `battery.json` (single thread, 2 repeats, all-`null` reference, no `dt_threshold`/`track_voxel_ratio`, `verification.expected_fingerprints` → the battery's preflight), container pin, QC screen, pinned upstreams, preprocessing moderator, and flagging any dataset with a failed-verification run in `sweep_failed.txt`.

## Known ceiling

`ponytail:` the expected report statements are those of the pinned 27 Sep build. A build that rephrases its report fails every run until `_REPORT`/`expected_execution` are updated — deliberately fail-closed; upgrading DSI Studio is a conscious step that starts with a new preflight.
