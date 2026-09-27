# Stage Manifest Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the battery pipeline one shared, tested library for "the
producer declares what it made, the consumer verifies it before trusting
it," instead of every stage boundary inventing its own hash/compare logic.

**Architecture:** One new module, `scripts/manifest.py`, in the OptiConn
repo (already the common `PYTHONPATH` dependency of the battery repo):
`write_manifest` hashes a stage's output files and writes `manifest.json`;
`verify_manifest` re-hashes and checks contract keys, returning a list of
problems (empty = OK), never raising for an expected failure. Scope is
deliberately small: build and fully test the library, then do one
low-risk migration (`battery/verification.py`'s `preflight_passed`, in the
`opticonn-multiverse-battery` repo) to prove it against a real, existing
boundary. `fetch.sh`/`stage.py`'s hash check is already fixed and tested
and is explicitly NOT touched by this plan — migrating a working
bash+text boundary onto a new Python/JSON call purely for consistency is
out of scope; do this later only if that file needs to change for another
reason.

**Tech Stack:** Python stdlib only (`hashlib`, `json`, `pathlib`) — no new
runtime dependency in either repo.

## Global Constraints

- No new runtime dependency in either repo.
- TDD: a failing test before every implementation step.
- Never weaken or delete an existing test. `battery/verification.py`'s
  existing tests (`tests/test_verification.py`) assert `preflight_passed`'s
  *behavior*, not its internals — Task 2 must keep every one of them
  passing unchanged, proving the migration is behavior-preserving.
- `verify_manifest` never raises for an expected failure (missing file,
  missing manifest, hash mismatch, contract mismatch) — it returns a list
  of problem strings; empty means OK. This matches the fail-closed style
  already used throughout the battery repo (`battery/verification.py`'s
  `audit`, `battery/stage.py`'s manifest check).
- Do not touch `fetch.sh`, `battery/stage.py`, or `scripts/dsi_preflight.py`
  in this plan.

---

### Task 1: `scripts/manifest.py` — write_manifest and verify_manifest

**Repo:** `/data/local/software/opticonn` (main branch or a feature branch
per your workflow — this repo has no open feature branch for this work,
create `feat/stage-manifest-contracts`).

**Files:**
- Create: `scripts/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Produces (used by Task 2 and any future boundary): `write_manifest(dir:
  Path, stage: str, outputs: dict[str, Path], contract: dict) -> Path`;
  `verify_manifest(dir: Path, expected: dict, filename: str =
  "manifest.json") -> list[str]`.

Both functions load/write JSON. `write_manifest`'s file always has the
shape `{"stage": ..., "produced_at": ..., "outputs": {relpath: sha256},
"contract": {...}}`. `verify_manifest` is written to also work directly
against a flat JSON file that has no `"contract"`/`"outputs"` keys at all
(like OptiConn's existing `preflight.json`) — it merges any `"contract"`
dict into the checkable namespace, then falls back to every other
top-level key, so a boundary that already has a flat JSON file needs no
reshaping to be checked.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_manifest.py`:

```python
import hashlib
import json
from pathlib import Path

import pytest

from scripts import manifest


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_write_then_verify_round_trips_clean(tmp_path):
    f = tmp_path / "sub-01.qsdr.fz"
    f.write_bytes(b"fake fib content")
    manifest.write_manifest(tmp_path, stage="fetch",
                            outputs={"sub-01.qsdr.fz": f},
                            contract={"hub_release_tag": "1.0.0"})
    problems = manifest.verify_manifest(tmp_path, {"hub_release_tag": "1.0.0"})
    assert problems == []


def test_verify_fails_closed_on_a_missing_manifest(tmp_path):
    problems = manifest.verify_manifest(tmp_path, {"hub_release_tag": "1.0.0"})
    assert problems == [f"manifest.json is missing in {tmp_path}"]


def test_verify_fails_closed_on_invalid_json(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json")
    problems = manifest.verify_manifest(tmp_path, {"k": "v"})
    assert problems == [f"manifest.json in {tmp_path} is not valid JSON"]


def test_verify_fails_closed_on_a_tampered_file(tmp_path):
    f = tmp_path / "sub-01.qsdr.fz"
    f.write_bytes(b"original content")
    manifest.write_manifest(tmp_path, stage="fetch", outputs={"sub-01.qsdr.fz": f},
                            contract={})
    f.write_bytes(b"tampered content")
    problems = manifest.verify_manifest(tmp_path, {})
    assert problems == ["sub-01.qsdr.fz does not match its manifest hash"]


def test_verify_fails_closed_on_a_missing_output_file(tmp_path):
    f = tmp_path / "sub-01.qsdr.fz"
    f.write_bytes(b"original content")
    manifest.write_manifest(tmp_path, stage="fetch", outputs={"sub-01.qsdr.fz": f},
                            contract={})
    f.unlink()
    problems = manifest.verify_manifest(tmp_path, {})
    assert problems == ["sub-01.qsdr.fz is missing"]


def test_verify_fails_closed_on_a_contract_mismatch(tmp_path):
    manifest.write_manifest(tmp_path, stage="fetch", outputs={},
                            contract={"hub_release_tag": "1.0.0"})
    problems = manifest.verify_manifest(tmp_path, {"hub_release_tag": "2.0.0"})
    assert problems == ["hub_release_tag is '1.0.0', expected '2.0.0'"]


def test_verify_fails_closed_when_expected_key_is_absent_from_the_manifest(tmp_path):
    manifest.write_manifest(tmp_path, stage="fetch", outputs={}, contract={})
    problems = manifest.verify_manifest(tmp_path, {"hub_release_tag": "1.0.0"})
    assert problems == ["hub_release_tag is '<missing>', expected '1.0.0'"]


def test_verify_checks_flat_json_with_no_contract_or_outputs_keys(tmp_path):
    # A file like OptiConn's own preflight.json: flat, no "contract"/"outputs" nesting.
    (tmp_path / "preflight.json").write_text(json.dumps({
        "passed": True, "dsi_apptainer_image": "dsi_studio_hou-2026-09-27.sif",
    }))
    problems = manifest.verify_manifest(
        tmp_path, {"passed": True, "dsi_apptainer_image": "dsi_studio_hou-2026-09-27.sif"},
        filename="preflight.json")
    assert problems == []
    problems = manifest.verify_manifest(
        tmp_path, {"passed": True, "dsi_apptainer_image": "some_other_build.sif"},
        filename="preflight.json")
    assert problems == [
        "dsi_apptainer_image is 'dsi_studio_hou-2026-09-27.sif', expected 'some_other_build.sif'"
    ]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /data/local/software/opticonn && python -m pytest tests/test_manifest.py -q`
Expected: `ModuleNotFoundError: No module named 'scripts.manifest'` (or
similar import error) for every test.

- [ ] **Step 3: Implement**

Create `scripts/manifest.py`:

```python
"""One shared, tested contract for a stage boundary: the producer writes
what it made and any facts the consumer must check; the consumer verifies
before trusting it. See docs/superpowers/specs/2026-09-27-stage-manifest-contracts-design.md.

    from scripts.manifest import write_manifest, verify_manifest
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(dir: Path, stage: str, outputs: dict[str, Path], contract: dict) -> Path:
    """Hash every path in `outputs`, write dir/manifest.json, return its path."""
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    data = {
        "stage": stage,
        "produced_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "outputs": {name: _sha256(Path(path)) for name, path in outputs.items()},
        "contract": contract,
    }
    path = dir / "manifest.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def verify_manifest(dir: Path, expected: dict, filename: str = "manifest.json") -> list[str]:
    """Check dir/filename against `expected`. [] means OK; never raises.

    Every output listed under "outputs" is re-hashed and must exist and match.
    Every key of `expected` is checked against the manifest's "contract" dict
    (if present) merged with every other top-level key -- so a flat JSON file
    with no "contract"/"outputs" nesting (e.g. OptiConn's own preflight.json)
    is checked directly, with no reshaping needed.
    """
    path = Path(dir) / filename
    if not path.exists():
        return [f"{filename} is missing in {dir}"]
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return [f"{filename} in {dir} is not valid JSON"]

    problems = []
    for name, want in (data.get("outputs") or {}).items():
        candidate = Path(dir) / name
        if not candidate.exists():
            problems.append(f"{name} is missing")
        elif _sha256(candidate) != want:
            problems.append(f"{name} does not match its manifest hash")

    facts = dict(data.get("contract") or {})
    for key, value in data.items():
        if key not in ("contract", "outputs"):
            facts[key] = value
    for key, want in expected.items():
        got = facts.get(key, "<missing>")
        if got != want:
            problems.append(f"{key} is '{got}', expected '{want}'")
    return problems
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/local/software/opticonn && python -m pytest tests/test_manifest.py -v`
Expected: all 9 tests PASS.

- [ ] **Step 5: Run the full OptiConn suite**

Run: `cd /data/local/software/opticonn && python -m pytest tests/ -q`
Expected: prior baseline count plus 9, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add scripts/manifest.py tests/test_manifest.py
git commit -m "$(cat <<'MSG'
feat(manifest): shared write/verify contract for stage boundaries

One tested implementation of "producer declares, consumer verifies"
instead of each battery-pipeline boundary inventing its own hash/compare
logic (each of which had its own bug: CRLF-normalized hashing, regex vs
literal filename matching, no image check at all).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 2: Migrate `battery/verification.py`'s `preflight_passed` onto it

**Repo:** `/data/local/software/opticonn-multiverse-battery`, branch
`feat/battery-readiness` (already checked out — this is a small addition
to the same branch; if that branch has already merged to `main` by the
time this task runs, branch from `main` instead as
`feat/stage-manifest-contracts` and note that in your report).

**Files:**
- Modify: `battery/verification.py` (`preflight_passed`, no signature
  change)
- Test: `tests/test_verification.py` (existing tests must keep passing
  unchanged; this task does not add or remove any of them)

**Interfaces:**
- Consumes: `scripts.manifest.verify_manifest` (Task 1) — requires
  `PYTHONPATH=/data/local/software/opticonn` at test/run time, exactly as
  every other `scripts.*` import in this repo already does (see
  `battery/qc.py`'s `from scripts.qc_gate import ...`,
  `battery/merge.py`'s `from scripts.variance_decomposition import
  collect_sweep_matrices`).
- Produces: nothing new — `preflight_passed(path: Path, expected_image:
  str | None = None) -> bool` keeps its exact existing signature and
  return type. This is a pure internal-implementation swap.

The current `preflight_passed`:

```python
def preflight_passed(path: Path, expected_image: str | None = None) -> bool:
    """True only if the preflight passed against the image this run would actually use.

    A stale preflight, or one recorded against a different DSI Studio build/config,
    must not gate a sweep -- so when `expected_image` is given, the preflight's own
    recorded `dsi_apptainer_image` (written by OptiConn's scripts/dsi_preflight.py)
    must match it, or this fails closed.
    """
    try:
        rec = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return False
    if rec.get("passed") is not True:
        return False
    if expected_image is not None and rec.get("dsi_apptainer_image") != expected_image:
        return False
    return True
```

`path` here is the *file* `preflight/<dsid>/preflight.json`, not its
parent directory — `verify_manifest` takes a directory, so pass
`Path(path).parent` and `filename=Path(path).name`.

- [ ] **Step 1: Confirm the existing tests, unmodified, are your spec**

Read `tests/test_verification.py`'s `test_preflight_passed` and
`test_preflight_passed_fails_closed_on_a_different_image` (already in the
repo — do not rewrite them). Run them now, before touching
`verification.py`, to record the GREEN baseline:

Run: `PYTHONPATH=/data/local/software/opticonn /data/local/software/opticonn/braingraph_pipeline/bin/python -m pytest tests/test_verification.py -v`
Expected: all pass (this is the battery repo's full-suite baseline at the
time you start — record the exact number from `pytest tests/ -q` too, for
your report).

- [ ] **Step 2: Replace the implementation**

In `battery/verification.py`, add the import and replace the function body:

```python
from scripts.manifest import verify_manifest
```

```python
def preflight_passed(path: Path, expected_image: str | None = None) -> bool:
    """True only if the preflight passed against the image this run would actually use.

    A stale preflight, or one recorded against a different DSI Studio build/config,
    must not gate a sweep -- so when `expected_image` is given, the preflight's own
    recorded `dsi_apptainer_image` (written by OptiConn's scripts/dsi_preflight.py)
    must match it, or this fails closed. Delegates to the shared manifest contract
    (scripts.manifest.verify_manifest) rather than re-implementing the field checks.
    """
    path = Path(path)
    expected = {"passed": True}
    if expected_image is not None:
        expected["dsi_apptainer_image"] = expected_image
    return verify_manifest(path.parent, expected, filename=path.name) == []
```

- [ ] **Step 3: Run the existing tests to confirm nothing broke**

Run: `PYTHONPATH=/data/local/software/opticonn /data/local/software/opticonn/braingraph_pipeline/bin/python -m pytest tests/test_verification.py -v`
Expected: the exact same tests from Step 1, all still PASS — no test file
was touched, so this is proof the migration is behavior-preserving.

- [ ] **Step 4: Run the full battery-repo suite**

Run: `PYTHONPATH=/data/local/software/opticonn /data/local/software/opticonn/braingraph_pipeline/bin/python -m pytest tests/ -q`
Expected: same total count as your Step 1 baseline, 0 failures.

- [ ] **Step 5: Commit**

```bash
git add battery/verification.py
git commit -m "$(cat <<'MSG'
refactor(verification): preflight_passed delegates to the shared manifest contract

No behavior change (tests/test_verification.py is untouched and still
green) -- swaps the bespoke field-by-field preflight.json check for
OptiConn's scripts.manifest.verify_manifest, the one shared, tested
implementation every future stage boundary should use instead of
reinventing this.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

## Out of scope (do not do these in this plan)

- Migrating `fetch.sh`/`battery/stage.py`'s hash check onto
  `scripts/manifest.py`. That boundary already works and is fully tested
  with a plain-text `sha256sum`-format manifest; converting it to shell
  out to Python for a JSON manifest is pure format churn with no bug to
  fix, and was explicitly declined for this plan.
- Reshaping `scripts/dsi_preflight.py`'s `preflight.json` output format.
  `scripts/manifest.py`'s `verify_manifest` is written specifically so it
  needs no reshaping (Task 1's flat-JSON test proves this).
- Any new boundary's manifest (survey→snapshot, snapshot→select,
  select→fetch) — no bug history, not touched.
