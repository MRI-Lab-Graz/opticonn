# OpenNeuro Multiverse Battery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained study repository that surveys public OpenNeuro DWI protocols, selects ~13 maximally diverse 3T datasets, downloads their pre-reconstructed `.fz` files from the Fiber Data Hub, runs the OptiConn multiverse sweep on each, and aggregates the results into the evidence base for the methods paper.

**Architecture:** Six small, independently runnable Python/bash stages in one new repo. No scheduler, no crawler, no preprocessing: the Fiber Data Hub supplies per-subject QSDR reconstructions and OpenNeuro's public S3 bucket supplies protocol metadata. OptiConn is consumed as an external pinned dependency and is not modified.

**Tech Stack:** Python 3.10 (stdlib `urllib`, `json`, `csv`, `re`, `collections`, `concurrent.futures`), pandas, `gh` CLI for GitHub releases, pytest. No new dependencies beyond what OptiConn already requires.

**Spec:** `docs/superpowers/specs/2026-09-25-openneuro-multiverse-battery-design.md`

## Global Constraints

- **Public data only.** Every input must be a public OpenNeuro accession plus a pinned Fiber Data Hub release tag. No in-house data anywhere in this repo.
- **OptiConn is not modified.** It is consumed via its CLI and pinned by commit (`dfad2d0`, OptiConn v2.0.0).
- **No scheduler.** Sequential local execution; measured 2.6 h per dataset at `--max-parallel 4`.
- **3T only.** The single 7T dataset (`ds003508`) is run and reported separately, never pooled.
- **Sample 3 subjects per dataset** in the survey. A dataset whose sampled subjects disagree on `Manufacturer`, `MagneticFieldStrength`, `ReceiveCoilName`, `EchoTime` or `RepetitionTime` is flagged `protocol_varies` and excluded from selection, with the differing values recorded.
- **Strata are `manufacturer` x `scheme` only.** Field strength and coil are covariates, never strata.
- `scheme` is derived from the bval: `single-shell` (1 distinct non-zero b), `multi-shell` (2-4), `free` (>=5).
- **Eligibility:** human brain, 3T, >= 20 subjects in the hub release, parseable bval, protocol constant across sampled subjects.
- **Selection rule:** two largest eligible datasets per occupied cell, cells filled rarest-first, ties broken by `dataset_id` ascending. Every exclusion carries a machine-readable reason.
- **Sweep settings, identical to the in-house pilot so results are comparable:** AAL3 atlas; `count` and `qa` connectivity; `tract_count = 50000`; sweep grid `fa_threshold_range [0.0, 0.1]` x `turning_angle_range [35, 50, 65]` x `track_voxel_ratio_range [1.0, 2.0]` (12 candidates); 10 subjects per wave, two waves, 2 repeats.
- **Nothing is dropped silently.** Every excluded dataset, failed fetch and missing field is recorded with a reason.
- Never `git add -A`; add only the files each task lists. Commit messages end with a `Co-Authored-By:` trailer naming the model that did the work (e.g. `Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>`).

## File Structure

| File | Responsibility | Task |
| --- | --- | --- |
| `README.md` | one-command reproduction path | 1 |
| `battery/hub.py` | Fiber Data Hub release listing (GitHub) | 2 |
| `battery/openneuro.py` | OpenNeuro S3 listing + sidecar/bval fetch and parse | 3 |
| `battery/survey.py` | join hub + OpenNeuro into `protocol_table.csv` | 4 |
| `battery/select.py` | eligibility + stratified selection | 5 |
| `fetch.sh` | `gh release download` + manifest | 6 |
| `run_all.sh` | sequential OptiConn sweeps | 6 |
| `battery/merge.py` | per-dataset reports -> `cross_dataset_summary.csv` | 7 |
| `battery/fragility.py` | fragility, effective dimensionality, cross-dataset agreement | 8 |
| `tests/` | unit tests per module, no network | 2-8 |

Modules are split by *external system*: `hub.py` talks to GitHub, `openneuro.py` talks to S3, and the rest is pure logic over their outputs. That boundary is what makes the pure logic testable without network access.

---

### Task 1: Repository skeleton

**Files:**
- Create: `README.md`, `pyproject.toml`, `.gitignore`, `battery/__init__.py`, `tests/__init__.py`, `configs/battery.json`

**Interfaces:**
- Consumes: nothing.
- Produces: an importable `battery` package and `configs/battery.json`, the OptiConn extraction config every sweep uses.

- [ ] **Step 1: Create the repository and package skeleton**

Create a new directory `opticonn-multiverse-battery` (sibling of the opticonn checkout), `git init`, and add:

`.gitignore`:
```
__pycache__/
*.pyc
.pytest_cache/
data/
results/
*.egg-info/
```

`pyproject.toml`:
```toml
[project]
name = "opticonn-multiverse-battery"
version = "0.1.0"
description = "Cross-protocol multiverse battery for structural connectomics on public OpenNeuro data"
requires-python = ">=3.10"
dependencies = ["pandas>=1.5"]

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
```

`battery/__init__.py` and `tests/__init__.py`: empty files.

- [ ] **Step 2: Write the OptiConn extraction config**

`configs/battery.json` — identical tracking settings to the in-house pilot so the results are comparable:

```json
{
  "description": "OpenNeuro multiverse battery: 12-candidate grid, 50k streamlines, AAL3",
  "dsi_studio_cmd": "/data/local/software/dsistuido/installation/apptainer/run_dsi_studio.sh",
  "atlas_dir": "/data/local/software/dsi_studio_atlases/human",
  "atlases": ["AAL3"],
  "connectivity_values": ["count", "qa"],
  "tract_count": 50000,
  "thread_count": 4,
  "tracking_parameters": {
    "method": 1,
    "otsu_threshold": 0.6,
    "fa_threshold": 0.1,
    "turning_angle": 65,
    "step_size": 1.0,
    "smoothing": 0.1,
    "min_length": 30,
    "max_length": 250,
    "track_voxel_ratio": 2.0,
    "dt_threshold": 0.2,
    "check_ending": 0,
    "random_seed": 0,
    "tip_iteration": 0
  },
  "sweep_parameters": {
    "fa_threshold_range": [0.0, 0.1],
    "turning_angle_range": [35, 50, 65],
    "step_size_range": [1.0],
    "smoothing_range": [0.1],
    "min_length_range": [30],
    "max_length_range": [250],
    "track_voxel_ratio_range": [1.0, 2.0],
    "tip_iteration_range": [0]
  }
}
```

The `dsi_studio_cmd` and `atlas_dir` paths are machine-specific; the README must say so.

- [ ] **Step 3: Write the README**

`README.md` must contain, in this order: what the battery is; the pinned OptiConn commit (`dfad2d0`); prerequisites (`gh` authenticated, OptiConn installed with DSI Studio, AAL3 atlas); and the reproduction sequence:

```bash
python -m battery.survey            # -> protocol_table.csv   (~10 min, network)
python -m battery.select            # -> selected_datasets.json  [REVIEW THIS]
./fetch.sh                          # -> data/<dsid>/*.fz
./run_all.sh                        # -> results/<dsid>/...   (~31 h)
python -m battery.merge             # -> cross_dataset_summary.csv
```

State plainly that `configs/battery.json` contains machine-specific paths that must be edited.

- [ ] **Step 4: Verify the package imports and commit**

Run: `python -c "import battery; print('ok')"`
Expected: `ok`

```bash
git add README.md pyproject.toml .gitignore battery/__init__.py tests/__init__.py configs/battery.json
git commit -m "feat: repository skeleton and battery extraction config

Co-Authored-By: <your model> <noreply@anthropic.com>"
```

---

### Task 2: Fiber Data Hub release listing

**Files:**
- Create: `battery/hub.py`, `tests/test_hub.py`

**Interfaces:**
- Consumes: the `gh` CLI.
- Produces:
  - `HUB_REPOS: list[str]` — the five OpenNeuro hub repositories.
  - `list_releases(repo: str) -> list[dict]` — `[{"hub_repo", "release_tag"}]`.
  - `release_assets(repo: str, tag: str) -> list[str]` — asset file names.
  - `dataset_id(tag: str) -> str` — tag up to the first underscore (`ds004856_4` -> `ds004856`).
  - `subjects_from_assets(assets: list[str]) -> list[tuple[str, str | None]]` — `(sub, ses)` pairs from `.fz` names, de-duplicated, input order preserved.

- [ ] **Step 1: Write the failing tests**

`tests/test_hub.py`:

```python
from battery.hub import dataset_id, subjects_from_assets


def test_dataset_id_strips_release_suffix():
    assert dataset_id("ds004856_4") == "ds004856"
    assert dataset_id("ds004639") == "ds004639"


def test_subjects_from_assets_dedupes_and_keeps_order():
    assets = [
        "sub-146_ses-wave1_dwi.qsdr.fz",
        "sub-146_ses-wave2_dwi.qsdr.fz",
        "sub-146_ses-wave1_dwi.qsdr.fz",   # duplicate
        "sub-1467_ses-wave3_dwi.qsdr.fz",
        "README.md",                        # not a .fz
    ]
    assert subjects_from_assets(assets) == [
        ("sub-146", "ses-wave1"),
        ("sub-146", "ses-wave2"),
        ("sub-1467", "ses-wave3"),
    ]


def test_subjects_from_assets_handles_sessionless_names():
    assert subjects_from_assets(["sub-01_dwi.qsdr.fz"]) == [("sub-01", None)]


def test_subjects_from_assets_ignores_unparseable():
    assert subjects_from_assets(["atlas_template.fz", "sub-02_dwi.qsdr.fz"]) == [("sub-02", None)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_hub.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'battery.hub'`.

- [ ] **Step 3: Implement `battery/hub.py`**

```python
"""Fiber Data Hub (brain.labsolver.org) release listing via the GitHub CLI.

The hub publishes per-subject QSDR reconstructions as GitHub release assets.
A release tag is the OpenNeuro accession, sometimes with a numeric suffix
(``ds004856_4``). Animal collections are excluded by repository name.
"""

from __future__ import annotations

import json
import re
import subprocess

HUB_REPOS = [
    "data-openneuro/brain",
    "data-openneuro/brain2",
    "data-openneuro/disease",
    "data-openneuro/disease2",
    "data-openneuro/others",
]

_SUBJECT_RE = re.compile(r"^(sub-[A-Za-z0-9]+)(?:_(ses-[A-Za-z0-9]+))?")


def _gh(args: list[str]) -> str:
    result = subprocess.run(["gh"] + args, capture_output=True, text=True, timeout=180)
    return result.stdout if result.returncode == 0 else ""


def dataset_id(tag: str) -> str:
    """OpenNeuro accession for a hub release tag (``ds004856_4`` -> ``ds004856``)."""
    return tag.split("_")[0]


def list_releases(repo: str) -> list[dict]:
    out = _gh(["release", "list", "-R", repo, "--limit", "500", "--json", "tagName"])
    return [{"hub_repo": repo, "release_tag": t["tagName"]} for t in (json.loads(out) if out else [])]


def release_assets(repo: str, tag: str) -> list[str]:
    out = _gh(["release", "view", tag, "-R", repo, "--json", "assets"])
    return [a["name"] for a in json.loads(out)["assets"]] if out else []


def subjects_from_assets(assets: list[str]) -> list[tuple[str, str | None]]:
    """(subject, session) pairs from .fz asset names, de-duplicated, input order kept."""
    seen: set[tuple[str, str | None]] = set()
    out: list[tuple[str, str | None]] = []
    for name in assets:
        if not name.endswith(".fz"):
            continue
        m = _SUBJECT_RE.match(name)
        if not m:
            continue
        key = (m.group(1), m.group(2))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_hub.py -q`
Expected: 4 passed.

- [ ] **Step 5: Verify against the live hub (network, read-only)**

Run:
```bash
python -c "
from battery.hub import list_releases, release_assets, dataset_id, subjects_from_assets
r = list_releases('data-openneuro/brain')
print(len(r), 'releases; first:', r[0])
a = release_assets('data-openneuro/brain', 'ds004856_4')
print(len(a), 'assets;', len(subjects_from_assets(a)), 'subjects; id:', dataset_id('ds004856_4'))"
```
Expected: about 87 releases; roughly 787 assets; a subject count of the same order; id `ds004856`. If the release list is empty, `gh` is not authenticated — run `gh auth status`.

- [ ] **Step 6: Commit**

```bash
git add battery/hub.py tests/test_hub.py
git commit -m "feat: Fiber Data Hub release listing

Co-Authored-By: <your model> <noreply@anthropic.com>"
```

---

### Task 3: OpenNeuro metadata fetch

**Files:**
- Create: `battery/openneuro.py`, `tests/test_openneuro.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `find_dwi_files(dataset_id: str, sub: str, ses: str | None) -> tuple[str | None, str | None]` — S3 keys of a `_dwi.json` and `_dwi.bval` for that subject, or `(None, None)`.
  - `parse_sidecar(text: str) -> dict` — the metadata fields, missing ones absent.
  - `parse_bval(text: str) -> dict` — `{"shells", "n_directions", "max_b", "scheme"}`.
  - `fetch(url: str) -> str | None` — GET returning text, `None` on any failure.
  - `S3_BASE: str` — `https://s3.amazonaws.com/openneuro.org`.

The hub's asset names omit BIDS `acq-`/`run-` entities that the real OpenNeuro paths carry, so the path cannot be derived from the asset name. `find_dwi_files` therefore lists the S3 prefix and picks the first matching files — the route verified during reconnaissance.

- [ ] **Step 1: Write the failing tests**

`tests/test_openneuro.py`:

```python
import json

from battery.openneuro import parse_bval, parse_sidecar


def test_parse_sidecar_extracts_present_fields_only():
    text = json.dumps({
        "Manufacturer": "Philips",
        "EchoTime": 0.051,
        "RepetitionTime": 4.41,
        "InstitutionName": "Some Hospital",
    })
    got = parse_sidecar(text)
    assert got["manufacturer"] == "Philips"
    assert got["echo_time"] == 0.051
    assert got["institution"] == "Some Hospital"
    assert "field_strength" not in got   # absent, never imputed
    assert "coil" not in got


def test_parse_sidecar_normalises_vendor_case_and_variants():
    assert parse_sidecar(json.dumps({"Manufacturer": "SIEMENS"}))["manufacturer"] == "Siemens"
    assert parse_sidecar(json.dumps({"Manufacturer": "GE MEDICAL SYSTEMS"}))["manufacturer"] == "GE"
    assert parse_sidecar(json.dumps({"Manufacturer": "Philips"}))["manufacturer"] == "Philips"


def test_parse_sidecar_survives_malformed_json():
    assert parse_sidecar("{not json") == {}


def test_parse_bval_single_shell():
    got = parse_bval("0 1000 1000 1000")
    assert got["scheme"] == "single-shell"
    assert got["n_directions"] == 3
    assert got["max_b"] == 1000


def test_parse_bval_multi_shell():
    got = parse_bval("0 " + " ".join(["1000"] * 20 + ["2000"] * 30 + ["3000"] * 64))
    assert got["scheme"] == "multi-shell"
    assert got["n_directions"] == 114


def test_parse_bval_free_qspace():
    # 5+ distinct non-zero b values -> free q-space sampling
    got = parse_bval("0 200 400 700 1200 2000 3000")
    assert got["scheme"] == "free"


def test_parse_bval_rounds_to_nearest_hundred():
    # scanner-reported values wobble; 995 and 1004 are one shell
    got = parse_bval("0 995 1004 1000")
    assert got["scheme"] == "single-shell"


def test_parse_bval_rejects_garbage():
    assert parse_bval("not a bval file") == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_openneuro.py -q`
Expected: `ModuleNotFoundError: No module named 'battery.openneuro'`.

- [ ] **Step 3: Implement `battery/openneuro.py`**

```python
"""Protocol metadata from OpenNeuro's public S3 bucket.

No clone and no API key: a DWI JSON sidecar and its .bval are two plain HTTPS
GETs. Hub asset names omit the BIDS ``acq-``/``run-`` entities that real
OpenNeuro paths carry, so the path is discovered by listing the S3 prefix
rather than derived from the asset name.
"""

from __future__ import annotations

import collections
import json
import re
import urllib.request

S3_BASE = "https://s3.amazonaws.com/openneuro.org"

_SIDECAR_FIELDS = {
    "Manufacturer": "manufacturer",
    "ManufacturersModelName": "model",
    "MagneticFieldStrength": "field_strength",
    "ReceiveCoilName": "coil",
    "InstitutionName": "institution",
    "EchoTime": "echo_time",
    "RepetitionTime": "repetition_time",
    "SliceThickness": "slice_thickness",
    "PhaseEncodingDirection": "pe_direction",
    "MultibandAccelerationFactor": "multiband_factor",
    "ParallelReductionFactorInPlane": "parallel_factor",
    "SequenceName": "sequence_name",
    "SoftwareVersions": "software_versions",
}


def fetch(url: str, timeout: int = 25) -> str | None:
    """GET returning text, or None on any failure (caller records the reason)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8", "replace")
    except Exception:
        return None


def _normalise_vendor(value: str) -> str:
    """Vendor strings vary in case and suffix: SIEMENS, 'GE MEDICAL SYSTEMS'."""
    upper = str(value).strip().upper()
    if upper.startswith("SIEMENS"):
        return "Siemens"
    if upper.startswith("GE"):
        return "GE"
    if upper.startswith("PHILIPS"):
        return "Philips"
    return str(value).strip()


def parse_sidecar(text: str) -> dict:
    """Recognised sidecar fields that are present. Absent fields stay absent."""
    try:
        doc = json.loads(text)
    except Exception:
        return {}
    if not isinstance(doc, dict):
        return {}
    out = {}
    for bids_key, our_key in _SIDECAR_FIELDS.items():
        if bids_key in doc:
            out[our_key] = doc[bids_key]
    if "manufacturer" in out:
        out["manufacturer"] = _normalise_vendor(out["manufacturer"])
    return out


def parse_bval(text: str) -> dict:
    """Shell structure from a .bval file. Empty dict if it does not parse."""
    try:
        # b < 50 s/mm^2 is b0 by convention; round(50,-2) would otherwise fold a real shell into b0
        values = [0 if float(x) < 50 else round(float(x), -2) for x in text.split() if x.strip()]
    except ValueError:
        return {}
    if not values:
        return {}
    counts = collections.Counter(values)
    non_zero = sorted(b for b in counts if b > 0)
    if not non_zero:
        return {}
    scheme = "single-shell" if len(non_zero) == 1 else "multi-shell" if len(non_zero) <= 4 else "free"
    return {
        "shells": "|".join(f"{int(b)}x{counts[b]}" for b in sorted(counts)),
        "n_directions": sum(counts[b] for b in non_zero),
        "max_b": int(max(counts)),
        "scheme": scheme,
    }


def find_dwi_files(dataset_id: str, sub: str, ses: str | None) -> tuple[str | None, str | None]:
    """S3 keys of a _dwi.json and _dwi.bval for this subject, via prefix listing."""
    prefix = f"{dataset_id}/{sub}/" + (f"{ses}/" if ses else "")
    xml = fetch(f"{S3_BASE}?list-type=2&prefix={prefix}&max-keys=400")
    if not xml:
        return None, None
    keys = re.findall(r"<Key>([^<]*)</Key>", xml)
    sidecar = next((k for k in keys if k.endswith("_dwi.json") and "/dwi/" in k), None)
    bval = next((k for k in keys if k.endswith("_dwi.bval") and "/dwi/" in k), None)
    return sidecar, bval
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_openneuro.py -q`
Expected: 8 passed.

- [ ] **Step 5: Verify against live S3 (network, read-only)**

Run:
```bash
python -c "
from battery.openneuro import find_dwi_files, fetch, parse_sidecar, parse_bval, S3_BASE
j, b = find_dwi_files('ds004856', 'sub-146', 'ses-wave1')
print('sidecar:', j); print('bval:', b)
print(parse_sidecar(fetch(f'{S3_BASE}/{j}')))
print(parse_bval(fetch(f'{S3_BASE}/{b}')))"
```
Expected: both keys found; the sidecar shows `manufacturer: Philips`, `echo_time: 0.051`; the bval shows `scheme: single-shell`, `n_directions: 30`.

- [ ] **Step 6: Commit**

```bash
git add battery/openneuro.py tests/test_openneuro.py
git commit -m "feat: OpenNeuro S3 protocol metadata fetch and parsing

Co-Authored-By: <your model> <noreply@anthropic.com>"
```

---

### Task 4: Protocol survey

**Files:**
- Create: `battery/survey.py`, `tests/test_survey.py`

**Interfaces:**
- Consumes: `battery.hub` (Task 2) and `battery.openneuro` (Task 3).
- Produces:
  - `survey_dataset(hub_repo, release_tag, assets, sampler) -> dict` — one protocol row.
  - `main() -> int` — writes `protocol_table.csv`.
  - Row keys: `dataset_id, hub_repo, release_tag, n_subjects, n_sampled, manufacturer, model, field_strength, coil, institution, echo_time, repetition_time, slice_thickness, pe_direction, multiband_factor, parallel_factor, sequence_name, software_versions, shells, n_directions, max_b, scheme, protocol_varies, varying_fields, note`.

`sampler(dataset_id, sub, ses) -> dict` returns one subject's parsed metadata. Injecting it is what makes `survey_dataset` testable without network access.

- [ ] **Step 1: Write the failing tests**

`tests/test_survey.py`:

```python
from battery.survey import CONSISTENCY_FIELDS, survey_dataset

ASSETS = [f"sub-{i:03d}_dwi.qsdr.fz" for i in range(1, 31)]


def _sampler(rows):
    """Return a sampler yielding the given rows in order, ignoring its arguments."""
    seq = list(rows)

    def sample(dataset_id, sub, ses):
        return seq.pop(0) if seq else {}

    return sample


def test_survey_dataset_reports_consistent_protocol():
    row = {"manufacturer": "Siemens", "field_strength": 3, "echo_time": 0.09,
           "scheme": "multi-shell", "n_directions": 114, "max_b": 3000, "shells": "0x3|1000x20"}
    got = survey_dataset("data-openneuro/brain", "ds000001", ASSETS, _sampler([row, row, row]))
    assert got["protocol_varies"] is False
    assert got["manufacturer"] == "Siemens"
    assert got["scheme"] == "multi-shell"
    assert got["n_subjects"] == 30
    assert got["n_sampled"] == 3


def test_survey_dataset_flags_varying_protocol():
    # 11% of real datasets vary; EchoTime is the field that varied in both cases found
    a = {"manufacturer": "Siemens", "echo_time": 0.073, "scheme": "multi-shell"}
    b = dict(a, echo_time=0.087)
    got = survey_dataset("data-openneuro/brain", "ds003508", ASSETS, _sampler([a, b, a]))
    assert got["protocol_varies"] is True
    assert "echo_time" in got["varying_fields"]


def test_survey_dataset_records_missing_fields_as_unknown():
    got = survey_dataset("data-openneuro/brain", "ds000002", ASSETS,
                         _sampler([{"manufacturer": "GE", "scheme": "single-shell"}] * 3))
    assert got["field_strength"] == "unknown"
    assert got["coil"] == "unknown"
    assert got["institution"] == "unknown"


def test_survey_dataset_notes_when_no_subject_resolved():
    got = survey_dataset("data-openneuro/brain", "ds000003", ASSETS, _sampler([{}, {}, {}]))
    assert got["scheme"] == "unknown"
    assert got["note"] == "no dwi metadata resolved"


def test_survey_dataset_notes_when_release_has_no_fz():
    got = survey_dataset("data-openneuro/brain", "ds000004", ["README.md"], _sampler([]))
    assert got["n_subjects"] == 0
    assert got["note"] == "no .fz assets"


def test_consistency_fields_are_the_spec_five():
    assert set(CONSISTENCY_FIELDS) == {
        "manufacturer", "field_strength", "coil", "echo_time", "repetition_time"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_survey.py -q`
Expected: `ModuleNotFoundError: No module named 'battery.survey'`.

- [ ] **Step 3: Implement `battery/survey.py`**

```python
"""Join Fiber Data Hub releases with OpenNeuro protocol metadata.

Three subjects are sampled per dataset, not one: reconnaissance found 11% of
datasets vary their acquisition across subjects, so a single-subject survey
mischaracterises roughly one dataset in nine.
"""

from __future__ import annotations

import argparse
import csv
import logging
from concurrent.futures import ThreadPoolExecutor

from battery import hub, openneuro

N_SAMPLE = 3
CONSISTENCY_FIELDS = ("manufacturer", "field_strength", "coil", "echo_time", "repetition_time")

COLUMNS = [
    "dataset_id", "hub_repo", "release_tag", "n_subjects", "n_sampled",
    "manufacturer", "model", "field_strength", "coil", "institution",
    "echo_time", "repetition_time", "slice_thickness", "pe_direction",
    "multiband_factor", "parallel_factor", "sequence_name", "software_versions",
    "shells", "n_directions", "max_b", "scheme", "protocol_varies",
    "varying_fields", "note",
]

_OPTIONAL = ("model", "field_strength", "coil", "institution", "echo_time",
             "repetition_time", "slice_thickness", "pe_direction", "multiband_factor",
             "parallel_factor", "sequence_name", "software_versions",
             "shells", "n_directions", "max_b")


def sample_subject(dataset_id: str, sub: str, ses: str | None) -> dict:
    """Parsed sidecar + bval metadata for one subject; {} if nothing resolved."""
    sidecar_key, bval_key = openneuro.find_dwi_files(dataset_id, sub, ses)
    out: dict = {}
    if sidecar_key:
        text = openneuro.fetch(f"{openneuro.S3_BASE}/{sidecar_key}")
        if text:
            out.update(openneuro.parse_sidecar(text))
    if bval_key:
        text = openneuro.fetch(f"{openneuro.S3_BASE}/{bval_key}")
        if text:
            out.update(openneuro.parse_bval(text))
    return out


def survey_dataset(hub_repo: str, release_tag: str, assets: list[str], sampler=sample_subject) -> dict:
    """One protocol row, sampling up to N_SAMPLE subjects."""
    dataset_id = hub.dataset_id(release_tag)
    row = {k: "" for k in COLUMNS}
    row.update(dataset_id=dataset_id, hub_repo=hub_repo, release_tag=release_tag,
               manufacturer="unknown", scheme="unknown", protocol_varies=False,
               varying_fields="", note="", n_sampled=0)

    subjects = hub.subjects_from_assets(assets)
    row["n_subjects"] = len(subjects)
    if not subjects:
        row["note"] = "no .fz assets"
        for field in _OPTIONAL:
            row[field] = "unknown"
        return row

    samples = [s for s in (sampler(dataset_id, sub, ses) for sub, ses in subjects[:N_SAMPLE]) if s]
    row["n_sampled"] = len(samples)
    if not samples:
        row["note"] = "no dwi metadata resolved"
        for field in _OPTIONAL:
            row[field] = "unknown"
        return row

    first = samples[0]
    for key in ("manufacturer", "shells", "n_directions", "max_b", "scheme"):
        if key in first:
            row[key] = first[key]
    for field in _OPTIONAL:
        row[field] = first.get(field, "unknown")

    varying = [f for f in CONSISTENCY_FIELDS
               if len({repr(s[f]) for s in samples if f in s}) > 1]
    row["protocol_varies"] = bool(varying)
    row["varying_fields"] = "|".join(varying)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-o", "--output", default="protocol_table.csv")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    releases = [r for repo in hub.HUB_REPOS for r in hub.list_releases(repo)]
    logging.info("%d releases across %d repositories", len(releases), len(hub.HUB_REPOS))

    def work(release):
        assets = hub.release_assets(release["hub_repo"], release["release_tag"])
        return survey_dataset(release["hub_repo"], release["release_tag"], assets)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(work, releases))

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    resolved = sum(1 for r in rows if r["scheme"] != "unknown")
    varying = sum(1 for r in rows if r["protocol_varies"])
    logging.info("wrote %s: %d rows, %d resolved, %d with varying protocol",
                 args.output, len(rows), resolved, varying)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_survey.py -q`
Expected: 6 passed.

- [ ] **Step 5: Run the real survey and sanity-check it against reconnaissance**

Run: `python -m battery.survey -o protocol_table.csv`
Expected: about 165 rows, roughly 130 resolved. Reconnaissance (single-subject) found 132 resolved of 165; this run samples three subjects, so a small difference is expected. If fewer than 100 resolve, stop and report — the S3 listing route has regressed.

- [ ] **Step 6: Commit**

```bash
git add battery/survey.py tests/test_survey.py
git commit -m "feat: protocol survey joining hub releases with OpenNeuro metadata

Co-Authored-By: <your model> <noreply@anthropic.com>"
```

---

### Task 5: Eligibility and stratified selection

**Files:**
- Create: `battery/select.py`, `tests/test_select.py`

**Interfaces:**
- Consumes: `protocol_table.csv` from Task 4.
- Produces:
  - `eligible(df) -> tuple[DataFrame, DataFrame]` — `(eligible_rows, excluded_rows_with_reason)`.
  - `select(eligible_df, per_cell=2, max_datasets=13) -> DataFrame`.
  - `main() -> int` — writes `selected_datasets.json` and `selected_ids.txt`.

- [ ] **Step 1: Write the failing tests**

`tests/test_select.py`:

```python
import pandas as pd

from battery.select import eligible, select


def _row(dsid, vendor="Siemens", scheme="multi-shell", n=50, fs=3.0, varies=False):
    return {"dataset_id": dsid, "manufacturer": vendor, "scheme": scheme,
            "n_subjects": n, "field_strength": fs, "protocol_varies": varies,
            "hub_repo": "data-openneuro/brain", "release_tag": dsid, "note": ""}


def test_eligible_applies_every_criterion_with_reasons():
    df = pd.DataFrame([
        _row("ds_ok"),
        _row("ds_small", n=19),
        _row("ds_7t", fs=7.0),
        _row("ds_unknown_scheme", scheme="unknown"),
        _row("ds_unknown_vendor", vendor="unknown"),
        _row("ds_varies", varies=True),
    ])
    ok, excluded = eligible(df)
    assert list(ok.dataset_id) == ["ds_ok"]
    reasons = dict(zip(excluded.dataset_id, excluded.exclusion_reason))
    assert "fewer than 20 subjects" in reasons["ds_small"]
    assert "not 3T" in reasons["ds_7t"]
    assert "unparseable bval" in reasons["ds_unknown_scheme"]
    assert "unknown vendor" in reasons["ds_unknown_vendor"]
    assert "protocol varies" in reasons["ds_varies"]


def test_eligible_accepts_unknown_field_strength():
    # 3T is the corpus default; absent field strength must not exclude a dataset
    ok, _ = eligible(pd.DataFrame([_row("ds_nofs", fs="unknown")]))
    assert list(ok.dataset_id) == ["ds_nofs"]


def test_select_takes_two_largest_per_cell():
    df = pd.DataFrame([
        _row("ds_a", n=100), _row("ds_b", n=90), _row("ds_c", n=80),   # same cell
        _row("ds_d", vendor="GE", n=70),
    ])
    got = select(eligible(df)[0], per_cell=2)
    assert set(got.dataset_id) == {"ds_a", "ds_b", "ds_d"}   # ds_c is third in its cell


def test_select_fills_rarest_cells_first_and_caps():
    rows = [_row(f"ds_s{i}", n=100 - i) for i in range(6)]                       # 6 Siemens
    rows += [_row("ds_ge", vendor="GE", n=10_000)]                               # 1 GE (rare)
    got = select(eligible(pd.DataFrame(rows))[0], per_cell=2, max_datasets=2)
    assert "ds_ge" in set(got.dataset_id)   # rare cell wins a slot despite the cap


def test_select_is_deterministic_on_ties():
    df = pd.DataFrame([_row("ds_b", n=50), _row("ds_a", n=50), _row("ds_c", n=50)])
    first = list(select(eligible(df)[0], per_cell=2).dataset_id)
    second = list(select(eligible(df)[0], per_cell=2).dataset_id)
    assert first == second == ["ds_a", "ds_b"]   # ties broken by dataset_id ascending
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_select.py -q`
Expected: `ModuleNotFoundError: No module named 'battery.select'`.

- [ ] **Step 3: Implement `battery/select.py`**

```python
"""Eligibility filtering and stratified selection.

Strata are manufacturer x scheme. Field strength is not a stratum: the survey
found one 7T dataset among 76 eligible, so there is no variance to stratify on.
Cells are filled rarest-first because rare vendor/scheme combinations buy the
most protocol spread, and two datasets per cell separate a protocol effect from
one dataset's idiosyncrasy.
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

MIN_SUBJECTS = 20
PER_CELL = 2
MAX_DATASETS = 13


def eligible(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(eligible, excluded-with-reason). Every exclusion carries a reason."""
    work = df.copy()
    work["n_subjects"] = pd.to_numeric(work["n_subjects"], errors="coerce").fillna(0)
    field = pd.to_numeric(work["field_strength"], errors="coerce")
    varies = work["protocol_varies"].astype(str).str.lower().isin(["true", "1"])

    reasons = pd.Series("", index=work.index)
    reasons[work["scheme"].astype(str) == "unknown"] = "unparseable bval"
    reasons[work["manufacturer"].astype(str).isin(["unknown", "Bruker", "nan"])] = "unknown vendor or non-human"
    reasons[field.notna() & (field != 3.0)] = "not 3T"
    reasons[work["n_subjects"] < MIN_SUBJECTS] = f"fewer than {MIN_SUBJECTS} subjects"
    reasons[varies] = "protocol varies across sampled subjects"

    keep = reasons == ""
    excluded = work.loc[~keep].copy()
    excluded["exclusion_reason"] = reasons.loc[~keep]
    return work.loc[keep].copy(), excluded


def select(df: pd.DataFrame, per_cell: int = PER_CELL, max_datasets: int = MAX_DATASETS) -> pd.DataFrame:
    """Up to `per_cell` largest datasets per manufacturer x scheme cell, rarest cell first."""
    if df.empty:
        return df
    ordered = df.sort_values(["n_subjects", "dataset_id"], ascending=[False, True])
    cells = {key: group for key, group in ordered.groupby(["manufacturer", "scheme"], sort=True)}
    # rarest cells first; ties by cell name so the order is deterministic
    order = sorted(cells, key=lambda k: (len(cells[k]), k))

    picked: list[pd.DataFrame] = []
    total = 0
    for rank in range(per_cell):
        for key in order:
            group = cells[key]
            if rank >= len(group) or total >= max_datasets:
                continue
            picked.append(group.iloc[[rank]])
            total += 1
    out = pd.concat(picked) if picked else df.iloc[0:0]
    return out.sort_values(["manufacturer", "scheme", "dataset_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-i", "--input", default="protocol_table.csv")
    parser.add_argument("-o", "--output", default="selected_datasets.json")
    parser.add_argument("--ids", default="selected_ids.txt")
    parser.add_argument("--per-cell", type=int, default=PER_CELL)
    parser.add_argument("--max-datasets", type=int, default=MAX_DATASETS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = pd.read_csv(args.input)
    ok, excluded = eligible(df)
    chosen = select(ok, args.per_cell, args.max_datasets)

    payload = {
        "rule": (f"two largest eligible datasets per manufacturer x scheme cell, "
                 f"rarest cell first, capped at {args.max_datasets}; 3T only"),
        "n_surveyed": int(len(df)),
        "n_eligible": int(len(ok)),
        "selected": chosen.to_dict("records"),
        "eligible_not_selected": ok[~ok.dataset_id.isin(chosen.dataset_id)].to_dict("records"),
        "excluded": excluded[["dataset_id", "exclusion_reason"]].to_dict("records"),
    }
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    with open(args.ids, "w") as handle:
        for dsid in chosen.dataset_id:
            handle.write(f"{dsid}\n")

    logging.info("surveyed %d, eligible %d, selected %d -> %s",
                 len(df), len(ok), len(chosen), args.output)
    for _, row in chosen.iterrows():
        logging.info("  %-12s %-8s %-13s n=%s", row.dataset_id, row.manufacturer,
                     row.scheme, row.n_subjects)
    logging.info("REVIEW %s before running fetch.sh - everything after this costs compute",
                 args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_select.py -q`
Expected: 5 passed.

- [ ] **Step 5: Run selection on the real survey output**

Run: `python -m battery.select -i protocol_table.csv`
Expected: roughly 70-80 eligible and up to 13 selected, spanning Siemens/Philips/GE and single-shell/multi-shell/free. `ds003508` (7T) must appear in `excluded` with reason `not 3T`. If any selected dataset has fewer than 20 subjects, stop and report.

- [ ] **Step 6: Commit**

```bash
git add battery/select.py tests/test_select.py
git commit -m "feat: eligibility filtering and stratified dataset selection

Co-Authored-By: <your model> <noreply@anthropic.com>"
```

---

### Task 6: Download and sweep runner

**Files:**
- Create: `fetch.sh`, `run_all.sh`, `tests/test_scripts.py`

**Interfaces:**
- Consumes: `selected_datasets.json` and `selected_ids.txt` (Task 5); `configs/battery.json` (Task 1).
- Produces: `data/<dataset_id>/*.fz` with `manifest.csv`; `results/<dataset_id>/` sweep outputs.

- [ ] **Step 1: Write the failing test**

`tests/test_scripts.py`:

```python
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fetch_is_executable_and_has_dry_run():
    script = ROOT / "fetch.sh"
    assert script.exists() and script.stat().st_mode & 0o111, "fetch.sh must be executable"
    out = subprocess.run(["bash", str(script), "--dry-run"], capture_output=True, text=True,
                         cwd=ROOT, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "gh release download" in out.stdout


def test_run_all_is_executable_and_has_dry_run():
    script = ROOT / "run_all.sh"
    assert script.exists() and script.stat().st_mode & 0o111, "run_all.sh must be executable"
    out = subprocess.run(["bash", str(script), "--dry-run"], capture_output=True, text=True,
                         cwd=ROOT, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "opticonn" in out.stdout and "tune-grid" in out.stdout
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_scripts.py -q`
Expected: both fail — `fetch.sh` and `run_all.sh` do not exist.

- [ ] **Step 3: Write `fetch.sh`**

```bash
#!/usr/bin/env bash
# Download selected datasets' .fz reconstructions from the Fiber Data Hub.
# Idempotent: files already present with a matching SHA256 are skipped.
set -euo pipefail
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

SEL=${SEL:-selected_datasets.json}
[ -f "$SEL" ] || { echo "missing $SEL - run: python -m battery.select" >&2; exit 1; }

python3 -c '
import json, sys
sel = json.load(open(sys.argv[1]))
for row in sel["selected"]:
    print(row["dataset_id"], row["hub_repo"], row["release_tag"])
' "$SEL" | while read -r dsid repo tag; do
  outdir="data/$dsid"
  cmd="gh release download $tag -R $repo --pattern '*.fz' --dir $outdir --skip-existing"
  if [ "$DRY" = "1" ]; then echo "[dry-run] $cmd"; continue; fi
  mkdir -p "$outdir"
  echo "== $dsid ($repo $tag)"
  eval "$cmd"
  ( cd "$outdir" && sha256sum *.fz > manifest.csv 2>/dev/null || true )
  echo "   $(ls "$outdir"/*.fz 2>/dev/null | wc -l) files"
done
```

Make it executable: `chmod +x fetch.sh`

- [ ] **Step 4: Write `run_all.sh`**

```bash
#!/usr/bin/env bash
# Run the OptiConn multiverse sweep on each selected dataset, sequentially.
# Idempotent: a dataset whose results already exist is skipped.
set -euo pipefail
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

IDS=${IDS:-selected_ids.txt}
CONFIG=${CONFIG:-configs/battery.json}
OPTICONN=${OPTICONN:-opticonn}
[ -f "$IDS" ] || { echo "missing $IDS - run: python -m battery.select" >&2; exit 1; }

while read -r dsid; do
  [ -z "$dsid" ] && continue
  if [ -d "results/$dsid/optimize/optimization_results" ]; then
    echo "== $dsid: already done, skipping"; continue
  fi
  cmd="$OPTICONN tune-grid -i data/$dsid -o results/$dsid --extraction-config $CONFIG --max-parallel 4"
  if [ "$DRY" = "1" ]; then echo "[dry-run] $cmd"; continue; fi
  echo "== $dsid: starting $(date -Is)"
  $cmd
  echo "== $dsid: finished $(date -Is)"
done < "$IDS"
```

Make it executable: `chmod +x run_all.sh`

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_scripts.py -q`
Expected: 2 passed.

If `fetch.sh --dry-run` fails because `selected_datasets.json` is absent, create a minimal one first:
```bash
echo '{"selected":[{"dataset_id":"ds000001","hub_repo":"data-openneuro/brain","release_tag":"ds000001"}]}' > selected_datasets.json
```

- [ ] **Step 6: Commit**

```bash
git add fetch.sh run_all.sh tests/test_scripts.py
git commit -m "feat: dataset download and sequential sweep runner

Co-Authored-By: <your model> <noreply@anthropic.com>"
```

---

### Task 7: Cross-dataset merge

**Files:**
- Create: `battery/merge.py`, `tests/test_merge.py`

**Interfaces:**
- Consumes: `results/<dataset_id>/optimize/` OptiConn outputs — `optimization_results/variance_decomposition.csv`, `optimization_results/graph_icc.csv`, and `wave1/combo_diagnostics.csv` / `wave2/combo_diagnostics.csv`.
- Produces: `dataset_summary(results_dir, dataset_id) -> list[dict]`, `main() -> int` writing `cross_dataset_summary.csv`.
- Row keys: `dataset_id, metric, n_candidates, discriminability_saturated, cross_wave_rho, winner_wave1, winner_wave2, winners_agree, top_two_separation, wave_variation, winner_determined, parameter_over_between_subject, tracking_noise_over_parameter, note`.

- [ ] **Step 1: Write the failing tests**

`tests/test_merge.py`:

```python
import csv
from pathlib import Path

import pytest

from battery.merge import dataset_summary


def _write(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(root: Path, dsid: str, w1_margins, w2_margins, discr=1.0):
    base = root / dsid / "optimize"
    for wave, margins in (("wave1", w1_margins), ("wave2", w2_margins)):
        rows = [{"sweep_id": f"sweep_{i:04d}", "discriminability": discr,
                 "discriminability_margin": m, "repeatability": 0.9, "rejected": ""}
                for i, m in enumerate(margins, 1)]
        _write(base / wave / "combo_diagnostics.csv", rows,
               ["sweep_id", "discriminability", "discriminability_margin",
                "repeatability", "rejected"])
    _write(base / "optimization_results" / "variance_decomposition.csv",
           [{"atlas": "AAL3", "metric": "count", "stratum": s, "mean_dissimilarity": v}
            for s, v in (("tracking_noise", 0.01), ("parameter", 0.05), ("between_subject", 0.25))],
           ["atlas", "metric", "stratum", "mean_dissimilarity"])
    return base


def test_dataset_summary_computes_headline_quantities(tmp_path):
    _fixture(tmp_path, "ds01", [0.10, 0.20, 0.30], [0.11, 0.21, 0.31])
    [row] = [r for r in dataset_summary(tmp_path, "ds01") if r["metric"] == "count"]
    assert row["discriminability_saturated"] is True
    assert row["cross_wave_rho"] == pytest.approx(1.0)
    assert row["winners_agree"] is True
    assert row["parameter_over_between_subject"] == pytest.approx(0.2)
    assert row["tracking_noise_over_parameter"] == pytest.approx(0.2)


def test_dataset_summary_flags_an_undetermined_winner(tmp_path):
    # top two separated by 0.001, waves disagree by 0.05 -> not determined
    _fixture(tmp_path, "ds02", [0.300, 0.299, 0.10], [0.250, 0.251, 0.05])
    [row] = [r for r in dataset_summary(tmp_path, "ds02") if r["metric"] == "count"]
    assert row["winner_determined"] is False
    assert row["top_two_separation"] < row["wave_variation"]


def test_dataset_summary_reports_missing_results(tmp_path):
    (tmp_path / "ds03").mkdir()
    rows = dataset_summary(tmp_path, "ds03")
    assert rows and rows[0]["note"] == "no sweep outputs found"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_merge.py -q`
Expected: `ModuleNotFoundError: No module named 'battery.merge'`.

- [ ] **Step 3: Implement `battery/merge.py`**

```python
"""Aggregate per-dataset OptiConn outputs into one cross-dataset table.

Computes, per dataset and connectivity metric, the quantities the paper
reports: whether the reliability screen saturated, whether the candidate
ordering replicated across waves, whether the leading candidate's margin over
the runner-up exceeds the wave-to-wave variation (the "determined vs not"
test), and the variance-decomposition ratios.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

COLUMNS = [
    "dataset_id", "metric", "n_candidates", "discriminability_saturated",
    "cross_wave_rho", "winner_wave1", "winner_wave2", "winners_agree",
    "top_two_separation", "wave_variation", "winner_determined",
    "parameter_over_between_subject", "tracking_noise_over_parameter", "note",
]


def _empty_row(dataset_id: str, note: str) -> dict:
    row = {k: "" for k in COLUMNS}
    row.update(dataset_id=dataset_id, metric="", note=note)
    return row


def _ratios(decomp_path: Path) -> dict[str, dict[str, float]]:
    """{metric: {parameter_over_between_subject, tracking_noise_over_parameter}}"""
    if not decomp_path.exists():
        return {}
    df = pd.read_csv(decomp_path)
    out: dict[str, dict[str, float]] = {}
    for metric, group in df.groupby("metric"):
        means = dict(zip(group.stratum, pd.to_numeric(group.mean_dissimilarity, errors="coerce")))
        param, between, noise = means.get("parameter"), means.get("between_subject"), means.get("tracking_noise")
        entry: dict[str, float] = {}
        if param and between:
            entry["parameter_over_between_subject"] = param / between
        if noise is not None and param:
            entry["tracking_noise_over_parameter"] = noise / param
        out[str(metric)] = entry
    return out


def dataset_summary(results_dir: Path, dataset_id: str) -> list[dict]:
    """One row per (dataset, connectivity metric)."""
    base = Path(results_dir) / dataset_id / "optimize"
    w1_path, w2_path = base / "wave1" / "combo_diagnostics.csv", base / "wave2" / "combo_diagnostics.csv"
    if not w1_path.exists() or not w2_path.exists():
        return [_empty_row(dataset_id, "no sweep outputs found")]

    w1, w2 = pd.read_csv(w1_path), pd.read_csv(w2_path)
    shared = sorted(set(w1.sweep_id) & set(w2.sweep_id))
    ratios = _ratios(base / "optimization_results" / "variance_decomposition.csv")
    metrics = sorted(ratios) or ["count"]

    rows = []
    for metric in metrics:
        row = {k: "" for k in COLUMNS}
        row.update(dataset_id=dataset_id, metric=metric, n_candidates=len(shared), note="")
        m1 = dict(zip(w1.sweep_id, pd.to_numeric(w1.discriminability_margin, errors="coerce")))
        m2 = dict(zip(w2.sweep_id, pd.to_numeric(w2.discriminability_margin, errors="coerce")))
        discr = pd.to_numeric(w1.discriminability, errors="coerce")
        row["discriminability_saturated"] = bool((discr.dropna() == 1.0).all())

        if len(shared) >= 3:
            rho, _ = spearmanr([m1[s] for s in shared], [m2[s] for s in shared])
            row["cross_wave_rho"] = float(rho)
            row["winner_wave1"] = max(shared, key=lambda s: m1[s])
            row["winner_wave2"] = max(shared, key=lambda s: m2[s])
            row["winners_agree"] = row["winner_wave1"] == row["winner_wave2"]
            mean_margin = {s: (m1[s] + m2[s]) / 2 for s in shared}
            top = sorted(mean_margin, key=lambda s: -mean_margin[s])[:2]
            row["top_two_separation"] = abs(mean_margin[top[0]] - mean_margin[top[1]])
            row["wave_variation"] = sum(abs(m1[s] - m2[s]) for s in shared) / len(shared)
            row["winner_determined"] = bool(row["top_two_separation"] > row["wave_variation"])
        else:
            row["note"] = f"fewer than 3 shared candidates ({len(shared)})"

        for key, value in ratios.get(metric, {}).items():
            row[key] = value
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-i", "--results-dir", default="results")
    parser.add_argument("--ids", default="selected_ids.txt")
    parser.add_argument("-o", "--output", default="cross_dataset_summary.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ids = [line.strip() for line in open(args.ids) if line.strip()]
    rows = [r for dsid in ids for r in dataset_summary(Path(args.results_dir), dsid)]
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    missing = [r["dataset_id"] for r in rows if r["note"] == "no sweep outputs found"]
    logging.info("wrote %s: %d rows from %d datasets", args.output, len(rows), len(ids))
    if missing:
        logging.warning("no results for: %s", ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Add `scipy` to `pyproject.toml` dependencies (it arrives with OptiConn, but this package uses it directly).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_merge.py -q`
Expected: 3 passed.

- [ ] **Step 5: Run the whole suite and commit**

Run: `python -m pytest tests -q`
Expected: all pass (about 28 tests).

```bash
git add battery/merge.py tests/test_merge.py pyproject.toml
git commit -m "feat: cross-dataset merge of per-dataset sweep reports

Co-Authored-By: <your model> <noreply@anthropic.com>"
```

---

### Task 8: Fragility and cross-dataset agreement

**Files:**
- Create: `battery/fragility.py`, `tests/test_fragility.py`
- Modify: `battery/merge.py` (call the new functions from `main`)

**Interfaces:**
- Consumes: matrices via OptiConn's `scripts.variance_decomposition.collect_sweep_matrices` and `scripts.compute_network_measures_from_connectivity.measures_from_matrix`; `dataset_summary` from Task 7.
- Produces:
  - `fragility(combo_matrices) -> dict` with keys `seed_rho`, `param_rho`, `gap`, `n_measures`.
  - `effective_dimensionality(combo_matrices) -> float`.
  - `cross_dataset_agreement(margins_by_dataset) -> list[dict]` with keys `dataset_a`, `dataset_b`, `rho`, `n_shared`.
  - `merge.main()` additionally writes `fragility_summary.csv` and `cross_dataset_agreement.csv`.

This is the spec's Component 6 analysis that `merge.py` alone does not cover: the seed-versus-parameter ordering contrast, the graph battery's effective dimensionality, and whether the candidate ranking transfers between datasets.

- [ ] **Step 1: Write the failing tests**

`tests/test_fragility.py`:

```python
import itertools

import numpy as np
import pytest

from battery.fragility import cross_dataset_agreement, effective_dimensionality, fragility

N = 12


def _matrix(rng, scale=100.0):
    mask = rng.random((N, N)) < 0.5
    upper = np.triu(rng.random((N, N)) * scale * mask, 1)
    return upper + upper.T


def _combo_matrices(n_subjects=8, n_candidates=3, seed=0, noise=0.01):
    """{combo_id: {subject: [rep1, rep2]}} with subject structure and small seed noise."""
    rng = np.random.default_rng(seed)
    bases = {f"sub-{i:03d}": _matrix(rng) for i in range(n_subjects)}
    out = {}
    for c in range(n_candidates):
        shift = 1.0 + 0.3 * c
        out[f"wave1/sweep_{c:04d}"] = {
            s: [b * shift * (1 + noise * rng.standard_normal(b.shape)) for _ in range(2)]
            for s, b in bases.items()
        }
    return out


def test_fragility_finds_seed_agreement_above_parameter_agreement():
    got = fragility(_combo_matrices())
    assert got["seed_rho"] > got["param_rho"]
    assert got["gap"] == pytest.approx(got["seed_rho"] - got["param_rho"])
    assert got["n_measures"] >= 5


def test_fragility_param_rho_undefined_with_one_candidate():
    got = fragility(_combo_matrices(n_candidates=1))
    assert np.isnan(got["param_rho"])


def test_effective_dimensionality_is_between_one_and_n_measures():
    eff = effective_dimensionality(_combo_matrices())
    assert 1.0 <= eff <= 8.0


def test_cross_dataset_agreement_is_one_for_identical_orderings():
    margins = {"dsA": {"sweep_0001": 0.1, "sweep_0002": 0.2, "sweep_0003": 0.3},
               "dsB": {"sweep_0001": 0.5, "sweep_0002": 0.6, "sweep_0003": 0.7}}
    [pair] = cross_dataset_agreement(margins)
    assert pair["rho"] == pytest.approx(1.0)
    assert pair["n_shared"] == 3


def test_cross_dataset_agreement_skips_pairs_with_too_few_shared():
    margins = {"dsA": {"sweep_0001": 0.1, "sweep_0002": 0.2},
               "dsB": {"sweep_0003": 0.5, "sweep_0004": 0.6}}
    assert cross_dataset_agreement(margins) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_fragility.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'battery.fragility'`.

- [ ] **Step 3: Implement `battery/fragility.py`**

```python
"""Graph-measure fragility, redundancy, and cross-dataset ranking agreement.

Fragility contrasts how well the subject ordering on each graph measure
survives re-running an identical specification (seed noise) against how well it
survives changing specification (parameter change). Effective dimensionality
reports how many independent quantities the graph battery actually carries.

Requires OptiConn importable: this module reuses its graph-measure code rather
than reimplementing it.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import spearmanr

from scripts.compute_network_measures_from_connectivity import measures_from_matrix

MIN_SUBJECTS = 3


def _per_subject_measures(combo_matrices: dict) -> dict:
    """{combo_id: {subject: [measures_rep1, measures_rep2]}}, combos with data only."""
    out = {}
    for combo_id, subjects in combo_matrices.items():
        rows = {
            subject: [measures_from_matrix(m) for m in reps[:2]]
            for subject, reps in subjects.items()
            if len(reps) >= 2
        }
        if rows:
            out[combo_id] = rows
    return out


def _measure_names(per: dict) -> list[str]:
    first_subject_reps = next(iter(next(iter(per.values())).values()))
    return sorted(first_subject_reps[0])


def fragility(combo_matrices: dict) -> dict:
    """Mean subject-ordering agreement across seeds vs across candidates."""
    per = _per_subject_measures(combo_matrices)
    if not per:
        return {"seed_rho": float("nan"), "param_rho": float("nan"),
                "gap": float("nan"), "n_measures": 0}
    names = _measure_names(per)

    seed_rhos: list[float] = []
    param_rhos: list[float] = []
    for name in names:
        for subjects in per.values():
            keys = sorted(subjects)
            if len(keys) < MIN_SUBJECTS:
                continue
            rho, _ = spearmanr([subjects[k][0][name] for k in keys],
                               [subjects[k][1][name] for k in keys])
            if not np.isnan(rho):
                seed_rhos.append(float(rho))
        averaged = {
            combo: {k: float(np.mean([v[0][name], v[1][name]])) for k, v in subjects.items()}
            for combo, subjects in per.items()
        }
        for a, b in itertools.combinations(sorted(averaged), 2):
            keys = sorted(set(averaged[a]) & set(averaged[b]))
            if len(keys) < MIN_SUBJECTS:
                continue
            rho, _ = spearmanr([averaged[a][k] for k in keys],
                               [averaged[b][k] for k in keys])
            if not np.isnan(rho):
                param_rhos.append(float(rho))

    seed = float(np.mean(seed_rhos)) if seed_rhos else float("nan")
    param = float(np.mean(param_rhos)) if param_rhos else float("nan")
    return {"seed_rho": seed, "param_rho": param, "gap": seed - param, "n_measures": len(names)}


def effective_dimensionality(combo_matrices: dict) -> float:
    """(sum eigenvalues)^2 / sum(eigenvalues^2) over the z-scored measure battery."""
    per = _per_subject_measures(combo_matrices)
    if not per:
        return float("nan")
    names = _measure_names(per)
    rows = [[float(np.mean([reps[0][n], reps[1][n]])) for n in names]
            for subjects in per.values() for reps in subjects.values()]
    if len(rows) < 3:
        return float("nan")
    x = np.asarray(rows, dtype=float)
    std = x.std(0, ddof=0)
    std[std == 0] = 1.0
    z = (x - x.mean(0)) / std
    eigenvalues = np.linalg.eigvalsh(np.cov(z, rowvar=False))
    eigenvalues = eigenvalues[eigenvalues > 0]
    if eigenvalues.size == 0:
        return float("nan")
    return float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum())


def cross_dataset_agreement(margins_by_dataset: dict) -> list[dict]:
    """Pairwise Spearman agreement of candidate ordering between datasets."""
    out = []
    for a, b in itertools.combinations(sorted(margins_by_dataset), 2):
        shared = sorted(set(margins_by_dataset[a]) & set(margins_by_dataset[b]))
        if len(shared) < MIN_SUBJECTS:
            continue
        rho, _ = spearmanr([margins_by_dataset[a][s] for s in shared],
                           [margins_by_dataset[b][s] for s in shared])
        out.append({"dataset_a": a, "dataset_b": b,
                    "rho": float(rho) if not np.isnan(rho) else float("nan"),
                    "n_shared": len(shared)})
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=/data/local/software/opticonn python -m pytest tests/test_fragility.py -q`
Expected: 5 passed. OptiConn must be importable, hence `PYTHONPATH`; note this in the README's prerequisites.

- [ ] **Step 5: Wire it into `merge.py`**

In `battery/merge.py`, extend `main()` so that after writing `cross_dataset_summary.csv` it also, per dataset:

```python
from scripts.variance_decomposition import collect_sweep_matrices
from battery.fragility import cross_dataset_agreement, effective_dimensionality, fragility

frag_rows, margins_by_dataset = [], {}
for dsid in ids:
    try:
        grouped = collect_sweep_matrices(Path(args.results_dir) / dsid / "optimize")
        combos = grouped.get(("AAL3", "count"), {})
        if not combos:
            logging.warning("%s: no AAL3/count matrices; skipped in fragility", dsid)
            continue
        f = fragility(combos)
        f.update(dataset_id=dsid, effective_dimensionality=effective_dimensionality(combos))
        frag_rows.append(f)
    except Exception as exc:
        logging.warning("%s: fragility skipped (%s)", dsid, exc)
```

Build `margins_by_dataset[dsid] = {sweep_id: mean margin over waves}` by re-reading each dataset's `wave1`/`wave2` `combo_diagnostics.csv` (the same files `dataset_summary` reads), then write `fragility_summary.csv` with columns `dataset_id, seed_rho, param_rho, gap, effective_dimensionality, n_measures` and `cross_dataset_agreement.csv` with columns `dataset_a, dataset_b, rho, n_shared`. Every per-dataset computation is wrapped in `try/except` and logged, so one unreadable dataset cannot abort the merge.

- [ ] **Step 6: Run the whole suite and commit**

Run: `PYTHONPATH=/data/local/software/opticonn python -m pytest tests -q`
Expected: all pass (about 33 tests).

```bash
git add battery/fragility.py tests/test_fragility.py battery/merge.py
git commit -m "feat: fragility, effective dimensionality and cross-dataset agreement

Co-Authored-By: <your model> <noreply@anthropic.com>"
```

---

## After implementation (not part of this plan)

1. Run `python -m battery.survey`, then `python -m battery.select`, and **review `selected_datasets.json`** — the last cheap checkpoint.
2. `./fetch.sh`, then `./run_all.sh` (~31 h), then `python -m battery.merge`.
3. Run the single 7T dataset (`ds003508`) separately and report it as an anecdote.
4. Populate the paper's results section from `cross_dataset_summary.csv`, and compare against the in-house pilot's predictions (parameter/between-subject 0.21-0.44; ordering 0.66-0.75 vs 0.93; effective dimensionality ~1.8).
