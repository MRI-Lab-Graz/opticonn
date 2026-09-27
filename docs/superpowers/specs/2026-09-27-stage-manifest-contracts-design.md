# Stage manifest contracts

## Problem

The battery pipeline (`opticonn-multiverse-battery`) chains eight stages —
survey, snapshot, select, fetch, stage, preflight_all, run_all, merge — each
consuming the previous stage's output. Plan B's final review found that
three of these handoffs pass along nothing but a filename and never check
the upstream guarantee:

- `fetch.sh` → `stage.py`: a hub file that changed after the manifest was
  written could still be staged and swept.
- `dsi_preflight.py` → `run_all.sh`: a preflight run against a stale build
  or a different config was not detected before a sweep started.
- `run_all.sh` → `merge.py`: an unverified/pre-verification dataset's
  results could still feed the paper's fragility and agreement tables.

All three were patched, but each fix reinvented its own hash/compare logic,
and each had its own bug on the way in (CRLF-normalized hashing in
`snapshot.py`, regex-vs-literal filename matching in `fetch.sh`, no image
check at all in `verification.py`). One-off checks at each boundary are
where "translation" bugs keep entering: every boundary re-derives what
"correct" means for what it was handed.

## Goal

Every stage-to-stage handoff going forward uses one shared, tested contract
format instead of a bespoke check: the producer writes a manifest of what
it produced and any facts the consumer must verify; the consumer's first
action is to verify that manifest before trusting the output. This is a
library, not a retrofit of every existing boundary — only the three
boundaries with an actual bug history migrate now; the pattern becomes the
standing rule for any boundary added later.

## Design

### The manifest format

One JSON file, `manifest.json`, written into the stage's output directory:

```json
{
  "stage": "fetch",
  "produced_at": "2026-09-27T14:03:00Z",
  "outputs": {"sub-01.qsdr.fz": "<sha256 hex>", "sub-02.qsdr.fz": "<sha256 hex>"},
  "contract": {"hub_release_tag": "1.0.0"}
}
```

- `outputs`: every file the stage produced that a consumer might depend on,
  keyed by path relative to the manifest, valued by its sha256. Verifying
  re-hashes every listed file — a changed byte anywhere is caught.
- `contract`: free-form key/value facts specific to that boundary (a pinned
  image string, a config hash, a release tag) that the consumer checks
  against its own expectation. Empty `{}` when a boundary has none.

### `opticonn/manifest.py`

```python
def write_manifest(dir: Path, stage: str, outputs: dict[str, Path],
                    contract: dict) -> Path:
    """Hash every path in `outputs`, write manifest.json into `dir`, return its path."""

def verify_manifest(dir: Path, expected_contract: dict) -> list[str]:
    """Load dir/manifest.json and check it. Returns a list of problems;
    empty means OK. Never raises for an expected failure:
    - manifest.json missing -> ["manifest.json is missing in <dir>"]
    - manifest.json is not valid JSON -> ["manifest.json in <dir> is not valid JSON"]
    - a listed output is missing -> ["<path> is missing"]
    - a listed output's hash doesn't match -> ["<path> does not match its manifest hash"]
    - a contract key differs from expected -> ["contract.<key> is '<got>', expected '<want>'"]
    A contract key present in `expected_contract` but absent from the
    manifest's `contract` counts as a mismatch (fails closed, not silently
    skipped).
    """
```

No new runtime dependency — this is `hashlib`, `json`, `pathlib`, stdlib
only, matching every other module in both repos.

### Migration (only these three boundaries; everything else is out of scope)

**1. `fetch.sh` → `battery/stage.py`.** `fetch.sh` is bash; it already
computes each file's sha256 (`sha256sum *.qsdr.fz`). Instead of writing the
plain-text `manifests/<dsid>.sha256` it currently writes, it shells out to
`python -m opticonn.manifest write <data-dir> --stage fetch --contract
hub_release_tag=<tag>` (a thin CLI wrapper added to `manifest.py`, `write`
subcommand takes `--stage`, repeatable `--contract key=value`, and hashes
every file already present in the directory as `outputs`). `stage.py`
replaces its `_manifest_hashes` parser with one call to `verify_manifest`
before staging any candidate; a non-empty problem list excludes the whole
dataset with the reason recorded, same as today.

**2. `dsi_preflight.py` → `verification.py`/`run_all.sh`.** `preflight.json`
already carries `passed`, `dsi_apptainer_image`, `expected_fingerprints` —
it becomes a `manifest.json`-shaped file with `contract = {"dsi_apptainer_image": ...}`
(no `outputs` — a preflight has nothing file-shaped for a consumer to
re-hash; `passed` becomes a `contract` key too, so `verify_manifest`'s
generic checks are the only thing `preflight_passed` needs).
`verification.preflight_passed(path, expected_image)` becomes a thin
wrapper: `verify_manifest(path.parent, {"passed": True, "dsi_apptainer_image": expected_image}) == []`.
This deletes the bespoke field-by-field comparison written during the
Plan B fix wave.

**3. `run_all.sh` → `battery/merge.py`.** Left unchanged. The per-run
`<prefix>.dsi_execution.json` (Plan A) and dataset-level `verification.audit()`
already are this exact pattern at a finer grain — one manifest per run,
aggregated per dataset. No rework; noted here so a future reader doesn't
wonder why it wasn't migrated.

### Out of scope

`survey→snapshot` (`SOURCE.json` is already manifest-shaped and has no bug
history), `snapshot→select`, `select→fetch` are not touched. Retrofitting a
boundary with no history of a translation bug is churn, not the goal here.
Any boundary added to the pipeline after this design must use
`opticonn/manifest.py` rather than write its own check — that is the
standing rule this design establishes.

### Error handling

`verify_manifest` never raises for an expected failure; it returns
problems. Every caller already has, or gets, the same fail-closed shape
used throughout Plan B: a CLI stage (`fetch.sh`, `run_all.sh`) prints the
problems and exits non-zero; a batch stage (`stage.py`) records the dataset
as excluded with the reason and stages nothing for it.

### Testing

`opticonn/manifest.py` gets direct unit tests for `write_manifest` +
`verify_manifest`: a clean round trip, a tampered file (hash mismatch), a
missing file, a missing manifest, and a contract-key mismatch — one test
per failure mode, each asserting the exact problem string. The three
migrated boundaries keep their existing battery-repo tests unchanged where
those tests assert behavior (a changed hub file is still refused; a
stale-image preflight is still refused) — only the implementation under
them changes — plus each gets one new test confirming its call site
actually reaches `verify_manifest` (not just that the standalone function
works).

## Self-review

- Placeholders: none.
- Consistency: the `contract` field's "absent key on the manifest side
  fails closed" rule matches Plan A's philosophy (`null` in a config means
  DSI Studio's default, but an *unset* verification check has always meant
  "not proven," never "assumed fine").
- Scope: bounded to one new module plus two migrations; the third boundary
  is explicitly declared already-conformant, not deferred work.
- Ambiguity: the CLI wrapper's exact flag syntax (`--contract key=value`,
  repeatable) is specified so an implementer doesn't have to guess it.
