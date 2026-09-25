# Cross-protocol multiverse battery on public OpenNeuro data

Date: 2026-09-25
Status: Approved design, spec awaiting review
Supports: `docs/papers/2026-09-25-multiverse-methods-paper.md` §5.3 (the draft's principal limitation)

## Problem

Every result in the methods draft comes from cohorts sharing one site, one scanner and one
population. The claim that parameter dependence has a characteristic magnitude — a fifth to
a half of a between-subject difference, a quarter to a third of graph-measure subject
ordering — cannot be defended from within-site replication alone. The battery exists to test
those magnitudes across protocols that differ in vendor, field strength, shell scheme,
direction count and resolution, using **only publicly available data** so the entire result
is reproducible by anyone.

## Two findings that shaped the design

Both were verified before this spec was written.

**1. The Fiber Data Hub removes the preprocessing subsystem.**
`brain.labsolver.org` (Yeh, 2025) publishes per-subject, QSDR-reconstructed `.fz` files for
public datasets via GitHub releases under `data-openneuro/{brain,brain2,disease,disease2,
others}`, `data-hcp/*` and others. Files are BIDS-named (`sub-146_ses-wave1_dwi.qsdr.fz`),
~1-2 MB each; a single release held 787 subjects, and `brain`+`brain2` alone carry 124
dataset releases.

A downloaded file was tracked end to end with the project's existing DSI Studio invocation
and AAL3 parcellation, producing a valid connectivity matrix in 20 seconds. No QSIPrep and
no reconstruction are required — eliminating what would otherwise have been CPU-months — and
because every file uses QSDR, reconstruction is held constant across datasets instead of
varying confoundingly with protocol.

**2. No crawler is needed.**
The initial plan was to extend `MRI-Lab-Graz/openneuro-crawler`. Its purpose is dataset
*discovery*, and the hub's release tags already supply the dataset list. Protocol metadata
comes from two plain HTTPS GETs against OpenNeuro's public S3 bucket:

```
https://s3.amazonaws.com/openneuro.org/<dsid>/<sub>/<ses>/dwi/<file>_dwi.json   -> vendor, TE, TR, ...
https://s3.amazonaws.com/openneuro.org/<dsid>/<sub>/<ses>/dwi/<file>_dwi.bval   -> shell scheme
```

Verified on ds004856: `Manufacturer: Philips`, `EchoTime: 0.051`, `RepetitionTime: 4.41`,
shells `b0 x1, b1000 x30`. This replaces a GraphQL crawl and a 1.8 GB metadata clone with
about 250 small requests. The crawler is therefore **not modified and not used**; it would
only return if datasets outside the hub's coverage were ever needed.

**Caveat found in the same probe:** that sidecar contained no `MagneticFieldStrength` and no
`ReceiveCoilName`. BIDS sidecars are inconsistently populated, which constrains the
stratification below.

## Architecture

One study repository, pinning OptiConn by commit. Six small stages, each independently
runnable:

```
list_datasets.py  gh release list across hub repos   -> datasets.csv (id, repo, tag, n_subjects)
survey.py         2 HTTPS GETs per dataset           -> protocol_table.csv
select.py         stratify + pick                    -> selected_datasets.json   [REVIEW GATE]
fetch.sh          gh release download                -> data/<dsid>/*.fz
slurm/sweep.sbatch  array, 1 task per dataset        -> results/<dsid>/...
merge.py          aggregate per-dataset reports      -> cross_dataset_summary.csv
```

## Component 1 — Dataset inventory

`gh release list` over `data-openneuro/{brain,brain2,disease,disease2,others}`. Each release
tag is an OpenNeuro dataset id, optionally suffixed (`ds004856_4`); the id is the tag up to
the first underscore. Subject count comes from the release's asset count, and the asset list
also yields the subject/session identifiers without downloading anything.

Output `datasets.csv`: `dataset_id, hub_repo, release_tag, n_assets, n_subjects`.

Animal collections are excluded by repository name. Human disease cohorts are retained —
pathology is part of the protocol diversity we are testing.

## Component 2 — Protocol survey

For each dataset, pick one representative scan (first asset, deterministic), derive its
OpenNeuro path, and fetch its `_dwi.json` and `_dwi.bval`. Recorded per dataset:

| Field | Source | Reliability |
| --- | --- | --- |
| `manufacturer`, `model` | sidecar | usually present |
| `field_strength` | sidecar | **often absent** |
| `echo_time`, `repetition_time` | sidecar | usually present |
| `multiband_factor`, `pe_direction` | sidecar | variable |
| `shells`, `n_directions`, `max_b` | bval | **always derivable** |
| `scheme` | derived from shells | always |

`scheme` is `single-shell` (one non-zero b), `multi-shell` (2-4 distinct non-zero b), or
`free` (5+ distinct non-zero b, i.e. q-space sampling).

Missing fields are recorded as `unknown`, never imputed. Fetch failures are recorded with
the HTTP status and the dataset is carried forward with whatever was retrieved; a dataset
lacking a usable bval is excluded with that reason.

Because the sidecar path must be derived from an asset name, datasets whose OpenNeuro layout
does not match the derived path (entity ordering, `run-`/`acq-` variations) will fail the
fetch. The survey therefore tries the S3 prefix listing for that subject's `dwi/` directory
as a fallback before giving up, and reports the count of datasets resolved by each route.

## Component 3 — Stratified selection

**Primary strata: `manufacturer` x `scheme`.** Both are reliably derivable, the second
always. Field strength enters only as a tie-break where present, because the probe shows it
cannot be relied on as a stratum; if the survey finds it present for most datasets, it is
promoted to a third stratum and the spec's assumption is revisited. Coil type is not used:
too rarely reported.

Eligibility: human brain, >= 20 subjects in the hub, parseable bval, reconstruction present.

Rule: within each occupied cell take the dataset with the most subjects. Fill cells
**rarest-first** (cells with fewest eligible datasets first), since rare combinations buy the
most protocol spread, and stop at 12. Ties broken by dataset id, ascending, for determinism.

`selected_datasets.json` records the chosen datasets, their stratum, every eligible
alternative per cell, and an exclusion reason for every dataset not chosen — so the selection
is auditable rather than curated.

**Review gate.** The selection is inspected before any download or compute. This is the last
cheap moment: everything after it costs cluster time.

## Component 4 — Acquisition

`gh release download <tag> -R <hub_repo> --pattern "*.fz"` into `data/<dataset_id>/`.
All subjects are downloaded; OptiConn's existing seeded sampling selects which enter a wave,
so subject selection needs no second mechanism and stays reproducible.

`fetch.sh` records the release tag and a SHA256 of each file in `data/<dataset_id>/
manifest.csv`. Re-running is idempotent (skips files already present with a matching hash).

## Component 5 — SLURM sweep

One array task per dataset. Each task runs the existing two-wave `tune-grid` into its own
directory, so tasks share nothing and any single task can be re-run after failure.

Held identical to the 129/134 sweeps, so results are directly comparable:

- the same 12-candidate grid (FA threshold x turning angle x track/voxel ratio)
- `tract_count = 50000` (shown to preserve candidate ordering against 5k, and to keep the
  tracking-noise share of the parameter effect below half)
- 10 subjects per wave, two waves, 2 repeats
- AAL3, `count`/`fa`/`qa` connectivity

```
#SBATCH --array=0-11
dsid=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" selected_ids.txt)
opticonn tune-grid -i data/$dsid -o results/$dsid --extraction-config configs/battery.json ...
```

Resources per task are set from the measured local runtime (~2.6 h at `--max-parallel 4`)
with headroom. The script takes `--dry-run` and runs unmodified on one dataset locally,
which is how it is tested without a cluster.

## Component 6 — Cross-dataset merge

`merge.py` reads each dataset's existing OptiConn outputs — `combo_diagnostics.csv`,
`variance_decomposition.csv`, `graph_icc.csv` — and computes per dataset:

- whether discriminability saturated (all candidates at 1.000)
- cross-wave rank agreement of the margin, and whether the wave winners agree
- top-two separation against wave-to-wave variation (the "determined vs not" test)
- `parameter / between_subject` per connectivity metric
- seed-vs-parameter subject-ordering agreement, and effective dimensionality of the graph
  battery

Output `cross_dataset_summary.csv`, one row per (dataset, connectivity metric), plus a
cross-dataset rank agreement of candidate ordering between every dataset pair — the
consensus-versus-per-dataset-optimisation question, now answered across protocols rather
than across two cohorts from one site.

## Error handling

- Every exclusion carries a machine-readable reason; nothing is dropped silently.
- A failed shard leaves its directory incomplete and is re-run individually; `merge.py`
  reports datasets with missing outputs rather than omitting them.
- Datasets whose sweep completes but whose statistics are unavailable (for example fewer
  than 5 candidates shared across waves) are reported as unavailable with the count.

## Testing

No cluster and no network needed for CI:

- `survey.py` parsing: synthetic sidecar/bval fixtures, including a sidecar missing
  `MagneticFieldStrength` and a bval that is single-shell, multi-shell and free.
- `select.py`: golden test — a fixed `protocol_table.csv` produces a fixed
  `selected_datasets.json`, proving the rule is deterministic.
- `merge.py`: fixture OptiConn outputs for two datasets produce the expected summary rows.
- `sweep.sbatch`: `--dry-run` prints the commands for one dataset.

## Reproducibility

OptiConn pinned by commit; hub release tags and per-file SHA256 recorded; deterministic
selection rule with all alternatives logged; one documented command sequence from clone to
`cross_dataset_summary.csv`.

## Limitation to state in the paper

Using hub-provided `.fz` inherits its preprocessing and QSDR reconstruction. The multiverse
explored is therefore the **tracking** multiverse with preprocessing held fixed — a narrower
claim than "all analytic choices", and it must be written as such. The compensating benefit
is that reconstruction is constant across all datasets, so protocol effects are not confounded
with reconstruction differences.

## Out of scope

- Modifying or using `openneuro-crawler` (its discovery role is unnecessary here).
- Varying preprocessing or reconstruction (would require raw data and CPU-months).
- Non-human data; the animal collections are excluded.
- Any scheduler support inside OptiConn.
- MRtrix3 cross-check; that remains a separate follow-up.
