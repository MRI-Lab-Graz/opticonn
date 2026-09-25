# Cross-protocol multiverse battery on public OpenNeuro data

Date: 2026-09-25
Status: Approved design, spec awaiting review
Supplies: the results section of `docs/papers/2026-09-25-multiverse-methods-paper.md`

## Problem

The paper's evidence must be **reproducible by any reader**, which private cohorts cannot be.
A reader of the current draft can check the method but not a single number.

This battery therefore supplies the paper's results outright, using only publicly available
data: every figure traces to a public dataset id and a pinned release tag, and the whole
analysis re-runs from a clone. It also dissolves the draft's principal limitation — the
site/scanner confound — because the datasets span vendors and protocols by construction
rather than by luck.

The in-house cohorts (three-shell, two-shell) are retained in the paper only as the pilot
that motivated the design and validated the implementation. Their measured magnitudes —
parameter effect 0.21-0.44 of the between-subject difference, subject-ordering agreement
0.66-0.75 against 0.93 for re-runs, effective dimensionality ~1.8 — become **predictions**
for this battery to confirm or refute. Disagreement is itself a reportable result.

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

**Survey reconnaissance (2026-09-25).** The full survey was run before this spec was
finalised: 165 hub releases, 132 resolved (80%), 76 eligible (>=20 subjects, known vendor,
human, parseable bval). Failures were benign — 18 releases carry no `.fz` assets, 11 had no
findable DWI sidecar.

Metadata availability, measured over 18 datasets x 3 subjects:

| Field | Coverage | Role |
| --- | --- | --- |
| Manufacturer, EchoTime, RepetitionTime | 94% | vendor is a stratum; TE/TR covariates |
| Model, MagneticFieldStrength, PhaseEncodingDirection, FlipAngle | 88% | covariates |
| TotalReadoutTime | 83% | covariate |
| **InstitutionName** | **72%** | **site count — supports the generalisation claim** |
| SoftwareVersions, SequenceName, SliceThickness, AcquisitionMatrixPE | 61-72% | covariates |
| ReceiveCoilName | 50% | reported where present, never a stratum |
| Parallel/multiband factors | 38-44% | reported where present |

Two findings changed the design:

1. **Field strength has no variance to stratify on.** Among eligible datasets 66 are 3T, 9
   unknown and exactly **one** is 7T. Public OpenNeuro DWI is a 3T corpus.
2. **Protocol is not always constant within a study.** Sampling 3 subjects per dataset,
   2 of 18 (11%) varied — `ds003508` (EchoTime 0.073/0.073496/0.087) and `ds003138`
   (0.104/0.113/0.125). A single-subject survey would have mischaracterised roughly one
   dataset in nine.

## Architecture

One study repository, pinning OptiConn by commit. Six small stages, each independently
runnable:

```
list_datasets.py  gh release list across hub repos   -> datasets.csv (id, repo, tag, n_subjects)
survey.py         2 HTTPS GETs per dataset           -> protocol_table.csv
select.py         stratify + pick                    -> selected_datasets.json   [REVIEW GATE]
fetch.sh          gh release download                -> data/<dsid>/*.fz
run_all.sh        sequential sweep per dataset       -> results/<dsid>/...
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

For each dataset, sample **three** subjects (deterministic: first three by asset order) and
fetch each one's `_dwi.json` and `_dwi.bval`. Sampling three rather than one is required: the
reconnaissance found 11% of datasets vary their protocol across subjects. A dataset whose
sampled subjects disagree on manufacturer, field strength, coil, TE or TR is flagged
`protocol_varies` and excluded from selection, with the differing values recorded.

Recorded per dataset:

| Field | Source | Reliability |
| --- | --- | --- |
| `manufacturer`, `model` | sidecar | 94% / 88% |
| `institution` | sidecar | 72% — report distinct site count |
| `field_strength` | sidecar | 88%, but ~no variance (3T corpus) |
| `coil` | sidecar | 50% — reported, not stratified |
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

**Primary strata: `manufacturer` x `scheme`.** Field strength is *not* a stratum: the survey
found only one 7T dataset among 76 eligible, so there is no variance to stratify on. It is
recorded as a covariate. Coil type is reported where present (50%) but never used for
selection.

**Scope restricted to 3T.** The corpus is effectively all 3T. The single 7T dataset
(`ds003508`, Philips multi-shell) is run separately and reported as an anecdote — "one 7T
dataset, insufficient for inference" — never pooled with the 3T results. It also happens to
be one of the two datasets with a varying within-study protocol, which is a second reason to
keep it apart.

Eligibility: human brain, 3T, >= 20 subjects in the hub, parseable bval, protocol constant
across the 3 sampled subjects.

Measured cell occupancy (7 of 9 cells; Philips-free and GE-free appear not to exist publicly):

| | single-shell | multi-shell | free q-space |
| --- | --- | --- | --- |
| Siemens | 28 (max n=290) | 22 (max n=195) | 3 (max n=211) |
| Philips | 13 (max n=464) | 3 (max n=161) | - |
| GE | 6 (max n=241) | 1 (max n=132) | - |

Because only 7 cells are occupied, the rule takes the **two** largest eligible datasets per
cell where available, giving ~13 datasets. Within-cell replication is a feature, not padding:
it separates a protocol effect from a single dataset's idiosyncrasy.

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

## Component 5 — Sweep runner

No scheduler. One dataset sweep measured 2.6 h at `--max-parallel 4` (16 of 32 cores), so
twelve run sequentially in ~31 h, or ~16 h two-at-a-time on a 32-core machine — an overnight
job. `run_all.sh` loops over the selected datasets, invoking the existing two-wave
`tune-grid` into a per-dataset directory; each is independent and re-runnable after failure,
and the loop skips datasets whose results already exist.

Dropping the scheduler removes a component, removes the need for a cluster to test it, and
removes a barrier for anyone replicating the study. If a future battery is large enough to
need one, the same per-dataset invocation is what a job array would call anyway.

Held identical to the 129/134 sweeps, so results are directly comparable:

- the same 12-candidate grid (FA threshold x turning angle x track/voxel ratio)
- `tract_count = 50000` (shown to preserve candidate ordering against 5k, and to keep the
  tracking-noise share of the parameter effect below half)
- 10 subjects per wave, two waves, 2 repeats
- AAL3, `count`/`fa`/`qa` connectivity

```
while read dsid; do
  [ -e results/$dsid/optimize/optimization_results ] && continue   # idempotent
  opticonn tune-grid -i data/$dsid -o results/$dsid \
    --extraction-config configs/battery.json --max-parallel 4
done < selected_ids.txt
```

`run_all.sh` takes `--dry-run`, and a single dataset can be run on its own for testing.

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
- `run_all.sh`: `--dry-run` prints the commands; idempotent skip verified on a populated results dir.

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
- Any scheduler support, in OptiConn or the battery: measured runtime makes a cluster
  unnecessary, and a sequential local run is easier for a reader to reproduce.
- MRtrix3 cross-check; that remains a separate follow-up.
