# Battery readiness: deterministic tracking, QC, pinned upstreams — design

Status: approved 2026-09-27, not implemented.
Repo: `opticonn-multiverse-battery` (sibling of this checkout). OptiConn itself is unchanged.
Follows the single-dataset pilot (ds000221, completed 2026-09-27 00:00).

## Why

The pilot surfaced one scientific defect and three reproducibility gaps. All four must be
fixed before the full battery runs, because every dataset would otherwise inherit them.

**Tracking repeats measured nothing.** Tracking noise in the pilot was ~0 (median exactly
0). Root cause, established by direct DSI Studio runs on the pinned build:

| run | 1 − r between two runs |
|---|---|
| 1 thread, `--random_seed` 1 vs 2 | 0.000000–0.000001 (0–10 of ~190k endpoints change) |
| 4 threads, same seed twice | 0.019 |
| 4 threads, seed 1 vs 2 | 0.019 |

At one thread DSI Studio tracking is deterministic at parcel resolution; the seed barely
matters. All run-to-run variability comes from multithreaded scheduling. The pilot ran one
thread only by accident (`thread_count` 8 ÷ `--max-parallel` 16, floored to 1). The
in-house pilots' "tracking noise" (4 threads) was scheduling variability.

Decision: **single thread, two repeats**. Single thread makes every result bit-reproducible.
Two repeats are kept, not dropped: they need no OptiConn change (the pilot already ran this
configuration to completion), and they measure the within-subject distance in every combo,
so determinism is a reported result rather than an assumption. Cost: ~2× compute.

**The reference combo never ran.** The battery plan was amended to include a DSI Studio
defaults reference, but `configs/battery.json` never received it.

**No image QC.** The in-house screen used a QSIPrep metric that does not exist for hub data.

**Upstreams not pinned in the repo.** Hub release tags equal the dataset id, not a version.
`fetch.sh` records SHA256 hashes, but in gitignored `data/`, so a reviewer has nothing to
verify a download against. OpenNeuro `participants.tsv` is read live from S3 at run time,
unversioned and uncached, so a later dataset release silently changes the control set.

## 1. Deterministic tracking config (`configs/battery.json`)

- `thread_count: 1` — explicit. Because OptiConn computes per-job threads as
  `max(1, thread_count // max_parallel)`, a base of 1 yields 1 at any `--max-parallel`.
- `reliability.repeats: 2` — explicit.
- `sweep_parameters.reference_candidate` — **DSI Studio's untouched defaults**: every
  tracking parameter set to the value OptiConn's extractor omits from the command line
  (`scripts/extract_connectivity_matrices.py:679-704`), so DSI Studio applies its own
  defaults:

  ```json
  "reference_candidate": {
    "method": 0, "otsu_threshold": 0.6, "fa_threshold": 0.0, "turning_angle": 0.0,
    "step_size": 0.0, "smoothing": 0.0, "min_length": 0, "max_length": 0,
    "track_voxel_ratio": 2.0, "tip_iteration": 0
  }
  ```

  The earlier plan reset only `fa_threshold`, `turning_angle` and `step_size`, inheriting the
  grid's smoothing and length limits — not the untouched setting. Streamline count stays at
  the battery's 50,000 so the comparison is at equal sampling.

  Consequence to report, not hide: DSI Studio's default step size and turning angle are
  randomised per streamline, so the reference's two repeats will differ while the grid's
  do not.

## 2. Pinned container (`run_all.sh`)

The default `DSI_APPTAINER_IMAGE` becomes
`/data/local/software/apptainer_images/dsi_studio/dsi_studio_hou-2026-09-27.sif`. Never
`dsi_studio_latest.sif`, which an automated rebuild repointed between the pilot and the
debugging runs.

## 3. QC screen (`battery/qc.py`, wired into `battery/stage.py`)

DSI Studio's QC runs on the reconstructed QSDR file itself, so no preprocessing pipeline's
output is needed. It requires a **file** path; pointed at a directory it looks only for raw
DWI and reports "no file found".

- `run_qc(fz: Path) -> dict` — boundary. Runs `dsi_studio --action=qc --source=<fz>` in a
  temporary directory and returns `parse_qc_tsv` of the written `qc.tsv`.
- `parse_qc_tsv(text: str) -> dict` — pure. Returns `{"coherence": float, "r2": float}`
  from the columns `Coherence Index` and `R2 (QSDR)`.
- Screen — per dataset, per metric, `scripts.qc_gate.flag_low_outliers` (OptiConn; robust
  z by median/MAD, threshold 3.5, one-sided low). A scan is excluded if it is a low outlier
  on **either** metric: low coherence indicates an orientation or b-table error, low R2 a
  failed template registration.
- `stage` screens candidate control scans before linking them.
- Every scan's metrics and verdict go to a committed `qc/<ds>.tsv`:
  `file, coherence, r2, coherence_z, r2_z, excluded, reason`.

Failure handling:
- DSI Studio fails on a file → that scan is excluded with reason `qc failed`. Never silently
  included.
- A dataset with fewer than 10 scans (`qc_gate.MIN_SCANS`) → recorded as
  `too few scans for robust QC`; nothing is excluded, and the note is carried to the paper.

## 4. Pinned upstreams, committed to git

**Hub files.** `fetch.sh` writes its hashes to tracked `manifests/<ds>.sha256` instead of
gitignored `data/<ds>/manifest.csv`. When a committed manifest exists and a downloaded
file's hash does not match, `fetch.sh` **fails loudly**, naming the file: the hub has
changed it, and the analysis must not proceed on a different file than the one published.
A dataset with no committed manifest is fetched and its manifest written, as today.

**Participant metadata.** A snapshot step (`python -m battery.snapshot`) saves, per
selected dataset, `metadata/<ds>/participants.tsv`, `participants.json` (when present) and
`metadata/<ds>/SOURCE` recording the OpenNeuro dataset id, its latest snapshot tag at the
time of fetching, the fetch date and each file's SHA256. `select`, `stage` and
`demographics` read only the snapshot; if it is missing they fail with a message naming the
command to run, never fall back to live S3.

## Testing (TDD for every new or changed function)

No test needs DSI Studio or network access:
- `parse_qc_tsv` against a fixture `qc.tsv` string, including a malformed row.
- The screen on synthetic values: a planted low outlier on each metric is excluded, a high
  outlier is not, a dataset under 10 scans excludes nothing and carries the note.
- `stage` with `run_qc` stubbed: flagged scans are not linked; a raising `run_qc` excludes
  the scan with reason `qc failed`.
- `configs/battery.json` invariants: `thread_count == 1`, `reliability.repeats == 2`, and
  every `reference_candidate` value equals the extractor's omit value.
- `run_all.sh`: the default image is a dated file, never `latest`.
- `fetch.sh` (dry-run harness): a committed-manifest mismatch exits non-zero naming the file.
- `select`/`stage`/`demographics` read the snapshot and fail clearly without it.

## Deliberately unchanged

- **OptiConn.** Two single-thread repeats run through the existing code; the pilot proved it.
- **`fragility.py`** keeps its between-repeat arm (now ≈ perfectly reproducible, reported).
- **The viewer.**
- **Caching QC results** — ~3 s per file (container start), run once per staging.
  `ponytail:` recompute each time; cache by file hash if staging is re-run often.

## Afterwards

Launch the full battery (8 datasets, ~2× the pilot's ~4.7 h each). While it runs, piece 3:
`reproduce.sh` chaining survey → select → fetch → snapshot → QC + stage → sweep →
aggregate, and the paper's §2.3/§2.4/§5.2 rewritten to describe it and the determinism
finding.
