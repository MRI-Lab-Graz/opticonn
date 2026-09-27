# Battery readiness: deterministic tracking, QC, pinned upstreams — design

Status: approved 2026-09-27, amended same day (§0 added after command verification);
not implemented.
Repos: OptiConn (§0 only) and `opticonn-multiverse-battery` (§1-§5).
Follows the single-dataset pilot (ds000221, completed 2026-09-27 00:00).

## 0. OptiConn: DSI Studio must do what the config says

DSI Studio echoes the effective value of every tracking parameter it applies (`├──key=value`
lines on stdout). Comparing that echo against what OptiConn sends, on the pinned build:

| intended | what DSI Studio actually did |
|---|---|
| `--connectivity_threshold=0.001`, sent on every run | "❗ not used/recognized" — ignored; every connectome so far is unthresholded |
| `dt_threshold: 0.2` in `battery.json` | never sent by the extractor; effective 0 |
| `track_voxel_ratio` (assumed default 2.0) | derived from `tract_count` (0.7385 on one subject); not a free parameter when `tract_count` is set |

The extractor decides what to send by comparing each value with a hard-coded table of
assumed DSI Studio defaults, omitting matches (`extract_connectivity_matrices.py:679-704`).
That is how a default can drift silently between builds. It also discards DSI Studio's
stdout on success and deletes the tract file, so no run records what DSI Studio applied.

Changes, all in `scripts/extract_connectivity_matrices.py`, the chokepoint every DSI Studio
tracking call goes through:

1. **Explicit intent, no assumed defaults.** Delete the default table. A tracking parameter
   set to `null` means "DSI Studio's own default" and is omitted; any other value is always
   sent. Known keys: `method, otsu_threshold, fa_threshold, turning_angle, step_size,
   smoothing, min_length, max_length, track_voxel_ratio, check_ending, tip_iteration,
   threshold_index, random_seed`. Any other key in `tracking_parameters` (e.g.
   `dt_threshold`) is a configuration error, never silently dropped.
2. **Drop `--connectivity_threshold`.** Matrices are unthresholded, matching every run so
   far; the density gates still reject implausible graphs. A config that sets
   `connectivity_threshold` or sweeps `connectivity_threshold_range` is rejected with a
   message saying the option is not recognised by DSI Studio.
3. **Verify the echo.** After each run, parse the echoed parameters (ANSI stripped). Fail the
   run if stdout contains "not used/recognized" for any flag, or if a sent tracking
   parameter, `tract_count`, `thread_count` or `random_seed` echoes a different value
   (numeric comparison). Omitted (`null`) parameters are recorded, not compared. Paths and
   connectivity options are recorded but not compared, since DSI Studio echoes them in
   rewritten form.
4. **Record every run.** Write `dsi_command.txt` and `dsi_effective_params.json` (every
   echoed parameter plus the build string) next to each connectivity matrix.

Tests (TDD, no DSI Studio): the echo parser against captured stdout fixtures (the two probe
runs of 2026-09-27); command building from `null` vs literal values; rejection of unknown
keys and of `connectivity_threshold`; verification failing on an unrecognised-option line and
on a changed value, passing on a faithful echo.

This supersedes "OptiConn is unchanged" below: the tracking-repeat decision still needs no
OptiConn change; §0 does.

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
  tracking parameter `null`, which under §0 means "omit, let DSI Studio decide":

  ```json
  "reference_candidate": {
    "method": null, "otsu_threshold": null, "fa_threshold": null, "turning_angle": null,
    "step_size": null, "smoothing": null, "min_length": null, "max_length": null,
    "tip_iteration": null
  }
  ```

  The echo on the pinned build shows what that resolves to: Otsu 0.6, FA 0 (Otsu-based),
  turning angle 0 and step size 0 (both randomised per streamline), smoothing 0, min length
  30, max length 200, Euler, no pruning. These are recorded per run by §0, so a later build
  that changes a default is visible rather than silent. The earlier plan reset only
  `fa_threshold`, `turning_angle` and `step_size`, inheriting the grid's smoothing and
  length limits — not the untouched setting. Streamline count stays at the battery's
  50,000 so the comparison is at equal sampling.

  Consequence to report, not hide: the default step size and turning angle are randomised
  per streamline, so the reference's two repeats will differ while the grid's do not.
- Remove `tracking_parameters.dt_threshold` (never applied) and leave
  `track_voxel_ratio` unset (derived from `tract_count`); §0 would reject both otherwise.

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

## 5. Preprocessing status as a recorded moderator

The hub applies topup and eddy only where an acquisition supports them (both / eddy only /
topup only / neither — two datasets each in the current selection). A pure
`preprocessing_steps(report: str) -> dict` reads each `.fz` file's own reconstruction
report and returns `{"topup": bool, "eddy": bool, "bias_field": bool}`; the per-dataset
result is recorded in the protocol table and feeds Table 1 and §3.7 of the paper as a
**moderator, reported descriptively**. With eight datasets it is confounded with site,
vendor and protocol, and the paper says so; it cannot establish what topup or eddy cause.

## Testing (TDD for every new or changed function)

No test needs DSI Studio or network access:
- `parse_qc_tsv` against a fixture `qc.tsv` string, including a malformed row.
- The screen on synthetic values: a planted low outlier on each metric is excluded, a high
  outlier is not, a dataset under 10 scans excludes nothing and carries the note.
- `stage` with `run_qc` stubbed: flagged scans are not linked; a raising `run_qc` excludes
  the scan with reason `qc failed`.
- `configs/battery.json` invariants: `thread_count == 1`, `reliability.repeats == 2`, every
  `reference_candidate` value is `null`, and no `dt_threshold`, `track_voxel_ratio` or
  `connectivity_threshold` anywhere.
- `preprocessing_steps` against report strings from each current dataset (both, eddy only,
  topup only, neither).
- `run_all.sh`: the default image is a dated file, never `latest`.
- `fetch.sh` (dry-run harness): a committed-manifest mismatch exits non-zero naming the file.
- `select`/`stage`/`demographics` read the snapshot and fail clearly without it.

## Deliberately unchanged

- **OptiConn, beyond §0.** Two single-thread repeats run through the existing code; the
  pilot proved it.
- **A controlled eddy/topup comparison** (reprocessing raw OpenNeuro DWI with and without
  the corrections). Considered and not chosen; preprocessing enters only as the moderator
  in §5.
- **`fragility.py`** keeps its between-repeat arm (now ≈ perfectly reproducible, reported).
- **The viewer.**
- **Caching QC results** — ~3 s per file (container start), run once per staging.
  `ponytail:` recompute each time; cache by file hash if staging is re-run often.

## Afterwards

Launch the full battery (8 datasets, ~2× the pilot's ~4.7 h each). While it runs, piece 3:
`reproduce.sh` chaining survey → select → fetch → snapshot → QC + stage → sweep →
aggregate, and the paper's §2.3/§2.4/§5.2 rewritten to describe it and the determinism
finding.
