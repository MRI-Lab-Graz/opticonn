# Battery readiness: deterministic tracking, QC, pinned upstreams — design

Status: approved 2026-09-27, amended same day (§0 added after command verification);
not implemented.
Repos: OptiConn (§0 only) and `opticonn-multiverse-battery` (§1-§5).
Follows the single-dataset pilot (ds000221, completed 2026-09-27 00:00).

## 0. OptiConn: the parameters sent must be the parameters executed

**Mandatory.** Every connectivity matrix in the battery must come from a DSI Studio run whose
executed parameters are proven equal to the intended ones. A run that cannot be proven is a
failed run.

### What went wrong, and why

Checked against the pinned build (27 Sep) and DSI Studio's own documentation
(`doc/cli_t3.html`), which agrees with every execution record below:

| intended | executed |
|---|---|
| `--connectivity_threshold=0.001` on every run | not a parameter (absent from the docs); "❗ not used/recognized", exit 0. Every connectome so far is unthresholded |
| `dt_threshold: 0.2` in `battery.json` | never sent; a differential-tractography parameter that needs `dt_metric1/2` |
| `track_voxel_ratio` | only applies "when counts not fixed"; derived from `tract_count` |
| `otsu_threshold` 0.6 / 0.8, `fa_threshold` 0 | documented as the **centre** of a window: threshold randomised over 0.5–0.7 / 0.7–0.9 × Otsu |
| `fa_threshold` 0.1 | fixed 0.1 |
| `turning_angle` unset (0) | randomised 45°–90° |
| `step_size` unset (0) | voxel spacing |
| `min_length`/`max_length` unset | "template/image dependent"; 30/200 mm on the QSDR template |

Five mechanisms combined: DSI Studio accepts unknown options and exits 0; some parameter
values select a strategy rather than a value (documented, but not reflected in the
parse-time echo); defaults are template-dependent and DSI Studio is a rolling release; some
parameters are derived; and OptiConn declared success on exit code plus file existence,
omitted values matching a stale table of assumed defaults, discarded stdout, and deleted the
tract file — the only record of what was executed.

### Sources of evidence, established on the pinned build

1. **Stdout echo** (`├──key=value`): what DSI Studio parsed. Not execution evidence — it
   prints `otsu_threshold=0.6` for a randomised window.
2. **Tract-file `report`**: DSI Studio's prose record of what it executed. Shows threshold
   windows, fixed/randomised angle, step, smoothing, length bounds, tract count, pruning
   iterations. Does **not** distinguish Euler from Runge–Kutta (identical wording).
3. **Tract-file `parameter_id`**: an encoded fingerprint of the executed parameters. Changes
   with every parameter varied, including the algorithm; identical settings give an
   identical id.
4. **Streamline geometry** from a direct `.trk` export: step length, turning angle, length,
   count — exact (a `.tt.gz` converted afterwards is quantised to ~0.03 mm, so it is not
   used for proof).
5. **Differential output**: tracking is deterministic at one thread, so a parameter that
   executes must change the output. Proven for the algorithm (RK4 tracks differ from Euler)
   and pruning (tract data 7.1 MB → 2.8 MB).

### Changes, in `scripts/extract_connectivity_matrices.py` (the chokepoint for every tracking call)

1. **Explicit intent.** Delete the table of assumed defaults. A tracking parameter set to
   `null` means "DSI Studio's own default" and is omitted; any other value is always sent.
   Known keys: `method, otsu_threshold, fa_threshold, turning_angle, step_size, smoothing,
   min_length, max_length, check_ending, tip_iteration, threshold_index, random_seed`. Any
   other key (`dt_threshold`, `track_voxel_ratio` with `tract_count` set) is a configuration
   error. `--connectivity_threshold` is never sent; a config setting it, or sweeping
   `connectivity_threshold_range`, is rejected.
2. **Keep the tract file until verified**, then archive `report` and `parameter_id` and
   delete it as today.
3. **Verify every run (fail closed):**
   - *Echo, positive confirmation:* every flag sent must appear in the echo. A sent flag that
     is not echoed back is treated as not executed → failure naming the flag. The
     "not used/recognized" line is a second trigger, not the only one, so a build that
     rewords or drops the warning is still caught. No echo block found → failure.
   - *Echo values:* each sent value must echo unchanged (numeric comparison).
   - *Execution report:* each intended parameter must map to its expected executed
     statement (table below). A statement that is missing, different, or unparseable →
     failure. Omitted (`null`) parameters are recorded as executed, not compared.
   - *Fingerprint:* `parameter_id` must equal the expected id recorded by the preflight for
     that (specification, repeat).
4. **Record every run:** `dsi_command.txt` and `dsi_execution.json` (echo, parsed report,
   `parameter_id`, DSI Studio build string) next to each connectivity matrix.

Expected executed statements (pinned build; any mismatch fails):

| intended | expected in `report` |
|---|---|
| `fa_threshold` f > 0 | "The anisotropy threshold was f." |
| `fa_threshold` 0, `otsu_threshold` t | "…randomly selected between t−0.1 and t+0.1 otsu threshold." |
| `turning_angle` a > 0 | "The angular threshold was a degrees." |
| `step_size` s > 0 | "The step size was s mm." |
| `smoothing` m > 0 | "…with 100·m% of the previous direction." |
| `min_length` l, `max_length` u | "…shorter than l or longer than u mm were discarded." |
| `tract_count` n | "A total of n tracts were tracked." |
| `tip_iteration` k > 0 | "…applied … with k iteration(s)…" |
| `method` | not in the report → proven by fingerprint and differential output |

Numbers are compared numerically (the report writes `10.00` and `30.0` for the same kind of
value).

### Preflight gate (before any battery launch)

For every distinct specification (72 grid + reference), one run per repeat on one subject:
all per-run checks, plus
- geometry from a direct `.trk` export: step equal to `step_size`, no turn above
  `turning_angle` where fixed, lengths within bounds (DSI Studio counts points × step, one
  step more than the segment sum), streamline count equal to `tract_count`;
- differential output: each grid axis changes the output when only it changes.

The preflight writes the expected `parameter_id` per (specification, repeat). The battery
does not start unless the preflight passes completely. Minutes per specification, run once
per pinned build; a new build means a new preflight.

### Tests (TDD, no DSI Studio)

Fixtures from the 2026-09-27 probes (stdout and `report`/`parameter_id` for grid, Otsu 0.8,
FA 0.1, RK4, pruning 2, reference): echo parser; positive-confirmation failure when a sent
flag is absent from the echo, and when the warning line is present; report parser and the
expected-statement table (passing on each fixture, failing on a changed number, a missing
sentence, an unparseable report); command building from `null` vs literals; rejection of
unknown keys and of `connectivity_threshold`; fingerprint mismatch failing; geometry checks
on a small synthetic `.trk` (a step, a turn, a length each just outside its bound).

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

  DSI Studio's execution report on the pinned build shows what that resolves to: anisotropy
  threshold randomised over 0.5–0.7 × Otsu, turning angle randomised over 45°–90°, step size
  equal to voxel spacing, no smoothing, lengths 30–200 mm, no pruning. Default lengths are
  documented as template/image dependent, so they are recorded per run (§0), never assumed.
  The earlier plan reset only `fa_threshold`, `turning_angle` and `step_size`, inheriting the
  grid's smoothing and length limits — not the untouched setting. Streamline count stays at
  the battery's 50,000 so the comparison is at equal sampling.

  Whether the randomised strategies make the reference's two repeats differ is measured, not
  assumed: the grid's Otsu window is randomised too, yet its repeats were identical at one
  thread.
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
