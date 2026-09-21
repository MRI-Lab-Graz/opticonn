# Cross-sectional optimization and graph-measure reliability

Date: 2026-09-21
Status: Approved design, spec awaiting review
Supersedes: `2026-09-21-longitudinal-mode-design.md`

## Decision

OptiConn optimizes tractography parameters for later graph-theory analysis, and does so
**cross-sectionally only**. No step uses, requires or reports repeat sessions.

Two reasons, both from the user:

1. Datasets with repeat DWI almost always acquired it to measure change, typically an
   intervention effect. Any statistic built on between-session variation treats that
   signal as noise, or as a benchmark, and is biased whichever way it is used.
2. Most datasets have one session. A tool whose core output depends on sessions is
   unavailable for most users.

The staging code added for sessions (`--sessions-per-subject`, session-aware sampling), the
`between_session` stratum of the variance decomposition, and the longitudinal-mode design
are removed.

## Evidence behind the graph-measure ICC

On the study-129 AAL3 sweep (count connectivity, 2 tracking repeats), the existing
statistics cannot tell two candidates apart, while the ICC of their graph measures (one-way
ICC, subjects against tracking repeats) points to differences:

| candidate | discr. | margin | repeatability | geff_bin | clust_bin | clust_w | sw_bin | modularity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (n=6) | 1.000 | 0.163 | 0.951 | 0.62 | 0.84 | 0.72 | 0.87 | 0.93 |
| 2 (n=5) | 1.000 | 0.158 | 0.950 | 0.92 | 0.71 | 0.93 | 0.91 | 0.91 |

Candidate 2 is an incomplete run: one scan lacks its second repeat, so the reliability gate
rejects it, and its ICC uses the 5 complete subjects.

This is **suggestive, not proof**. At these sample sizes the 95% intervals are very wide,
roughly −0.17 to 0.94 for the 0.62 and 0.53 to 0.99 for the 0.92, and they overlap. What the
probe does support:

- **ICC does not obviously share discriminability's ceiling.** A scalar graph measure
  compresses the connectome, so tracking noise is no longer negligible against
  between-subject spread. Values spread from 0.62 to 0.93 rather than sitting at 1.0.
- **Measures may disagree on the better candidate.** Here global efficiency favours
  candidate 2 and binary clustering favours candidate 1. If that holds at adequate sample
  size, "best for graph theory" depends on which measures the study analyses. So ICC is
  reported per measure, never collapsed into a ranking.

On the 3-scan pilot sweep, ICC went negative and varied wildly: at that sample size it is
noise. Hence the confidence intervals and the raised default sample size below. The fresh
study-129 sweep (see "After implementation") is what confirms or refutes the separation.

Probe script (not shipped): `scratchpad/icc_probe.py`.

## Component 1 — One baseline scan per subject

New helper in `scripts/utils/discovery.py`:

```
baseline_scans(pool: list[Path]) -> list[Path]
```

Returns one path per subject: the first session in natural order of the session id
(`ses-2` before `ses-10`), using the existing `parse_subject_session` and `_natural_key`.
A subject with a single, session-less scan contributes that scan. Paths whose subject does
not parse are kept, each counted as its own subject, with one warning naming the count and
up to three examples. This preserves today's behaviour for annex hashes and for the
synthetic ids used in tests. Output order is the input order of the chosen paths, so
seeded sampling stays reproducible.

`select_scans` becomes:

```
select_scans(pool: list[Path], n_subjects: int, seed: int) -> list[Path]
```

the existing seeded `random.Random(seed).sample`, or the whole pool when `n_subjects`
covers it. The `sessions_per_subject` parameter and the session-aware branch are deleted.

Both backends' DSI sampling paths use it:

- `scripts/cross_validation_bootstrap_optimizer.py::run_wave_pipeline` stages through
  `select_scans`.
- `scripts/bayesian_optimizer.py::_get_all_subjects` returns `baseline_scans(...)` of what
  it finds today. Without this, on a multi-session dataset the Bayesian path could sample a
  post-intervention scan.

**Interaction with `exclude_scans`:** `baseline_scans` runs first, and exclusion is
applied to its result, so a subject whose baseline is excluded **drops out**. It does not
fall back to a later session, because that would reintroduce a post-intervention scan
through the QC path. The number of subjects dropped this way is logged.

`select_scans` therefore stays a plain seeded sampler over an already one-per-subject pool;
`run_wave_pipeline` calls `baseline_scans`, then applies the exclusion, then calls
`select_scans`.

The MRtrix3 backend is unaffected: it already processes one user-named session per run
(`--session`).

## Component 2 — Remove session options, fail loudly on old configs

Delete:

- `--sessions-per-subject` from `scripts/cross_validation_bootstrap_optimizer.py` and from
  `tune-grid` in `scripts/opticonn_hub.py`, including the forwarding code.
- The `sessions_per_subject` parameter of `generate_wave_configs` and
  `generate_single_wave_config`, and the key they write.
- The `sessions_per_subject` validation rule in `scripts/json_validator.py`, and the two
  "is ignored" warnings in the optimizer's wave-config resolution.
- `tests/test_sessions_flag_warnings.py`, and the session-aware tests in
  `tests/test_session_selection.py` and `tests/test_discovery.py`.

Old configs must not silently change meaning, since staging two sessions per subject and
staging one give different numbers. `run_wave_pipeline` raises `ValueError` before staging
when `data_selection` contains `sessions_per_subject`:

```
data_selection.sessions_per_subject was removed: OptiConn stages one scan per subject
(the first session). Remove the key from <config path>.
```

This check lives only in `run_wave_pipeline`. That is the single point every wave passes
through, including runs with `--no-validation`, and it runs before any tracking. A removed
CLI flag already fails through argparse ("unrecognized arguments"), so no extra code is
needed for it.

## Component 3 — Variance decomposition without sessions

In `scripts/variance_decomposition.py`:

- Remove the `between_session` stratum and every ratio that uses it
  (`parameter_over_between_session`, `between_session_over_between_subject`).
- Three strata remain: `tracking_noise`, `parameter`, `between_subject`.
- The headline becomes `parameter_over_between_subject`: how far the parameter choice moves
  a connectome, relative to the difference between two people. For example: "parameter
  choice moves the connectome 0.43x as far as the difference between two subjects".
- Session parsing is no longer needed there; subject parsing stays, for `between_subject`.

The module stays diagnostic and must not feed `rank()` or `rank_with_fallback()`.

## Component 4 — Graph-measure ICC report

### Shared measure computation

Split `scripts/compute_network_measures_from_connectivity.py::compute_measures` into:

```
measures_from_matrix(mat: np.ndarray, compute_smallworld: bool = False,
                     smallworld_nrand: int = 10, seed: int = 42,
                     weight_type: Literal["strength", "distance"] = "strength") -> dict[str, float]
```

and a `compute_measures(connectivity_csv, ...)` that reads the CSV and delegates to it. The
behaviour of `compute_measures` is unchanged apart from the addition below.

Add `modularity`: weighted Louvain (`networkx.community.louvain_communities`, `seed` from
the argument), scored with `networkx.community.modularity`. It is added to
`measures_from_matrix`, so the network-measure CSVs OptiConn already writes and the ICC
report use the same set of measures. Louvain is stochastic. The fixed seed makes it
reproducible, and part of modularity's within-subject variance comes from the algorithm,
not from tractography. The report's summary text says so.

### ICC

New module `scripts/graph_icc.py`:

```
icc_1_1(x: np.ndarray) -> tuple[float, float, float]
    # x: subjects x repeats. Returns (icc, ci_low, ci_high).
    # One-way random effects ICC(1,1), 95% CI from the F distribution (scipy.stats.f).

compute_graph_icc(combo_matrices: dict[str, dict[str, list[np.ndarray]]]) -> list[dict]
    # combo_matrices as one value of variance_decomposition.collect_sweep_matrices().
    # One row per (combo_id, measure): n_subjects, icc, ci_low, ci_high,
    # low_confidence, reason.

run(sweep_optimize_dir: Path, output_dir: Path) -> None
```

- Repeats per subject are truncated to the minimum count in that combo, so `x` is
  rectangular.
- Subjects with fewer than 2 repeats are dropped from that combo's ICC.
- `MIN_SUBJECTS_FOR_ICC = 10`. Below it, a row is still reported, with
  `low_confidence=True`.
- With fewer than 3 subjects, or zero total variance, ICC is not reported: the row carries a
  `reason` and empty numeric fields, never a zero.
- A measure that raises during computation on a matrix is NaN for that matrix. A subject
  with any NaN measure is dropped from that measure's ICC, and the count dropped goes into
  `reason`.

Output in `optimization_results/`:

- `graph_icc.csv` — columns `atlas, metric, combo_id, measure, n_subjects, icc, ci_low,
  ci_high, low_confidence, reason`.
- `graph_icc_summary.txt` — per (atlas, metric), the most reliable candidate per measure,
  plus the modularity caveat.

`run` deletes previous outputs first, as `variance_decomposition.run` does, so a re-run
never appends duplicates.

It is called from the same sweep hook as the variance decomposition, in its own
`try/except` so a failure in one report never suppresses the other. It is also runnable
standalone (`python -m scripts.graph_icc <optimize_dir>`) on existing sweeps, as the probe
was.

### Explicitly not a selection criterion

ICC is reported, never ranked, for three reasons. The probe shows measures disagree on the
better candidate. Choosing which measures count is a study decision. And a candidate that
flattens individual differences can still score well on some measures. Ranking stays:
discriminability, margin, repeatability, fewer tracts.

## Component 5 — Default sample size

`--subjects` default goes from 3 to 10 in `scripts/cross_validation_bootstrap_optimizer.py`
and in `tune-grid` (`scripts/opticonn_hub.py`), with help text stating it counts subjects,
one scan each. `--quick` keeps its own small sample for smoke tests. The defaults of
`generate_wave_configs`/`generate_single_wave_config` follow, so direct callers get the
same default.

This roughly triples the compute of a default run. That is the price of ICC and
discriminability estimates that are not noise.

## Documentation

- `docs/methods.md` — replace the "Session-aware wave staging" section with a
  cross-sectional statement (one baseline scan per subject, why). Add an ICC section with
  the probe table. Remove the `between_session` row from the saturation table and its
  discussion.
- `docs/user_guide.md` — remove the session bullets. Update `--subjects`. Add ICC to
  "Reading the results". Replace the `between_session` stratum row and the
  `parameter / between_session` headline.
- `paper.md` — cross-sectional scope. Headline ratio `parameter / between_subject`. Graph
  ICC as a contribution. A limitations paragraph: reliability is not validity, and
  reproducible false-positive connections can still distort graph measures (Zalesky et al.
  2016, *NeuroImage*, "Connectome sensitivity or specificity: which is more important?").
  Add the reference to `paper.bib`. Study-129 figures are updated only from the fresh sweep
  (see below), not in this change.
- Mark `2026-09-21-longitudinal-mode-design.md` as superseded by this spec. Add a note to
  `2026-09-18-discriminability-noise-floor-design.md` that its `between_session` stratum
  was removed.

## Testing

Pure Python, synthetic data, no DSI Studio:

- `baseline_scans` picks `ses-1` over `ses-2`, orders `ses-2` before `ses-10`, keeps a
  session-less scan, and keeps unparseable paths as individual subjects with one warning.
- `select_scans` is seed-reproducible and returns the whole pool when `n_subjects` covers it;
  a wave stages exactly one scan per subject.
- `run_wave_pipeline` raises on a config containing `sessions_per_subject`, before staging.
- Excluding a subject's baseline scan drops the subject; it does not stage `ses-2`.
- `_get_all_subjects` returns one file per subject on a multi-session layout.
- Variance decomposition reports three strata, headline `parameter_over_between_subject`,
  and has no `between_session` key.
- `icc_1_1` returns about 1 when subjects differ and repeats agree, about 0 when repeats
  vary as much as subjects, and a CI containing the point estimate. Check it against a
  hand-computed value on a small fixed matrix.
- `compute_graph_icc` flags `low_confidence` below 10 subjects, reports a reason and no
  number below 3, and drops subjects with fewer than 2 repeats.
- `measures_from_matrix` returns the same values as `compute_measures` on the same matrix
  via CSV, plus `modularity`, which is reproducible for a fixed seed.
- `graph_icc.run` twice does not duplicate rows.
- The sweep hook calls both reports, each isolated in its own `try/except`. Extend the
  existing AST guard test.

## After implementation (not part of this change)

1. One fresh cross-sectional `tune-grid` on study 129: AAL3, baseline scans, 10 subjects per
   wave, scans flagged by the QC gate excluded.
2. Update `paper.md`, `docs/methods.md` and the user-guide numbers from that run only.
3. Remove the force-added `.pyc` files from history, push, submit to JOSS.

## Out of scope

- **Any session-based statistic**, including the longitudinal robustness report of the
  superseded spec.
- **ICC in ranking.** Revisit only if the fresh sweep and later datasets show one measure
  set consistently separating candidates.
- **Bootstrap repeats.** Residual or wild bootstrap of the diffusion signal would give the
  repeats measurement noise instead of seed noise only, while staying cross-sectional. It
  needs a reconstruction per resample. It is the leading idea for the follow-up paper, not
  this change.
- **Nodal ICC** (per-node degree or strength).
