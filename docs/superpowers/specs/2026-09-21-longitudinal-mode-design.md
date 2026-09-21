# Cross-sectional and longitudinal optimization modes

Date: 2026-09-21
Status: Approved, not yet implemented

## Problem

OptiConn treats every cohort the same way, and that is wrong for the two cases users
actually have.

**Cross-sectional cohorts** have one DWI session per subject. The current design fits
them: repeats are re-runs of one scan, discriminability asks whether a subject's repeats
are closer to each other than to other subjects, and there is nothing else to measure.

**Longitudinal cohorts** have two or more sessions per subject. Scanner time is expensive,
so a study that acquired repeat DWI almost always did so because between-session change is
the effect of interest. Two things are wrong for these cohorts today:

1. `--sessions-per-subject 2` stages both sessions, but `reliability.discriminability`
   keys on the scan, so `sub-043_ses-2` lands in the *between-subject* pool and is compared
   against `sub-043_ses-1` as though it were a stranger. `docs/methods.md` documents this as
   a caveat. It is not a defensible design: that pair is neither a repeat (it carries real
   biological change) nor a different subject.
2. Having staged the sessions, OptiConn extracts almost nothing from them. The
   `between_session` stratum of the variance decomposition becomes available, and that is
   all. The question a longitudinal user needs answered — *would my finding have been
   different under a different parameter choice?* — is never asked.

The obvious alternative, making sessions the repeats so discriminability becomes the
conventional test-retest statistic, is rejected for the reason recorded in
`2026-09-18-discriminability-noise-floor-design.md`: it treats the signal as noise, and
would reward parameter sets that are blind to the change being studied.

On study 129 the parameter effect is already 1.23x the between-session effect
(0.136 vs 0.111). A parameter choice can therefore be as large as the longitudinal effect
being reported, which is precisely why the question deserves a direct answer.

## Component 1 — "between-subject" must mean between subject

`scripts/reliability.py::discriminability` and `::discriminability_margin` both build the
between pool as *every vector under every other key*:

```
others = [v for other, vs in vecs.items() if other != subject for v in vs]
```

Change both to group keys by parsed subject, using the existing
`scripts.utils.discovery.parse_subject_session`. A key's own other sessions go in neither
pool: not repeats, not strangers. Concretely, for a scan key `k` with subject `s`, the
between pool is every vector whose key parses to a subject other than `s`.

Keys whose subject does not parse — git-annex content hashes, the synthetic `sub0`/`sub1`
ids used throughout the test suite — fall back to today's behaviour, each key being its own
subject. Cross-sectional runs and every existing test are therefore unaffected.

Both functions already return NaN when the between pool is empty, which is the correct
result for a cohort that collapses to a single subject; no new error path is needed.

`variance_decomposition.compute_strata` already excludes same-subject pairs from its
`between_subject` stratum. This change brings `reliability` into line with it, rather than
introducing a new convention.

**Consequence to state in the docs:** discriminability and margin on multi-session runs
will shift, generally upward, because the hardest comparisons are removed from the between
pool. Any study-129 figure quoted from a multi-session run must be recomputed before
publication. Cross-sectional figures do not change.

## Component 2 — Longitudinal robustness report

Two statistics, added to `scripts/variance_decomposition.py`, both computed from the
per-(combo, key) rep-0 edge vectors that `compute_strata` already builds. No new distance
metric, no extra tracking runs.

### 2a. Per-candidate parameter sensitivity

For each candidate `C`, over the scans `C` shares with at least one other candidate:

```
sensitivity(C) = mean over keys k of ( mean over other candidates D of distance(vec[C,k], vec[D,k]) )
```

Reported against the median `between_session` dissimilarity for the same (atlas, metric):

```
ratio(C) = sensitivity(C) / median_between_session
```

A ratio at or above `SENSITIVITY_FLAG_RATIO = 1.0` is flagged: moving away from `C` to a
neighbouring candidate shifts the connectome at least as far as a real session change does.
The threshold is that module-level constant, not a scattered literal.

When `between_session` is unavailable for an (atlas, metric), `sensitivity(C)` is still
reported and the ratio column is left empty with the reason, rather than the whole row
being dropped.

### 2b. Change-ordering agreement

This is the statistic a longitudinal paper should quote.

For each candidate `C`, build a vector over subjects, where each subject's entry is the
mean dissimilarity across that subject's session pairs under `C`. The vector is the
candidate's answer to "who changed, and by how much". Compute Kendall's tau
(`scipy.stats.kendalltau`, already a dependency) between these vectors for every candidate
pair, over the subjects the pair shares, and report the mean, minimum and number of pairs.

A high mean tau means the ordering of who changed most survives the parameter choice, so a
longitudinal finding rests on the data. A low mean tau means the ordering is an artefact of
the parameter set, and no single candidate's result should be trusted on its own.

### Availability

Both statistics require at least 2 candidates. 2b additionally requires at least
`MIN_SUBJECTS_FOR_TAU = 3` subjects with 2+ parsed sessions, since a tau over two subjects
is meaningless. Below these thresholds the report is emitted as explicitly unavailable,
with the reason naming the count, exactly as `between_session` already is. An unavailable
statistic is never rendered as zero.

Candidate pairs sharing fewer than 3 subjects are skipped rather than contributing a
degenerate tau, and the count of skipped pairs is reported.

## Component 3 — Mode detection

In `scripts/cross_validation_bootstrap_optimizer.py`, after staging, count the subjects
with 2+ sessions among the staged scans (via `parse_subject_session`). Three or more means
`longitudinal`; otherwise `cross-sectional`. Log the resolved mode and the count that
decided it, next to the existing `Staged %d scans` line.

Add `--mode {auto,cross-sectional,longitudinal}`, default `auto`:

- `auto` — as above.
- `cross-sectional` — skip the longitudinal report even on a multi-session cohort.
- `longitudinal` — required. If the staged cohort has fewer than 3 multi-session subjects,
  fail at config load with a message naming the actual count, following the fail-fast
  precedent `resolve_repeats` sets for `repeats < 2`. Failing after hours of tracking is
  the outcome this avoids.

The mode is recorded in the run's output so a reader can tell which report to expect.

**Selection is identical in both modes:** discriminability, then margin, then
repeatability, then fewer tracts. The mode changes what is *reported*, never what is
*chosen*.

## Explicitly not a selection criterion

Neither statistic may be wired into `rank()` or `rank_with_fallback()`. Selecting for low
parameter sensitivity would reward degenerate settings that flatten real differences,
reintroducing the circularity `paper.md` documents for `quality_score_raw`. Selecting for
high change-ordering agreement would reward whichever candidate is most typical of the
candidate set, which is a property of the set, not of the data.

## Output

Written by the existing sweep hook alongside the decomposition, into
`optimization_results/`:

- `longitudinal_robustness.csv` — one row per (atlas, metric, candidate) with
  `sensitivity`, `median_between_session`, `ratio`, `flagged`, plus one row per
  (atlas, metric) carrying the tau summary (`tau_mean`, `tau_min`, `n_pairs`,
  `n_pairs_skipped`).
- A `longitudinal robustness` section appended to `variance_decomposition_summary.txt`,
  carrying the tau headline and the flagged candidates.
- The tau headline is logged at the end of a sweep, as the decomposition headline already is.

On a cross-sectional cohort neither artifact is written, and the summary states the mode and
why the report is absent.

## Error handling

- Unavailable statistics are reported with a reason naming the failing count, never as zero
  or NaN.
- Scans whose subject or session cannot be parsed are skipped with a warning, as
  `compute_strata` already does.
- A candidate present in only one wave still contributes to sensitivity, since candidates
  are already wave-scoped by `combo_id`.
- `kendalltau` returns NaN for a constant input vector (every subject changed identically);
  such pairs are counted as skipped rather than propagating NaN into the mean.

## Testing

All pure-Python on synthetic matrices, no DSI Studio, added to the existing pytest suite:

- `discriminability` excludes a subject's own other session from the between pool, and is
  unchanged for keys that do not parse.
- `discriminability_margin` does the same.
- A subject's own session no longer being in the between pool raises discriminability on a
  fixture where the two sessions are more similar to each other than to other subjects.
- Change-ordering agreement returns a high tau when candidates preserve an injected
  per-subject change ordering, and a low tau when one candidate scrambles it.
- The tau report is unavailable, with a reason naming the count, for 2 multi-session
  subjects; available at 3.
- Per-candidate sensitivity flags the candidate whose injected offset exceeds the injected
  session change, and does not flag the others.
- Mode auto-detection resolves longitudinal at 3 multi-session subjects and cross-sectional
  at 2.
- `--mode longitudinal` on an inadequate cohort fails at config load, before tracking.

## Documentation to update

- `docs/user_guide.md` — a section naming the two modes, what each reports, and how to read
  a low tau.
- `docs/methods.md` — replace the existing "discriminability's unit is the scan, so a
  subject's second session is compared as a different subject" caveat, which Component 1
  removes.
- `paper.md` — the modes and the change-ordering statistic, once measured on study 129.

## Out of scope

- **Sessions as repeats.** Rejected in the prior spec and re-rejected here: it treats the
  effect of interest as noise.
- **Changing selection in longitudinal mode.** The reports inform the user; they do not
  rank.
- **Runs (multiple DWI acquisitions within one session).** `parse_subject_session` parses
  `sub-`/`ses-` only. Treating within-session runs as a third level is a separate change,
  and today they are handled as whatever the file names say.
