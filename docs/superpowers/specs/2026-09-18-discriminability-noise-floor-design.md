# Discriminability noise floor and candidate resolution

Date: 2026-09-18
Status: Approved, not yet implemented

## Problem

OptiConn ranks candidate parameter sets by repeat-run discriminability. In practice the
criterion saturates: every reasonable candidate scores exactly 1.0, so ranking falls
through to repeatability and the sweep cannot separate good candidates from great ones.

Measured on study 129 (AAL3, 3 subjects, 4 candidates bracketing the production settings
at 5M tracts):

| Quantity | Correlation |
| --- | --- |
| Same subject, same candidate, two tracking runs (the noise floor) | r ≈ 0.997 |
| Same subject, two different candidates | r ≈ 0.86 |
| Different subjects, same candidate | r ≈ 0.70–0.76 |

Between-subject anatomy differs far more than either tracking noise or the parameter
perturbation, so "is a repeat of this subject closer to this subject than to another
subject" is trivially true for every candidate. All four scored discriminability 1.0;
only repeatability separated them, across a span of 0.0021.

Two root causes:

1. **The noise floor is unrealistically low.** `docs/methods.md` defines repeats as
   tracking runs of the same scan with varying random seeds. That measures tractography
   stochasticity only. At converged streamline counts it is nearly zero, so the test is
   easy to pass. Bridgeford 2021, which `paper.md` cites for discriminability, computes it
   over genuine multi-session/multi-site repeats — a materially harder test than the one
   OptiConn implements.
2. **The statistic has no headroom.** Discriminability is a win/loss proportion. Once the
   within- and between-subject distance distributions are fully separated it pins at 1.0
   and stops discriminating, regardless of how much margin separates candidates.

A third defect was found while investigating and is a prerequisite for fixing the first:
file discovery returns every scan twice.

## Component 0 — Fix file discovery (prerequisite)

`rglob("*.fz")` under a DataLad/git-annex dataset returns both the friendly symlink
(`sub-079/fib/sub-079_ses-1.odf.qsdr.fz`) and the annex object it resolves to
(`.git/annex/objects/../MD5E-...fz`). Both satisfy `is_file()`. On study 129: 342 real
scans, 684 discovered files.

Consequences:

- The sampling pool contains each scan twice, so a draw of N subjects can include the same
  scan under two names and then treat it as two different subjects. Comparing a scan
  against itself as a "different subject" yields a between-subject distance near zero and
  corrupts discriminability.
- Annex-named paths carry no subject or session identifiers, so any subject-level or
  session-level grouping is impossible for that half of the pool. This is also why run
  logs report `MD5E-...` as subject names.

Fix: exclude any path containing a `.git` component, then deduplicate by `Path.resolve()`.
Excluding `.git` leaves the friendly path; the resolve-based dedupe is a safety net for
other symlink arrangements.

This logic currently exists in three copies (`bayesian_optimizer.py` sampling,
`bayesian_optimizer.py` CLI pre-check, `cross_validation_bootstrap_optimizer.py` wave
staging) and has already drifted — the copies did not receive the same fixes. Consolidate
into one helper:

```
scripts/utils/discovery.py::find_subject_files(root, patterns) -> list[Path]
```

All three call sites use it. `reliability.py` already documents the intent that the DSI and
MRtrix paths "cannot drift apart again"; this extends that to discovery.

## Component 1 — Session-based repeats

Let real scan sessions provide the repeat axis, so the noise floor becomes the full
measurement-error stack (head position, physiology, registration, reconstruction,
tractography) rather than RNG jitter alone.

New helper:

```
scripts/utils/discovery.py::parse_subject_session(path) -> tuple[str | None, str | None]
```

Regex `sub-<id>` and `ses-<id>` from the path, preferring the filename.

Changes to `reliability.py::collect_matrices`: the grouping key becomes configurable.

- `runs` mode (default, current behaviour): key is person-session, repeats come from
  `rep_*` directories. Unchanged.
- `sessions` mode: key is the person; each session contributes one matrix to that person's
  repeat list.

Changes to wave staging in `cross_validation_bootstrap_optimizer.py`: in `sessions` mode,
group the pool by subject, drop subjects with fewer than two sessions, sample N subjects,
and stage `sessions_per_subject` sessions for each. When a subject has more sessions than
`sessions_per_subject`, take the first N by sorted session id rather than sampling, so a
given seed and cohort always stage the same scans.

`resolve_repeats` returns 1 in `sessions` mode. Paying 2x tracking cost for a noise source
measured at r ≈ 0.997 buys nothing once sessions supply the repeats. `score_combo`'s
existing "fewer than 2 repeats for some subject" gate then operates on session count,
which is the intended behaviour.

Config surface, defaulting to today's behaviour:

```json
"reliability": { "repeat_source": "runs" | "sessions", "sessions_per_subject": 2 }
```

### Known limitation to document

In a longitudinal intervention study, between-session variance contains real biological
change, so session-based repeats conflate measurement noise with true change. Connectome
fingerprints are typically far more stable than intervention effects, so subject identity
should still dominate, but this is a real caveat. `docs/methods.md` must state it, and
users studying large within-subject effects should be told to restrict repeat sessions to
a pre-intervention window.

`paper.md` and `docs/methods.md` both describe repeats as varying random seeds. Both need
updating when this ships.

## Component 2 — Margin statistic

Give the criterion headroom above its ceiling. In `reliability.py::discriminability`, also
compute the nearest-neighbour margin: for each within-subject pair, the minimum
between-subject distance minus the within-subject distance. Report the mean as
`discriminability_margin`.

Insert it into `rank()` between discriminability and repeatability:

```
(-discriminability, -discriminability_margin, -repeatability, tract_count)
```

Candidates tied at 1.0 then resolve on how much separation they achieved rather than
falling straight through to repeatability. This also serves single-session users, who
cannot use Component 1.

A NaN margin sorts last, matching how `rank_with_fallback`'s existing `_desc` helper treats
missing values, so a candidate whose margin could not be computed never outranks one whose
margin is known.

## Error handling

Fail fast at config load rather than after hours of compute, following the precedent
`resolve_repeats` sets for `repeats < 2`:

- `sessions` mode with fewer than two eligible multi-session subjects aborts with a message
  naming how many were found.
- Files whose subject or session cannot be parsed are skipped with a warning rather than
  silently mis-grouped. If skipping leaves too few subjects, abort.
- When every candidate ties on discriminability, emit an explicit saturation warning naming
  what selection actually fell through to. `paper.md` claims a Computation Integrity
  Validation layer that detects "artificial 1.0 scores"; that check exists only in
  `bayesian_optimizer.py` and is absent from the grid path, which is the path that produced
  the saturated result described above.

## Testing

All pure-Python, no DSI Studio required, added to the existing pytest suite:

- `find_subject_files` excludes `.git` paths and collapses a symlink and its target into one
  entry.
- `parse_subject_session` handles BIDS-style names, annex-hash names (unparseable), and
  paths without a session.
- `collect_matrices` groups two sessions under one subject in `sessions` mode and preserves
  current grouping in `runs` mode.
- `rank()` orders by margin when discriminability ties at 1.0.
- `sessions` mode aborts cleanly when too few multi-session subjects are available.

## Out of scope

- Streamline split-half resampling as an alternative noise floor. At converged streamline
  counts, split halves of one run converge like independent seeded runs do, so it does not
  address the ceiling; it mainly makes `tract_count` sensitivity visible, which does not
  justify the added complexity now.
- Changing the composite `quality_score_raw`. `paper.md` already documents why it does not
  drive selection.
