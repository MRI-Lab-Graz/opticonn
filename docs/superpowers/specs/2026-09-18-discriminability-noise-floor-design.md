# Discriminability's ceiling, and reporting what actually moves the connectome

Date: 2026-09-18
Status: Components 0, 1 and 2 implemented

Note (2026-09-21): OptiConn became cross-sectional only; the between_session stratum and session-aware staging described here were removed. See 2026-09-21-cross-sectional-graph-icc-design.md.

## Problem

OptiConn ranks candidate parameter sets by repeat-run discriminability. In practice the
criterion saturates: every reasonable candidate scores exactly 1.0, so ranking falls
through to repeatability and the sweep cannot separate good candidates from great ones.

Measured on study 129 (AAL3), as edge-vector correlations (`log1p`, upper triangle):

| Comparison | r | dissimilarity (1-r) |
| --- | --- | --- |
| Same scan, same params, different seed — tracking noise | 0.997 | 0.003 |
| Same subject, different session, same params — biology + scan-rescan | 0.889 (median 0.900, n=114 pairs / 54 subjects) | 0.111 |
| Same scan, different parameters (fa 0->0.1, angle->45) | 0.864 (n=1 pair) | 0.136 |
| Different subjects, same params | 0.684 (median 0.703, n=780 pairs) | 0.316 |

Two conclusions drive this spec.

**The noise floor is far below everything else.** `docs/methods.md` defines repeats as
tracking runs of the same scan with varying random seeds, which measures tractography
stochasticity alone — about 2.5% of the between-session effect. Discriminability asks
whether a repeat of a subject is closer to that subject than to another subject; with a
noise floor that low and between-subject dissimilarity 100x larger, every non-broken
candidate passes trivially. All four candidates in the study 129 verification scored
exactly 1.0; only repeatability separated them, across a span of 0.0032 (0.9959 to 0.9991).

**Parameter choice rivals the biological effect.** A modest parameter change moved the
connectome slightly more than a real between-session change did (0.136 vs 0.111). That
ratio, not a candidate ranking, is the most decision-relevant thing OptiConn can tell a
user, and it fits `paper.md`'s no-ground-truth framing: it claims no correctness, it
quantifies how much the choice matters.

Correction: the repeatability values first quoted for that sweep (span 0.0021) were
inflated by a duplicate-loading defect in `collect_matrices`, which loaded each DSI Studio
repeat twice (the combined `.mat` and its identical `.csv` copy). One candidate reported
0.99727 against a true 0.99590, and `n_repeats` was 4 instead of 2. This is fixed; the
figures above are recomputed from the real sweep output, and the candidate ordering by
repeatability is unchanged.

A third defect was found while investigating and is a prerequisite for the rest: file
discovery returns every scan twice.

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

A companion helper parses identifiers from a path, used by Component 1:

```
scripts/utils/discovery.py::parse_subject_session(path) -> tuple[str | None, str | None]
```

Regex `sub-<id>` and `ses-<id>` from the path, preferring the filename. Returns `None` for
unparseable names rather than guessing.

## Component 1 — Variance decomposition report

Report how much the connectome moves under each source of variation, so a user can judge
whether parameter choice deserves care relative to the effect they are studying.

Four strata, all computed with the existing `reliability.edge_vector` and
`reliability._distance` so the numbers are directly comparable and no new distance metric
is introduced:

| Stratum | Pairs compared | Data source |
| --- | --- | --- |
| `tracking_noise` | same scan, same candidate, different seed | `rep_*` dirs of one combo |
| `parameter` | same scan, different candidate | same subject across combo dirs |
| `between_session` | same subject, different session, same candidate | multi-session cohorts only |
| `between_subject` | different subjects, same candidate | within one combo |

Three of the four strata are computable from what a normal sweep already writes — no extra
tracking runs. `between_session` requires a cohort with repeat scans and is reported as
unavailable otherwise.

Every stratum is reported as a distribution (n, mean, median, IQR of dissimilarity), never
a point estimate. The single-pair `parameter` figure above is exactly the weakness this
must avoid: it is one perturbation on one subject, against 114 session-pairs for
`between_session`. Concretely, `parameter` pools every (subject, candidate-pair)
combination the sweep produced rather than one chosen pair, so a 3-candidate sweep over 10
subjects yields 30 pairs, not 1.

Derived ratios, which are the actual headline:

- `parameter / between_session` — does parameter choice rival the biological effect
- `tracking_noise / parameter` — how much of the apparent parameter effect is just noise
- each stratum as a fraction of `between_subject`

New module `scripts/variance_decomposition.py` consumes a completed sweep output directory
and emits `variance_decomposition.csv` plus a summary block into
`optimization_results/`, with the headline ratio also logged at the end of a sweep.

### Explicitly not a selection criterion

The decomposition is diagnostic context and must not be wired into `rank()`. Selecting
parameters that minimise parameter-sensitivity would reward degenerate settings that
flatten real differences, reintroducing the circularity `paper.md` already documents for
`quality_score_raw`. Selection stays discriminability, then margin, then repeatability.

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
falling straight through to repeatability.

A NaN margin sorts last, matching how `rank_with_fallback`'s existing `_desc` helper treats
missing values, so a candidate whose margin could not be computed never outranks one whose
margin is known.

## Error handling

Fail fast at config load rather than after hours of compute, following the precedent
`resolve_repeats` sets for `repeats < 2`:

- `between_session` is reported as "not available: single-session cohort" rather than as a
  silent NaN, so a missing stratum is never mistaken for a zero effect.
- Any stratum with fewer than 10 pairs is emitted with a low-confidence flag naming the
  count, so a reader does not over-read an n=1 figure. The threshold is a module-level
  constant, not scattered literals.
- Files whose subject or session cannot be parsed are skipped with a warning rather than
  silently mis-grouped.
- When every candidate ties on discriminability, emit an explicit saturation warning naming
  what selection actually fell through to. `paper.md` claims a Computation Integrity
  Validation layer that detects "artificial 1.0 scores"; that check exists only in
  `bayesian_optimizer.py` and is absent from the grid path, which is the path that produced
  the saturated result described above.

## Documentation to update

`paper.md` and `docs/methods.md` both describe repeats as varying random seeds and present
discriminability as the headline contribution. Both need to state the saturation behaviour
honestly — discriminability screens out bad candidates but has limited resolution among
reasonable ones, and its resolution depends on cohort diversity and N.

## Testing

All pure-Python, no DSI Studio required, added to the existing pytest suite:

- `find_subject_files` excludes `.git` paths and collapses a symlink and its target into one
  entry.
- `parse_subject_session` handles BIDS-style names, annex-hash names (unparseable), and
  paths without a session.
- Variance decomposition on synthetic matrices with known injected structure returns strata
  in the expected order and computes the ratios correctly.
- `between_session` is reported unavailable, not NaN, for a single-session fixture.
- `rank()` orders by margin when discriminability ties at 1.0.

## Out of scope

- **Session-based repeats as the noise floor.** Considered and rejected: scanner time is
  expensive, so datasets with repeat DWI almost always acquired them because between-session
  change is the effect of interest. Treating that variance as noise would discard the
  signal. The measurements above are what remains of the idea.
- **Streamline split-half resampling.** At converged streamline counts, split halves of one
  run converge like independent seeded runs do, so it does not address the ceiling.
- **Changing the composite `quality_score_raw`.** `paper.md` already documents why it does
  not drive selection.

## Follow-up, outside this spec

The `between_session` minimum was r=0.357 against a median of 0.900. That is far enough out
to warrant checking whether it is a real change or a QC failure, since it affects the study
129 analysis already run, not just this tool.
