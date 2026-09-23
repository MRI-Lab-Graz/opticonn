# Criterion stability, the discriminability gate, and streamline-count transferability

Date: 2026-09-23
Status: Approved design, spec awaiting review
Follows: `2026-09-21-cross-sectional-graph-icc-design.md`

## Problem

The first real cross-sectional sweep on study 129 (AAL3, 12 candidates, 2 independent
10-subject waves, 2 repeats, 5,000 streamlines) produced three findings that invalidate
parts of how OptiConn currently presents itself.

### 1. Discriminability saturates for structural reasons, not because the data are clean

Every one of the 12 candidates scored exactly 1.000, in both waves. Pooling the waves to
20 subjects did not change that.

The existing docs explain saturation by saying the tracking-noise floor is negligible
(r ≈ 0.997). That explanation does not survive this run. Measured here:

| Metric | tracking noise | between subject | noise as share of between-subject |
| --- | --- | --- | --- |
| count | 0.0505 | 0.2522 | 20% |
| fa | 0.1661 | 0.4224 | 39% |
| qa | 0.1666 | 0.4291 | 39% |

Noise is large, and discriminability still pins at 1.0. The reason is that discriminability
is an *ordinal* statistic: it asks only whether the within-subject distance is smaller than
the between-subject distance, never by how much. It is a step function, so it cannot order
candidates that all pass. More subjects do not fix this (n=20 was tested); only a candidate
broken enough to make some subject pair closer than a repeat pair would score below 1.0.

Discriminability is therefore a pass/fail gate. Presenting it as the selection criterion,
as `paper.md` and `docs/methods.md` currently do, misdescribes what drives selection.

### 2. The margin is a real criterion for weighted metrics and noise for `count`

Using the two waves as independent replicates, rank agreement of the margin across waves:

| Metric | cross-wave Spearman rho | candidate spread | wave-to-wave noise | winner agreement |
| --- | --- | --- | --- | --- |
| count | **+0.03** (p=0.93) | 7.9% | 2.6% | wave1 sweep_0012, wave2 sweep_0009 — disagree |
| fa | **+0.81** (p<0.01) | 18.6% | 2.8% | both sweep_0012 |
| qa | **+0.68** (p=0.02) | 17.6% | 3.0% | both sweep_0012 |

On `count`, the margin ordering is indistinguishable from sampling noise, yet the pipeline
still reported a confident winner per wave. A user reading one wave's result would quote a
candidate chosen by chance. Nothing in the tool detects this today.

### 3. Streamline count was never varied, and the screen does not bracket production

All 12 candidates ran at `tract_count = 5000`. The study-129 production analysis uses
**5,000,000** streamlines — 1000x more — and also differs in `method` (1 vs 0),
`step_size` (1 vs 0.5) and `smoothing` (0.1 vs 0.0). Density, and through it every graph
measure, depends strongly on streamline count. A ranking obtained at 5k may not transfer to
the setting the study actually uses, and OptiConn currently offers no evidence either way.

`sweep_utils.build_param_grid_from_config` already supports `tract_count_range`
(`scripts/sweep_utils.py:210`), so varying it needs configuration, not new sweep code.

## Component 1 — Discriminability becomes an explicit gate

In `scripts/reliability.py`:

- Add a module constant `DISCRIMINABILITY_GATE = 0.95` beside the existing gates.
- `gate_reason` (or `score_combo`'s gate step) rejects a candidate whose discriminability is
  below the gate, with the reason naming the value and the threshold. NaN discriminability
  keeps its current meaning (not gradeable) and is not newly rejected by this gate.
- `rank()` drops discriminability as a sort key. The ordering becomes
  `(_desc(discriminability_margin), -repeatability, tract_count)`, with a NaN margin sorting
  last exactly as now.
- The discriminability value stays in every diagnostics row and CSV; only its role changes.

Because all plausible candidates score 1.0, this changes no current result. It removes a
sort key that never sorts.

**Interaction to document:** `tract_count` ascending remains the final tie-break. Once
`tract_count` is swept it is a real parameter, so an exact tie on both margin and
repeatability would systematically favour the cheapest candidate. It fires only on exact
ties, so it stays, but `docs/methods.md` must state it.

## Component 2 — New `scripts/criterion_stability.py`

The two-wave design already provides independent replicates. This module uses them to audit
the selection criterion itself, from the matrices a completed sweep has already written.

```
criterion_values(combo_matrices) -> dict[str, dict[str, float]]
    # {criterion: {combo_id: value}} for criterion in {"discriminability_margin", "repeatability"}

stability_rows(grouped) -> list[dict]
    # one row per (atlas, metric, criterion)

run(sweep_optimize_dir: Path, output_dir: Path) -> list[dict]
```

Per (atlas, metric, criterion), computed over the candidates present in both waves:

- `rho` — Spearman rank agreement of the criterion between wave 1 and wave 2
- `spread` — (max − min) / mean across candidates, averaged over the two waves
- `noise` — mean |wave1 − wave2| / mean, across candidates
- `snr` — `spread / noise`
- `stable` — `rho >= STABILITY_RHO_MIN` (0.5)
- `noise_share` — the variance decomposition's `tracking_noise / parameter` ratio for this
  (atlas, metric), carried through so the diagnosis can name the right remedy
- `diagnosis` — empty when stable; otherwise, in this order:
  - `noise_share > NOISE_SHARE_MAX` (0.5): "tracking noise dominates the parameter effect;
    increase tract_count" — the criterion is not estimable at this streamline count
  - `snr < SNR_MIN` (2.0): "candidates do not differ meaningfully; widen the parameter range"
  - otherwise: "estimate imprecise; more subjects will help"
- `winner_wave1`, `winner_wave2`, `winners_agree`

Constants `STABILITY_RHO_MIN = 0.5`, `SNR_MIN = 2.0`, `NOISE_SHARE_MAX = 0.5` and
`MIN_CANDIDATES_FOR_STABILITY = 5` are module-level.

Reported as unavailable, with a reason naming the count and never a number, when fewer than
2 waves are present or fewer than `MIN_CANDIDATES_FOR_STABILITY` candidates are shared by
both waves. Candidates whose criterion value is NaN in either wave are dropped from that
criterion's correlation, and the count dropped goes in the reason.

Output in `optimization_results/`: `criterion_stability.csv` (columns: atlas, metric,
criterion, n_candidates, rho, p_value, spread, noise, snr, stable, winner_wave1,
winner_wave2, winners_agree, diagnosis, reason) and `criterion_stability_summary.txt`. It is
called from the same end-of-sweep hook as the variance decomposition and the graph ICC, in
its own `try/except`, and is runnable standalone on an existing sweep.

This module reports; it never ranks and must not be imported by `rank()` or
`rank_with_fallback()`.

## Component 3 — `select` refuses an unstable winner

`opticonn select` reads `criterion_stability.csv` from the sweep's `optimization_results/`
when present. The criterion it checks is `discriminability_margin`, because Component 1
makes the margin the first ranking key and therefore the one that decides the winner; the
`repeatability` row is reported but does not gate promotion. For each (atlas, metric) whose
margin is unstable, `select` declines to promote a winner and prints the diagnosis instead
of a candidate. A metric whose criterion
is stable is promoted as today.

If the stability report is missing (an older sweep, or the report failed), `select` behaves
exactly as it does now and says that stability was not assessed. Selection inside each wave
is unchanged; only the promotion step gains the check.

On the current study-129 run this means: no winner for AAL3/count, winners for fa and qa.

## Component 4 — Streamline-count transferability

Configuration, plus one addition to Component 2. `tract_count_range` goes into the study
config so candidates span streamline counts. The stability module additionally reports, per
(atlas, metric, criterion), the rank agreement of the *other* parameters **across**
`tract_count` levels: for each level, rank the candidates sharing that level; correlate
those orderings between levels pairwise.

Row fields: `level_a`, `level_b`, `rho`, `n_candidates`, `transfers`
(`rho >= STABILITY_RHO_MIN`). Written to `tract_count_transfer.csv`, omitted entirely when
the sweep holds a single `tract_count` level (with a reason line in the summary).

This converts "is cheap screening valid?" into a measured statement. If orderings transfer,
screening at a low streamline count is defensible and the report says so; if they do not,
screening must run near the production setting, and the tool will have shown it.

### Measured result (2026-09-23)

A paired run — same 12 candidates, same subjects and seeds, at 50,000 streamlines — was
rank-correlated against the completed 5,000-streamline sweep:

| Metric | 5k vs 50k rank agreement | winner at 5k | winner at 50k |
| --- | --- | --- | --- |
| count | +0.68 (p=0.015) | sweep_0012 | sweep_0012 |
| fa | +0.93 (p<0.001) | sweep_0012 | sweep_0012 |
| qa | +0.91 (p<0.001) | sweep_0012 | sweep_0012 |

The ordering transfers across a 10x change in streamline count, and the winner is identical
at both levels on every metric. Cheap screening is therefore defensible for this cohort and
parameter range, tested to 50k — still 100x below the 5M production setting, which the
documentation must state rather than generalise beyond the evidence.

The same run showed that `count`'s margin instability at 5k (rho=+0.03) was an artifact of
too few streamlines, not a flaw in the criterion: at 50k it is stable (rho=+0.87) and its
tracking noise falls 4.4x (0.0505 to 0.0114). This is why the diagnosis in Component 2 checks
`noise_share` first. It also means the 5k sweep overstated the parameter effect — the
`tracking_noise / parameter` share was 64.8% (count) and ~76% (fa, qa) at 5k against 22.2%
and ~48% at 50k, and the headline ratio fell correspondingly (count 0.31x to 0.21x, fa and qa
0.51x to 0.44x).

## Documentation

- `docs/methods.md`: discriminability described as a gate, with the measured noise shares
  above showing why saturation is structural rather than a consequence of clean data; the
  `tract_count` tie-break interaction; the stability report and how to read a low rho.
- `docs/user_guide.md`: a section on the stability report, and what to do for each
  diagnosis.
- `paper.md`: replace the placeholder study-129 figures with this run's real numbers
  (parameter/between_subject 0.31x for count, 0.51x for fa and qa; graph ICC 0.85–1.00 at
  n=10). **Remove the claim that measures disagree on the better candidate** — that came
  from an n=5–6 exploratory probe and does not reproduce at n=10. Reframe discriminability
  as a gate, and present criterion stability as a contribution: a screening tool that
  reports when its own criterion cannot separate candidates.

## Error handling

- Every unavailable statistic carries a reason naming the failing count; never zero, never a
  bare NaN.
- A criterion whose values are identical across all candidates (zero spread) is reported
  unstable with the diagnosis "candidates do not differ meaningfully", not as a divide-by-
  zero.
- `scipy.stats.spearmanr` returns NaN for constant input; such rows are reported unavailable
  with that reason.

## Testing

Pure Python on synthetic matrices, no DSI Studio:

- A candidate ordering preserved across both waves yields `stable=True` and a high rho.
- The same ordering shuffled in one wave yields `stable=False` with the
  "estimate imprecise" diagnosis when spread is large relative to noise.
- Candidates constructed to differ only trivially yield `stable=False` with the
  "candidates do not differ meaningfully" diagnosis (`snr < SNR_MIN`).
- Fewer than 5 shared candidates, or a single wave, is reported unavailable with the count
  in the reason.
- `rank()` orders by margin with discriminability absent from the key, and a candidate below
  `DISCRIMINABILITY_GATE` is rejected with a reason naming the value.
- `select` declines to promote a winner for an unstable metric and promotes for a stable
  one; with no stability file it behaves as before and says so.
- Transfer rows are omitted for a single-level sweep and computed for a two-level one.

## Out of scope

- Changing what the margin measures, or adding a new selection statistic.
- Pooling waves into one estimate: the two waves are the independent replicates that make
  the stability check possible, and pooling would destroy it.
- Re-running study 129 at production scale (5M streamlines). The 50k paired run decides
  whether that is necessary.
- Any session-based statistic; OptiConn remains cross-sectional.
