# Sensitivity report: `opticonn report` (design)

## Why

`opticonn measures`, `outcome` and `pca` answer *how far does my estimate move across the
specifications of a sweep*. The paper's Results also rest on a set of sweep-level statistics that
exist only in the study repository: whether the leading specification is determined, how fragile
the ordering of subjects is, how many independent axes the graph measures carry, and how far the
specifications sit from DSI Studio's untouched defaults. They are general (they need nothing but a
sweep), so they belong in OptiConn, where any group can produce the same report for its own data.
This is the second of three pieces (first: measures/outcome/pca; third: rewiring the study
repository to call OptiConn).

## The command

`opticonn report -i <optimize> [--atlas A] [--metric M] [--workers N] [-o DIR] [--recompute]`

Reads a completed sweep's `optimize/` directory (any number of waves and repeats) and writes
`<optimize>/report/report.json` (every number, plus a note for anything skipped) and
`report.csv` (one flat row, stable column names, for cross-dataset tables). Each statistic below
is computed if the sweep has what it needs and otherwise skipped with a stated reason; the report
never fails because a sweep lacks, say, a second repeat.

### Statistics (definitions fixed so the paper's numbers reproduce)

Terminology: a *specification* is one `sweep_NNNN` of one wave, identified wave-scoped as
`<wave>/sweep_NNNN`; a *scan* is one subject's matrix in it. The seven *standard measures* are those
of `measures_from_matrix` (density, global efficiency binary and weighted, clustering binary and
weighted, modularity, small-worldness binary). `MIN_SUBJECTS = 3`.

**Gating for the pooled statistics.** Specifications whose `rejected` field in their wave's
`combo_diagnostics.csv` is non-empty are dropped from fragility, effective dimensionality,
displacement and the density relation. (`outcome` and `pca` keep their own gate, judged on repeat 1.)

1. **Leader test** (needs exactly two waves). From each wave's `combo_diagnostics.csv`, excluding the
   reference rows and the rows the sweep rejected (in real sweeps those have a blank margin, so this
   changes no count): candidates are the sweep ids present in both waves with a non-missing
   `discriminability_margin` in both; `n_candidates`. `discriminability_saturated` is true when every
   non-missing wave-1 `discriminability` equals 1.0. With at least 3 candidates: `cross_wave_rho`
   (Spearman of the two waves' margins; absent with a reason if either is constant), each wave's
   winner (maximum margin), `winners_agree`, `top_two_separation` (absolute gap between the two best
   mean-over-waves margins), `wave_variation` (mean over candidates of the absolute difference
   between the waves' margins) and `winner_determined` (separation > variation).
2. **Fragility** (needs at least two repeats). For each standard measure and each specification, the
   Spearman correlation across subjects between repeat 1 and repeat 2 (`seed_rho`, mean over
   measures and specifications; pairs with fewer than `MIN_SUBJECTS` subjects or a NaN correlation
   are skipped); and, for each measure, every pair of different specifications of the same wave,
   Spearman across the subjects they share on repeat 1 (`param_rho`, pooled mean). `gap = seed_rho -
   param_rho`. Counts of pairs and measures entering each arm are reported. The reference is pooled
   as one more specification here, as in the paper.
3. **Effective dimensionality** (needs two repeats). One row per (specification, subject) with the
   mean of the two repeats of each standard measure; rows with a non-finite value are dropped and
   counted; the rows are z-scored (population standard deviation, constant columns left at 1),
   and the eigenvalues of their covariance give `(sum l)^2 / sum l^2`. Undefined with fewer than
   `MIN_SUBJECTS` subjects or 3 rows. The reference is pooled as above.
4. **Displacement from the reference** (needs a reference in at least one wave; reads matrices).
   For every non-reference specification and every subject it shares with its wave's reference,
   the distance 1 - r between the log1p edge vectors of repeat 1 (`scripts.reliability.edge_vector`
   and `_distance`); `reference_over_between_subject` is the mean of these distances divided by the
   mean between-subject distance of repeat 1, pooled over every kept specification (reference
   included) and every pair of different subjects. `reference_rho`: for each standard measure and
   each non-reference specification whose wave has a reference, the Spearman correlation across
   shared subjects (at least `MIN_SUBJECTS`) of repeat 1 with the reference's, pooled mean.
5. **Density relation** (any sweep). Per specification, the mean over subjects, waves and repeats of
   density and of each standard measure; `<measure>_vs_density` is the Spearman correlation across
   the kept non-reference specifications. The reference's gate verdict (`reference_rejected`: its
   `rejected` field non-empty in any wave) and `reference_density_percentile` (the fraction of a
   wave's kept non-reference specifications with lower density than the reference, mean over waves).

Variance decomposition and graph ICC already have OptiConn commands and are not repeated.

### How it gets its inputs

`report` builds the per-subject measure tables for repeat 1 and repeat 2 across all waves itself
(the code of `measures`), into `<out>/tables/repeat1` and `tables/repeat2`, and reuses a table that
is already there when its `measures.json` records the same atlas, metric, repeat, wave setting and
sweep file count; `--recompute` forces a rebuild. Only the displacement reads matrices directly.
`measures` gains `--wave all`: every wave is included and the specification ids are wave-scoped
(`wave1/sweep_0001`); with a single wave the ids are unchanged.

## Architecture

- `scripts/report_stats.py`: the five statistics as pure functions over simple inputs (dicts of
  per-scan measures, lists of margins, edge vectors); no I/O.
- `scripts/report.py`: reads the sweep and tables, applies the gating, calls the statistics,
  writes `report.json` and `report.csv`, and the CLI.
- `scripts/measures_table.py`: `--wave all` and the cache-reuse check.
- `scripts/opticonn_hub.py`: register the command (the pass-through table already used).
Reused: `reliability.collect_matrices`, `edge_vector`, `_distance`, `measures_from_matrix`.

## Errors

A clean `SystemExit` with a message for: an unreadable sweep, a sweep with no matrices for the
chosen atlas/metric, an ambiguous atlas/metric. A statistic that cannot be computed is not an
error: it is reported as absent with its reason in `report.json` and an empty cell in `report.csv`.

## Testing (test first)

- Unit tests of each statistic on hand-built inputs with known answers (a margin table where the
  winner is determined and one where it is not; perfectly reproducible repeats giving
  `seed_rho = 1`; independent measures giving an effective dimensionality near the number of
  measures and perfectly correlated ones giving 1; a displacement ratio from vectors with known
  distances; the density relation and percentile).
- A synthetic sweep fixture with two waves, two repeats and a reference runs the whole command and
  checks the skip notes (one wave, one repeat, no reference).
- **Acceptance against the paper:** on the seven original sweeps (`results/<dataset>`: two waves of
  ten subjects, two repeats, a reference, AAL3 count matrices) `report` reproduces to 1e-9 the
  study repository's committed `fragility_summary.csv` (`seed_rho`, `param_rho`, `gap`,
  `effective_dimensionality`, `reference_over_between_subject`, `reference_rho` and the pair/measure
  counts) and the leader columns of `cross_dataset_summary.csv` (`n_candidates`,
  `discriminability_saturated`, `cross_wave_rho`, winners, `winners_agree`, `top_two_separation`,
  `wave_variation`, `winner_determined`). The density relation is defined on tables that average
  repeats, whereas the committed `defaults_gates.csv` used OptiConn's per-combo diagnostics; it is
  compared with a tolerance of 0.01 and the same sign, and the acceptance test says so.

## Out of scope

Cross-dataset aggregation (agreement matrix, demographic modulation, preprocessing summaries), any
change to the statistics' definitions, other estimators, and rewiring the study repository.

## Working constraints

Developed in a git worktree on branch `feature/sensitivity-report`; merged to `main` after the
acceptance run. The only existing files touched are `scripts/measures_table.py` and
`scripts/opticonn_hub.py`.
