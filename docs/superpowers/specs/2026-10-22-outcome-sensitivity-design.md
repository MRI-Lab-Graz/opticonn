# Outcome sensitivity: measures, outcome curve, PCA (design)

## Why

OptiConn screens tractography specifications and, with `view`, shows how far subjects re-order
between them. It cannot yet answer the question a study actually has: *how much does the estimate
I will report depend on the specification?* The paper that accompanies OptiConn answers it with
a continuous outcome, a group outcome and a PCA on one sample; that code lives in the study
repository and is hard-wired to age, sex and one measure. This design moves it into OptiConn as
generic, documented commands, so that anyone can run it on any sweep with any outcome. Age and sex
remain only as the worked example in the documentation and the study repository.

This is the first of three pieces (A+B+C below). The sensitivity statistics of the paper's Results
(fragility, effective dimensionality, leader test, displacement, defaults versus gates) are a
second spec; rewiring the study repository to call OptiConn is a third.

## Commands

All three read a completed sweep's `optimize/` directory, as `view` does, and work on one wave
directory, i.e. one `*/combos` parent of the sweep (`--wave` is required if the sweep has two).

**`opticonn measures -i <optimize> [--atlas A] [--metric M] [--repeat N] [--extended] [--workers N]`**
Writes `<optimize>/measures/measures.csv`, one row per (specification, subject): the seven
standard global measures of `scripts.compute_network_measures_from_connectivity`, with
`--extended` five more (local efficiency, mean strength, characteristic path length, mean
betweenness, degree assortativity; definitions in the study repository's PCA addendum), plus
`density` and `isolated_fraction`. Also writes `params.csv` (the swept parameters of each
specification, from each specification's own `diagnostics.json`, as `view` does) and a sidecar JSON recording the atlas, metric,
repeat and OptiConn version. `--atlas`/`--metric` are required only if the sweep holds more than
one. The first repeat is used by default.

**`opticonn outcome -i <optimize> --variables FILE --predictor COL [--covariate COL ...] --measure NAME [...]`**
For each specification that passes the gates, the partial Spearman correlation of each measure
with the predictor, covariates partialled out, with p and a 95% interval. Writes `outcome.csv`
(one row per specification), `outcome_summary.json` and `outcome.png`. The summary gives, per
measure: the range, the number of specifications whose sign differs from the median, the
fraction with p < 0.05, the Spearman correlation of the effect with density, and the value, gate
status and density percentile of the reference specification; and `n_gated`.

**`opticonn pca -i <optimize> [--measures NAME ...] [--anchor NAME] [--variables FILE --predictor COL [--covariate COL ...]]`**
Per specification: z-score the measures across subjects, principal components by SVD, PC1 and PC2,
the sign of each component fixed so its loading on the anchor measure
(default `global_efficiency(weighted)`) is positive. Reports variance explained, the PC1 loadings,
their stability (cosine similarity with the median loading vector across specifications), and,
if a predictor is given, the partial Spearman of each component score with it. Also reports
which swept parameter, if any, separates the specifications into non-overlapping ranges of PC1
variance explained. Writes `pca.csv`, `pca_summary.json`, `pca.png`.

## Conventions

1. **Variables table:** a delimited file with a `participant_id` column, matched to subject ids
   with or without the `sub-` prefix; a BIDS `participants.tsv` works. All other column names are
   the user's.
2. **Binary columns:** a predictor or covariate with exactly two distinct values is coded 0/1 in
   sorted order, and the coding is written to the summary. More than two levels is an error (no
   dummy coding). Numeric columns are used as they are.
3. **One estimator:** partial Spearman (Pearson correlation of the ranks after regressing the
   ranks of the covariates out; p from the t distribution with n - 2 - k degrees of freedom for
   k covariates; 95% interval by Fisher's z with standard error 1/sqrt(n - 3 - k)).
4. **Gates:** OptiConn's (`scripts.reliability.DEFAULT_GATES`: density 0.02-0.6, at most 10%
   isolated nodes), judged on a specification's mean density and mean isolated fraction;
   `--density-range` and `--max-isolated` override.
5. **Reference specification:** taken from each specification's `diagnostics.json` (`reference`
   flag), as `view` does; never a candidate, always reported.
6. **Missing data:** a subject missing the predictor, a covariate or a measure (or with an
   undefined measure) is dropped from that specification; the number dropped per specification
   is a column and its total is in the summary.

## Architecture

New modules in `scripts/`, dispatched from `opticonn_hub.py` like the existing commands:
`measures_table.py`, `outcome_curve.py`, `pca_curve.py`, `variables.py`. Existing code is reused,
not copied: `reliability.collect_matrices`, `reliability.gate_reason`,
`compute_network_measures_from_connectivity.measures_from_matrix`. Only `measures` reads matrices;
`outcome` and `pca` read `measures.csv`, `params.csv` and the variables table, and stop with an
instruction to run `opticonn measures` if `measures.csv` is missing.

## Errors (a clear message, never a silent fallback)

A missing column or measure name; a binary-coded variable with more than two levels; subjects
absent from the variables table (dropped and counted); no specification passing the gates; a
sweep whose `diagnostics.json` files carry no parameters (the separating-parameter step is skipped with a note); a
two-wave sweep without `--wave`.

## Testing (test first)

- Unit: the estimator against the textbook partial-correlation formula; perfect fits give p = 0
  and no exception; binary coding and its record; id matching with and without `sub-`; gates;
  the summary statistics on hand-built tables; a PCA with a planted component and the sign
  convention; the separating-parameter check; the five extra measures on a path graph, a complete
  graph and a disconnected graph.
- End to end: a small synthetic sweep directory built in the test (matrices in the layout
  `collect_matrices` reads), so each command runs without DSI Studio.
- **Acceptance against the paper:** on the ds005270 sweep, `measures --extended`, `outcome` (age,
  covariate sex; sex, covariate age) and `pca` (7 and 12 measures) reproduce the study repository's
  committed `speccurve_ds005270.csv`, `pcacurve_ds005270.csv` and `pcacurve12_ds005270.csv` to
  numerical tolerance. This is the proof that moving the code changed no result.

## Out of scope

Estimators other than partial Spearman; dummy coding; mixed models; the sensitivity statistics of
the paper's Results (second spec); any change to existing OptiConn modules other than registering the
three subcommands in `scripts/opticonn_hub.py`; rewiring the study repository (third spec).

## Working constraints

Developed in a git worktree on branch `feature/outcome-sensitivity`, merged to `main` only after
the study repository's sweeps, which run from the live checkout, have finished. OptiConn code is
imported by the study repository (`PYTHONPATH`), which is why the branch is kept off `main` until then;
the only existing file touched is `scripts/opticonn_hub.py`, to register the commands.
