# Outcome sensitivity: how much does your estimate depend on the specification?

A completed sweep (`opticonn tune-grid`) holds one connectome per subject for every tractography
specification. These commands turn that into the question a study has: *how much does the number I
will report depend on the specification?* They are not tied to any outcome: you name the predictor,
the covariates and the measure.

```bash
# 1. one table of graph measures per subject and specification (add --extended for five more)
opticonn measures -i studies/demo/sweep-*/optimize --extended

# 2. the effect of a predictor on a measure in every specification
opticonn outcome -i studies/demo/sweep-*/optimize --variables participants.tsv \
    --predictor age --covariate sex --measure "global_efficiency(weighted)"

# 3. PCA of the measures in every specification (optionally with a predictor)
opticonn pca -i studies/demo/sweep-*/optimize --variables participants.tsv --predictor age --covariate sex
```

## Inputs

`participants.tsv` (or any delimited file) needs a `participant_id` column, with or without the
`sub-` prefix; all other column names are yours. A numeric column is used as it is; a column with
exactly two distinct values (for example `F`/`M`) is coded 0/1 in sorted order and the coding is
written to the summary; more than two levels is an error. Subjects with a missing value are
dropped from that analysis and counted.

## What you get

`outcome/outcome.csv` has one row per specification: its gate verdict (density 0.02-0.6, at most
10% isolated nodes; override with `--density-range` and `--max-isolated`), and for each measure the
partial Spearman correlation with the predictor, its p value and 95% interval. The summary gives
the range of the effect, how many specifications differ in sign from the median, the fraction with
p < 0.05, how the effect relates to density, and where the reference specification (DSI Studio's
defaults, if the sweep included it) falls. `pca/` reports variance explained, loadings and their
stability, the associations of PC1 and PC2 with a predictor, and which swept parameter, if any,
separates the specifications into regimes.

Age and sex above are only an example of a continuous and a group outcome; any column works.
