# Workflows

All OptiConn workflows follow the same core pattern: **parameter space exploration → discriminability + QC/QA scoring → robust selection → freeze config → downstream graph analysis**. The sections below show the concrete commands for the most common variants. Add `--backend mrtrix` to any `tune-grid`/`tune-bayes`/`apply` command to use the MRtrix3 backend instead of DSI Studio (see the main `README.md`).

## Grid/random sweep → Apply (recommended — screened by discriminability)

1. Screen candidate parameters on a pilot subset:

    ```console
    python opticonn.py tune-grid \
        -i /path/to/pilot_data \
        -o studies/grid_opt \
        --config configs/braingraph_default_config.json \
        --quick
    ```

2. Inspect selection:

    ```console
    python opticonn.py select -i studies/grid_opt/sweep-<uuid>/optimize
    ```

3. Apply to the full dataset:

    ```console
    python opticonn.py apply \
        -i /path/to/full_dataset \
        --optimal-config studies/grid_opt/sweep-<uuid>/optimize/selected_candidate.json \
        -o studies/final_analysis
    ```

## Bayesian proposal, re-screened by discriminability

`tune-bayes` alone selects by the composite score (its acquisition function), not discriminability — see [Methods](methods.md). To screen its candidates by discriminability too, feed them back into the grid path with `--candidates-from-bayes`/`--bayes-top-k` (see `python scripts/cross_validation_bootstrap_optimizer.py --help`) before applying.

## Cross-validation bootstrap (with Bayes seeding)

1. Run Bayesian optimization (small pilot) as above.
2. Seed cross-validation with the Bayes result:

    ```sh
    python scripts/cross_validation_bootstrap_optimize.py \
        -i /path/to/pilot_data \
        -o studies/cv \
        --extraction-config configs/demo_config.json \
        --from-bayes studies/bayes_opt/qa/bayesian_optimization_results.json \
        --subjects 3 \
        --max-parallel 1 \
        --verbose
    ```

- Uses two waves by default; metrics/atlases stay fixed from the base config.
- Seeded parameters come from `best_parameters` in the Bayes results.

## Apply-only with known optimal config

```console
python opticonn.py apply \
    -i /path/to/full_dataset \
    --optimal-config studies/bayes_opt/qa/bayesian_optimization_results.json \
    -o studies/final_analysis
```

