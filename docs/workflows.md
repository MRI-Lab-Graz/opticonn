# Workflows

All OptiConn workflows follow the same core pattern: **parameter space exploration → objective + QC/QA scoring → robust selection → freeze config → downstream graph analysis**. The sections below show the concrete commands for the most common variants.

## Bayesian optimization → Apply (recommended)
1. Find optimal parameters on a pilot subset:
   ```bash
   python opticonn.py tune-bayes \
     -i /path/to/pilot_data \
     -o studies/bayes_opt \
     --config configs/braingraph_default_config.json \
     --modalities qa \
     --n-iterations 30 \
     --sample-subjects
   ```
2. Inspect selection:
   ```bash
   python opticonn.py select -i studies/bayes_opt --modality qa
   ```
3. Apply to the full dataset:
   ```bash
   python opticonn.py apply \
     -i /path/to/full_dataset \
     --optimal-config studies/bayes_opt/qa/bayesian_optimization_results.json \
     -o studies/final_analysis
   ```

## Cross-validation bootstrap (with Bayes seeding)
1. Run Bayesian optimization (small pilot) as above.
2. Seed cross-validation with the Bayes result:
   ```bash
   python scripts/cross_validation_bootstrap_optimizer.py \
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
```bash
python opticonn.py apply \
  -i /path/to/full_dataset \
  --optimal-config studies/bayes_opt/qa/bayesian_optimization_results.json \
  -o studies/final_analysis
```
