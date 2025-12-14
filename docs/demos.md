# Demos

## Bayesian + Apply quick demo
Runs on a tiny sample (downloads small HCP YA subject).
```bash
python scripts/opticonn_demo.py --step all
```
Outputs under `demo_workspace/results/bayes/<modality>/...` and `demo_workspace/results/apply/<modality>/...`.

## Cross-validation demo (seeded from Bayes)
Seeds parameters from the Bayes demo by default.
```bash
python scripts/opticonn_cv_demo.py --workspace demo_workspace_cv
```
Outputs under `demo_workspace_cv/results/cv/optimize/` with wave configs and logs. Override seed with `--from-bayes <path>`.

## Notes
- DSI Studio must be installed and reachable; errors include the download link.
- The demos fix connectivity metrics/atlases from the base config and vary tractography parameters only.
