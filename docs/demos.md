# Demos

## MRtrix3 + OpenNeuro Demo (Recommended)
This is the primary demo for JOSS reviewers and new users. It uses `openneuro-py` to download metadata from the Slackline study (ds003138) and runs a dry-run of the Bayesian optimization workflow.

See the [Detailed Demo Walkthrough](demo.md) for a step-by-step explanation.

```bash
python scripts/opticonn_mrtrix_demo.py
```

## Bayesian + Apply quick demo (DSI Studio)
Runs on a tiny sample (downloads small HCP YA subject). Requires DSI Studio.
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
