# Troubleshooting

## DSI Studio not found
- Ensure `--dsi-path` was set during install.
- Verify the path exists and is executable.
- Download if missing: https://github.com/frankyeh/DSI-Studio/releases.

## Missing connectivity matrices in apply
- Check DSI Studio version/path; rerun with `--verbose`.
- Ensure atlases/metrics are valid for your installation.
- Confirm input files are `.fz` or `.fib.gz` and readable.

## SciPy/.mat conversion warnings
- Install `scipy` if .mat → CSV conversion is needed; the curated venv already includes it.

## Cross-validation seed issues
- If `--from-bayes` is set, confirm the JSON has `best_parameters`.
- Keep `atlases` and `connectivity_values` fixed between Bayes and cross-validation to avoid metric-driven differences.

## Slow runs
- Reduce `tract_count`, or lower `--n-iterations` for Bayes demos.
- Use `--max-parallel 1` on constrained machines.

## Validation failures
- Run `python scripts/validate_setup.py --config configs/braingraph_default_config.json` to check environment and config schema.
