# Configuration

Key configs:
- `configs/demo_config.json`: small, fast defaults for demos.
- `configs/braingraph_default_config.json`: fuller defaults for real runs.
- `configs/dsi_studio_config_schema.json`: validation schema for extraction configs.

Important fields:
- `dsi_studio_cmd`: absolute path to DSI Studio executable.
- `atlases`, `connectivity_values`: keep fixed between Bayes and cross-validation to avoid metric-induced differences.
- `tract_count`, `tracking_parameters`: parameters Bayes and cross-validation tune.
- `connectivity_options.connectivity_threshold`: tuned in Bayes; propagated into apply/cross-val when seeded.

Validation:
- `scripts/run_pipeline.py`, `scripts/opticonn_hub.py`, and the demos call JSON validation before running.
- If DSI Studio is missing, errors include the download link.
