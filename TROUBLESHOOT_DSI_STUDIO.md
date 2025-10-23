# Troubleshooting: DSI Studio Connectivity Extraction

## Issue
DSI Studio is running but not producing `.count..pass.connectivity.mat` files

## Cause
Likely one of:
1. DSI Studio command format issue
2. Missing connectivity threshold parameter
3. DSI Studio version incompatibility
4. Parameter serialization issue

---

## Quick Diagnostic: Run Single Subject Manually

```bash
cd /data/local/software/braingraph-pipeline
source braingraph_pipeline/bin/activate

# Create test directory
mkdir -p /tmp/test_single
cp /data/local/Poly/derivatives/meta/fz/P040_105.fz /tmp/test_single/

# Run single pipeline step manually
python scripts/run_pipeline.py \
  --data-dir /tmp/test_single \
  --output /tmp/test_single_output \
  --config configs/sweep_ultra_minimal.json \
  --step extract \
  --verbose \
  --no-emoji
```

This will show exactly what's failing.

---

## Alternative: Use sweep_nano.json Instead

The `sweep_ultra_minimal.json` might be too aggressive. Try:

```bash
python opticonn.py sweep \
  -i /data/local/Poly/derivatives/meta/fz/ \
  -o /tmp/demo_sweep_nano \
  --extraction-config configs/sweep_nano.json \
  --subjects 1 \
  --no-emoji \
  --verbose
```

---

## Known Working Configuration

If sweep_nano fails too, use the default with fewer tracts:

Create `/tmp/test_config.json`:
```json
{
  "description": "Test config with proven settings",
  "dsi_studio_cmd": "dsi_studio",
  "atlases": ["FreeSurferDKT_Cortical"],
  "connectivity_values": ["count", "fa"],
  "tract_count": 5000,
  "thread_count": 4,
  "tracking_parameters": {
    "method": 0,
    "otsu_threshold": 0.4,
    "fa_threshold": 0.1,
    "turning_angle": 0.0,
    "step_size": 0.0,
    "smoothing": 0.0,
    "min_length": 10,
    "max_length": 200,
    "track_voxel_ratio": 3.0,
    "check_ending": 0,
    "random_seed": 0,
    "dt_threshold": 0.1
  },
  "connectivity_options": {
    "connectivity_type": "pass",
    "connectivity_threshold": 0.00001,
    "connectivity_output": "matrix"
  },
  "sweep_parameters": {
    "otsu_range": [0.4],
    "fa_threshold_range": [0.1],
    "min_length_range": [10],
    "track_voxel_ratio_range": [3.0],
    "connectivity_threshold_range": [0.00001],
    "tract_count_range": [5000, 10000],
    "sampling": {
      "method": "grid",
      "n_samples": 2,
      "random_seed": 42
    }
  }
}
```

Then:
```bash
python opticonn.py sweep \
  -i /data/local/Poly/derivatives/meta/fz/ \
  -o /tmp/demo_sweep_test2 \
  --extraction-config /tmp/test_config.json \
  --subjects 1 \
  --no-emoji
```

---

## What We Learned From This Run

✅ Good news:
- DSI Studio is installed and working
- Configuration parsing works
- File staging works
- Bootstrap wave creation works
- Sweep parameter generation works

❌ Problem:
- Connectivity matrix output format issue
- DSI Studio not creating expected `.count..pass.connectivity.mat` files

---

## Recommended Next Step

Run the diagnostic:
```bash
python scripts/run_pipeline.py \
  --data-dir /tmp/test_single \
  --output /tmp/test_single_output \
  --config configs/sweep_ultra_minimal.json \
  --step extract \
  --verbose \
  --no-emoji
```

This will tell us exactly which DSI Studio command is failing and why.

