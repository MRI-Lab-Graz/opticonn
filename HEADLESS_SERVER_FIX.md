# Headless Server Fix for DSI Studio

## Problem
When running on a remote server without GUI/display, DSI Studio was failing with:
```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
This application failed to start because no Qt platform plugin could be initialized.
Aborted (core dumped)
```

This caused **all atlas processing to fail** during Bayesian optimization, resulting in:
- Only network_measures files being generated (no connectivity matrices)
- QA scores based on incomplete data
- Optimization results being invalid

## Root Cause
DSI Studio uses Qt framework which tries to connect to a display server by default. On headless systems, this fails immediately with exit code 134.

## Solution
Enable Qt offscreen rendering mode by setting environment variable:
```bash
export QT_QPA_PLATFORM=offscreen
```

## Fixes Applied

### 1. scripts/utils/runtime.py
Modified `propagate_no_emoji()` function to add `QT_QPA_PLATFORM=offscreen` to environment for all DSI Studio subprocess calls.

**Location:** Lines 172-179
**Impact:** All direct DSI Studio calls from extraction scripts

### 2. scripts/bayesian_optimizer.py  
Modified `run_with_spinner()` function to add `QT_QPA_PLATFORM=offscreen` when spawning the pipeline.

**Location:** Lines 332-333 (added)
**Impact:** All Bayesian optimization pipeline runs

### 3. Storage Configuration
The system already uses `/data/local/tmp_big` for temporary files (not `/tmp` which is small).

**Location:** scripts/bayesian_optimizer.py line 149
**Impact:** Prevents out-of-disk errors during optimization

## Verification

### Test DSI Studio manually:
```bash
export QT_QPA_PLATFORM=offscreen
/data/local/software/dsi-studio/dsi_studio --action=trk \
  --source=/data/local/Poly/derivatives/meta/fz/P040_149.fz \
  --tract_count=100000 \
  --connectivity=FreeSurferDKT_Cortical \
  --connectivity_value=count \
  --connectivity_type=pass \
  --connectivity_threshold=0.001 \
  --connectivity_output=matrix,measure \
  --thread_count=4 \
  --output=/tmp/test_output.tt.gz \
  --export=stat
```

Expected result:
- Exit code: 0
- Output files created successfully
- No Qt error messages

## Next Steps

After applying these fixes, you should:

1. **Clear Python cache**: `rm -rf scripts/__pycache__ scripts/utils/__pycache__`
2. **Re-run Bayesian optimization** with the fixed code
3. **Verify all atlases process successfully** (check extraction logs)
4. **Validate connectivity matrices are generated** (not just network_measures)

## Testing Command

```bash
cd /data/local/software/braingraph-pipeline
PYTHONPATH=/data/local/software/braingraph-pipeline python scripts/bayesian_optimizer.py \
  --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir final_run_fixed \
  --config configs/sweep_production_full.json \
  --n-iterations 2 \
  --verbose
```

This should show:
- DSI Studio processing all 15 atlases
- All files created without Qt errors
- Proper connectivity matrices in output directories
