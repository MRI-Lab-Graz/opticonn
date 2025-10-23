# ✅ Sweep Demo Status: EXTRACTION WORKING!

## Good News
✅ **DSI Studio connectivity extraction is WORKING!**
- Connectivity matrices are being created successfully  
- Files being generated: `.count..pass.connectivity.csv`
- Data is valid (tract count values present)

## Issue Identified
❌ **Step 02 (Network Measures Aggregation) fails to find Step 01 outputs**
- The aggregation script looks for network_measures.csv
- File locations are in nested subject directories  
- File discovery logic doesn't handle the current directory structure

## Impact
- Sweeps cannot complete because Step 02 fails
- Step 01 (extraction) works fine
- Step 03 (selection) never runs because Step 02 fails

---

## Root Cause

The aggregation script expects files in:
```
01_connectivity/
  network_measures.csv  ← Looking here
```

But files are actually in:
```
01_connectivity/
  P040_105_20251023_135553/
    tracks_1k_streamline_fa0.10/
      results/
        FreeSurferDKT_Cortical/
          *.count..pass.connectivity.csv  ← Files are here
```

---

## Solution Options

### Option 1: Quick Fix for Testing (Recommended)
Modify `aggregate_network_measures.py` to search recursively for CSV files

### Option 2: Workaround - Manual Aggregation
Create a simple script to collect and aggregate the CSV files:

```python
import glob
import pandas as pd
from pathlib import Path

combo_dir = Path("/tmp/demo_sweep_test/sweep-c6f8e9a6-b055-4208-879e-36430e689299/optimize/bootstrap_qa_wave_1/combos/sweep_0001")

# Find all connectivity CSVs
csv_files = list(combo_dir.glob("01_connectivity/**/results/*/*.simple.csv"))
print(f"Found {len(csv_files)} CSV files")

if csv_files:
    # Read and aggregate
    dfs = [pd.read_csv(f, header=None) for f in csv_files]
    combined = pd.concat(dfs, axis=0, ignore_index=True)
    combined.to_csv(combo_dir / "01_connectivity" / "network_measures.csv", index=False, header=False)
    print("✅ Aggregated CSV saved!")
```

### Option 3: Use Different Config
Some configs might have different directory structures. Try:
```bash
python opticonn.py sweep \
  -i /data/local/Poly/derivatives/meta/fz/ \
  -o /tmp/demo_sweep_fixed \
  --extraction-config configs/braingraph_default_config.json \
  --subjects 1 \
  --no-emoji
```

---

## Next Steps

1. **Option 1 (Recommended)**: Fix the aggregation script
   - Edit: `scripts/aggregate_network_measures.py`  
   - Add recursive search for CSV files
   - ~5 minutes to implement

2. **Option 2 (Quick)**: Use manual aggregation workaround
   - Run manual aggregation after extraction
   - Then sweep should complete

3. **Option 3 (Alternative)**: Test with existing configs
   - These may have different directory structures
   - Might avoid the aggregation issue

---

## Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Setup validation | ✅ Pass | DSI Studio found, config valid |
| Data staging | ✅ Pass | Files copied correctly |
| Bootstrap waves | ✅ Pass | 2 waves created |
| Parameter sweep | ✅ Pass | 2 combos generated |
| Step 01: Extraction | ✅ Pass | Connectivity CSVs created |
| Step 02: Aggregation | ❌ Fail | File discovery issue |
| Step 03: Selection | ⏸️ Blocked | Needs Step 02 |
| QA Validation | ⏸️ Blocked | Needs Step 02 |

---

## Proposed Fix

The extraction is working perfectly. We just need to fix how Step 02 finds the output from Step 01.

The issue is in a file discovery loop. Would you like me to:
1. Fix the `aggregate_network_measures.py` script? (5 min)
2. Try a different existing configuration? (2 min)
3. Run manual post-processing? (5 min)

