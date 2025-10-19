# Validation Improvements - Bulletproof Pipeline

## Summary

The Bayesian optimizer now includes comprehensive validation that **stops immediately** with clear error messages when issues are detected. No more silent failures or wasted computation!

---

## Input Data Validation

### ✅ Data Directory Exists
```bash
python opticonn.py bayesian --data-dir /nonexistent/path/ ...
```
**Error:**
```
ERROR - ❌ Data directory does not exist: /nonexistent/path
ERROR -    Please create the directory or check the path.
```

### ✅ Data Directory is Not Empty
```bash
python opticonn.py bayesian --data-dir /tmp/empty_dir/ ...
```
**Error:**
```
ERROR - ❌ No .fz or .fib.gz files found in: /tmp/empty_dir/
ERROR -    Expected to find tractography data files (.fz or .fib.gz)
ERROR -    Found: 0 other files
```

### ✅ Data Directory Contains Correct File Types
```bash
python opticonn.py bayesian --data-dir /dir_with_csv/ ...
```
**Error:**
```
ERROR - ❌ No .fz or .fib.gz files found in: /dir_with_csv/
ERROR -    Expected to find tractography data files (.fz or .fib.gz)
ERROR -    Found: 5 other files
ERROR -    Sample files in directory:
ERROR -      - data1.csv
ERROR -      - data2.txt
```

---

## Configuration Validation

### ✅ Valid JSON Syntax
```bash
python opticonn.py bayesian ... --config /tmp/bad.json
```
Where bad.json has syntax errors (missing comma, bracket mismatch, etc.)

**Error:**
```
ERROR - ❌ Configuration validation failed for /tmp/bad.json:
ERROR -    • Invalid JSON in configuration file: /tmp/bad.json
ERROR - 
💡 Suggested fixes:
ERROR -    • Fix JSON syntax errors in configuration file
```

### ✅ Required Fields Present
```json
{
  "connectivity_values": ["count"],
  "tract_count": 10000
}
```

**Error:**
```
ERROR - ❌ Configuration validation failed:
ERROR -    • ❌ 'atlases' field is required and must contain at least one atlas name
ERROR - 
💡 Suggested fixes:
ERROR -    • Add at least one atlas to the 'atlases' array
```

### ✅ Valid Atlas Names
```json
{
  "atlases": ["InvalidAtlas123"],
  "connectivity_values": ["count"]
}
```

**Error:**
```
ERROR - ❌ Configuration validation failed:
ERROR -    • Unknown atlas: InvalidAtlas123. Valid atlases: 
             ['AAL3', 'ATAG_basal_ganglia', 'BrainSeg', ...]
```

### ✅ Valid Connectivity Metrics
```json
{
  "connectivity_values": ["invalid_metric"]
}
```

**Error:**
```
ERROR - ❌ Configuration validation failed:
ERROR -    • Unknown connectivity metric: invalid_metric. 
             Valid metrics: ['ad', 'count', 'fa', 'iso', 'md', 'ncount2', 'qa', 'rd']
```

---

## Parameter Validation

### ✅ Parameter Ranges Not Inverted
```json
{
  "tracking_parameters": {
    "fa_threshold": [0.3, 0.05]
  }
}
```

**Error:**
```
ERROR - ❌ Configuration validation failed:
ERROR -    • fa_threshold range inverted: min=0.3 > max=0.05. 
             Should be [0.05, 0.3]
```

### ✅ Parameter Values Within Valid Bounds
```json
{
  "tracking_parameters": {
    "fa_threshold": [1.5, 2.0]
  }
}
```

**Error:**
```
ERROR - ❌ Configuration validation failed:
ERROR -    • fa_threshold maximum must be <= 1.0, got 2.0
```

### ✅ Constraint Checking
- `fa_threshold`: 0.0 - 1.0
- `otsu_threshold`: 0.0 - 1.0
- `turning_angle`: 0.0 - 90.0 degrees

---

## Iteration & Worker Validation

### ✅ Iteration Count Valid
```bash
python opticonn.py bayesian ... --n-iterations 0
```

**Error:**
```
ERROR - ❌ Number of iterations must be >= 1, got 0
```

Also works for negative values:
```bash
python opticonn.py bayesian ... --n-iterations -5
ERROR - ❌ Number of iterations must be >= 1, got -5
```

### ✅ Worker Count Valid
```bash
python opticonn.py bayesian ... --max-workers 0
```

**Error:**
```
ERROR - ❌ Number of workers must be >= 1, got 0
```

### ✅ Worker Count Capped to CPU Count
```bash
python opticonn.py bayesian ... --max-workers 1000
```

**Warning:**
```
WARNING - ⚠️  Requested 1000 workers but only 32 CPUs available
WARNING -    Capping workers to 32
```

---

## DSI Studio Path Validation

### ✅ DSI Studio Path Exists
```bash
unset DSI_STUDIO_PATH
python opticonn.py bayesian ...
```

**Error:**
```
ERROR - ❌ Configuration validation failed:
ERROR -    • DSI Studio executable not found: dsi_studio
ERROR - 
💡 Suggested fixes:
ERROR -    • Update dsi_studio_cmd path to correct DSI Studio executable location
```

---

## Validation Flow

```
┌─ Command Arguments ─┐
│  bayesian --data-dir /path/to/data
│           --config config.json
│           --n-iterations 10
│           --max-workers 4
└─────────────────────┘
           ↓
    ┌─────────────────┐
    │ Config File     │
    │ JSON Validation │───→ Invalid JSON? ───→ ERROR
    └─────────────────┘
           ↓ Valid
    ┌─────────────────────┐
    │ Config Content      │
    │ Schema Validation   │───→ Missing field? ───→ ERROR
    │ (Atlas, metrics,    │───→ Invalid value? ──→ ERROR
    │  parameters)        │───→ Range inverted?─→ ERROR
    └─────────────────────┘
           ↓ Valid
    ┌─────────────────────┐
    │ Data Directory      │───→ Not exist?   ───→ ERROR
    │ Validation          │───→ Empty?       ───→ ERROR
    │ (Files, types)      │───→ Wrong types? ───→ ERROR
    └─────────────────────┘
           ↓ Valid
    ┌─────────────────────┐
    │ CLI Arguments       │───→ Invalid iter count? → ERROR
    │ Validation          │───→ Invalid worker count?→ ERROR
    │ (iter, workers)     │
    └─────────────────────┘
           ↓ Valid
    ┌─────────────────────┐
    │ ✅ START OPTIMIZATION│
    └─────────────────────┘
```

---

## Exit Codes

| Scenario | Exit Code | Message |
|----------|-----------|---------|
| Success | 0 | Optimization completed |
| Config invalid | 1 | Configuration validation failed |
| Data invalid | 1 | No data files found |
| Args invalid | 1 | Invalid iteration/worker count |
| Other error | 1 | Error message |

---

## User Experience Improvements

### Before
```
🚀 Starting Bayesian optimization...
   Data: /nonexistent/path/
   Output: /tmp/output
   Iterations: 1
WARNING - ⚠️  No .fz or .fib.gz files found in /nonexistent/path
WARNING - ⚠️  No subjects available for optimization
INFO - 
... (process continues and wastes time)
```

### After
```
🚀 Starting Bayesian optimization...
   Data: /nonexistent/path/
   Output: /tmp/output
   Iterations: 1
ERROR - ❌ Data directory does not exist: /nonexistent/path
ERROR -    Please create the directory or check the path.
❌ Bayesian optimization failed with error code 1
```

✅ **Immediate failure** with **clear guidance** on how to fix the issue!

---

## Testing Checklist

- [x] Missing data directory
- [x] Empty data directory
- [x] Wrong file types
- [x] Invalid JSON
- [x] Missing required fields (atlases)
- [x] Invalid atlas names
- [x] Invalid connectivity metrics
- [x] Inverted parameter ranges
- [x] Out-of-range parameter values
- [x] Zero iterations
- [x] Negative iterations
- [x] Zero workers
- [x] Too many workers
- [x] Missing DSI_STUDIO_PATH
- [x] Valid configuration passes

---

## Files Modified

1. **scripts/bayesian_optimizer.py**
   - Added data directory validation (exists, is_dir, contains .fz/.fib.gz)
   - Added JSONValidator integration
   - Added iteration count validation
   - Added worker count validation and capping

2. **scripts/json_validator.py**
   - Enhanced parameter range validation (handles [min, max] lists)
   - Added inverted range detection
   - Added required field validation
   - Added out-of-bounds checking

---

## Next: User Testing

Use `MANUAL_TEST_CHECKLIST.md` or `QUICK_STRESS_TESTS.md` to verify:

1. All error messages are clear
2. Suggestions are helpful
3. Exit codes are correct
4. Valid configs still work
5. No cryptic tracebacks

Then implement any improvements based on feedback!
