# Bayesian Optimizer - Stress Testing & Error Scenarios

## Overview
This document outlines comprehensive stress tests and error scenarios to identify and fix potential issues, making the pipeline bulletproof and user-friendly.

---

## Category 1: Input Data Stress Tests

### 1.1 Missing or Invalid Data Directory
**Test**: What happens when the data directory doesn't exist or is empty?
```bash
# Test 1a: Non-existent directory
python opticonn.py bayesian --data-dir /nonexistent/path/ --output-dir test_output \
  --config configs/sweep_nano.json --n-iterations 2

# Test 1b: Empty directory
mkdir -p /tmp/empty_data
python opticonn.py bayesian --data-dir /tmp/empty_data --output-dir test_output \
  --config configs/sweep_nano.json --n-iterations 2

# Test 1c: Directory with wrong file types
mkdir -p /tmp/wrong_files
touch /tmp/wrong_files/test.txt
python opticonn.py bayesian --data-dir /tmp/wrong_files --output-dir test_output \
  --config configs/sweep_nano.json --n-iterations 2
```

**Expected Behavior**: 
- ✅ Clear error message about no .fz/.fib.gz files found
- ✅ Suggestions on file format requirements
- ✅ Graceful exit with exit code 1

---

### 1.2 Insufficient Data (Not Enough Subjects)
**Test**: What if we have only 1-2 subjects but need 3+ for bootstrap validation?
```bash
# Create minimal test data with just 1 subject
mkdir -p /tmp/minimal_data
cp /data/local/Poly/derivatives/meta/fz/P100_1002.fz /tmp/minimal_data/

# Try to run
python opticonn.py bayesian --data-dir /tmp/minimal_data --output-dir test_output \
  --config configs/sweep_nano.json --n-iterations 2
```

**Expected Behavior**:
- ✅ Detect minimum subject count
- ✅ Warn user but continue (or adapt automatically)
- ✅ Adjust bootstrap strategy if needed

---

### 1.3 Corrupted Data Files
**Test**: What if .fz files are corrupted or truncated?
```bash
# Create truncated file (simulating corruption)
dd if=/data/local/Poly/derivatives/meta/fz/P100_1002.fz of=/tmp/corrupted.fz bs=1 count=1000

# Try to run
mkdir -p /tmp/corrupted_data
mv /tmp/corrupted.fz /tmp/corrupted_data/
python opticonn.py bayesian --data-dir /tmp/corrupted_data --output-dir test_output \
  --config configs/sweep_nano.json --n-iterations 1
```

**Expected Behavior**:
- ✅ Skip corrupted files with warning
- ✅ Continue with valid files
- ✅ Report which files were skipped

---

## Category 2: Configuration Stress Tests

### 2.1 Invalid Configuration File
**Test**: What happens with malformed JSON?
```bash
# Create invalid config
cat > /tmp/bad_config.json << 'EOF'
{
  "atlases": ["FreeSurferDKT_Cortical",
  "connectivity_values": ["count", "fa"],
  "tract_count": 10000
}
EOF

python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir test_output --config /tmp/bad_config.json --n-iterations 1
```

**Expected Behavior**:
- ✅ Clear JSON parsing error
- ✅ Point to line number of error
- ✅ Suggest valid config format

---

### 2.2 Missing Required Config Fields
**Test**: Config file missing critical fields
```bash
# Config missing atlases
cat > /tmp/incomplete_config.json << 'EOF'
{
  "connectivity_values": ["count", "fa"],
  "tract_count": 10000
}
EOF

python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir test_output --config /tmp/incomplete_config.json --n-iterations 1
```

**Expected Behavior**:
- ✅ List missing required fields
- ✅ Show example of valid field
- ✅ Point to config validation documentation

---

### 2.3 Invalid Parameter Ranges
**Test**: Parameter ranges with invalid values
```bash
# Config with invalid range (min > max)
cat > /tmp/invalid_range.json << 'EOF'
{
  "atlases": ["FreeSurferDKT_Cortical"],
  "connectivity_values": ["count"],
  "tract_count": 10000,
  "tracking_parameters": {
    "fa_threshold": [0.3, 0.05],
    "turning_angle": [90.0, 30.0]
  }
}
EOF

python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir test_output --config /tmp/invalid_range.json --n-iterations 1
```

**Expected Behavior**:
- ✅ Validate min < max for all ranges
- ✅ Auto-fix if possible (swap values)
- ✅ Clear error if not fixable

---

### 2.4 All Fixed Parameters (No Optimization)
**Test**: Config where all parameters are fixed (no optimization possible)
```bash
# Config with all fixed parameters (single values)
cat > /tmp/all_fixed.json << 'EOF'
{
  "atlases": ["FreeSurferDKT_Cortical"],
  "connectivity_values": ["count"],
  "tract_count": 10000,
  "tracking_parameters": {
    "fa_threshold": [0.1],
    "turning_angle": [45.0],
    "step_size": [1.0]
  }
}
EOF

python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir test_output --config /tmp/all_fixed.json --n-iterations 5
```

**Expected Behavior**:
- ✅ Detect no parameters to optimize
- ✅ Suggest adding ranges to parameters
- ✅ Graceful error with helpful message

---

## Category 3: DSI Studio Path Stress Tests

### 3.1 Missing DSI Studio
**Test**: What if DSI Studio path is not set?
```bash
# Unset DSI_STUDIO_PATH
unset DSI_STUDIO_PATH

python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir test_output --config configs/sweep_nano.json --n-iterations 1
```

**Expected Behavior**:
- ✅ Clear error message: "DSI_STUDIO_PATH not found"
- ✅ Instructions to reinstall with --dsi-path flag
- ✅ Alternative: check common installation paths

---

### 3.2 DSI Studio Path Points to Wrong Location
**Test**: What if DSI_STUDIO_PATH points to a different executable?
```bash
# Point to wrong executable
export DSI_STUDIO_PATH=/bin/ls

python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir test_output --config configs/sweep_nano.json --n-iterations 1
```

**Expected Behavior**:
- ✅ Validation catches this before execution
- ✅ Suggests running setup validator first
- ✅ Clear error: "DSI Studio test failed"

---

### 3.3 DSI Studio Not Executable
**Test**: What if file exists but isn't executable?
```bash
# Make copy and remove execute bit
cp /data/local/software/dsi-studio/dsi_studio /tmp/dsi_studio_noexec
chmod -x /tmp/dsi_studio_noexec
export DSI_STUDIO_PATH=/tmp/dsi_studio_noexec

python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir test_output --config configs/sweep_nano.json --n-iterations 1
```

**Expected Behavior**:
- ✅ Validation catches permission error
- ✅ Suggest: `chmod +x /path/to/dsi_studio`
- ✅ Clear message about permissions

---

## Category 4: Output Directory Stress Tests

### 4.1 Output Directory Already Exists
**Test**: What if output directory already has data?
```bash
# Create output dir with existing data
mkdir -p test_output/iterations
touch test_output/iterations/test.txt

python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir test_output --config configs/sweep_nano.json --n-iterations 2
```

**Expected Behavior**:
- ✅ Option 1: Ask user before overwriting
- ✅ Option 2: Auto-create iteration_0002, iteration_0003, etc.
- ✅ Add --force flag to overwrite without asking

---

### 4.2 Output Directory Not Writable
**Test**: What if user doesn't have write permissions?
```bash
# Create read-only directory
mkdir -p /tmp/readonly_output
chmod 444 /tmp/readonly_output

python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/readonly_output --config configs/sweep_nano.json --n-iterations 1
```

**Expected Behavior**:
- ✅ Check permissions before starting
- ✅ Clear error: "No write permission to output directory"
- ✅ Suggest creating new directory or fixing permissions

---

### 4.3 Disk Space Issues
**Test**: What if disk is full during execution?
```bash
# Simulate by running on a small tmpfs or filling disk
# (This is environment-dependent, but pipeline should handle gracefully)

# Check before starting
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir test_output --config configs/sweep_nano.json --n-iterations 5
# Scenario: Disk fills during iteration 3
```

**Expected Behavior**:
- ✅ Pre-flight check: estimate space needed
- ✅ Warn if insufficient disk space
- ✅ Catch disk full errors during execution
- ✅ Save progress and exit gracefully

---

## Category 5: Parallel Execution Stress Tests

### 5.1 Too Many Workers
**Test**: What if user specifies more workers than subjects?
```bash
python opticonn.py bayesian --data-dir /tmp/minimal_data --output-dir test_output \
  --config configs/sweep_nano.json --n-iterations 2 --max-workers 100
```

**Expected Behavior**:
- ✅ Detect workers > subjects
- ✅ Auto-adjust to (subjects - 1)
- ✅ Warn user: "Adjusted workers from 100 to 2"

---

### 5.2 Invalid Worker Count
**Test**: Negative or zero workers
```bash
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir test_output --config configs/sweep_nano.json \
  --n-iterations 2 --max-workers 0

python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir test_output --config configs/sweep_nano.json \
  --n-iterations 2 --max-workers -5
```

**Expected Behavior**:
- ✅ Validate: workers >= 1
- ✅ Clear error with valid range
- ✅ Default to safe value (CPU count - 1)

---

### 5.3 Worker Crash/Timeout
**Test**: What if a worker process crashes or hangs?
```bash
# This requires special test setup, but pipeline should handle:
# - Catch worker timeouts
# - Detect process exit codes
# - Retry logic
# - Graceful degradation

python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir test_output --config configs/sweep_nano.json --n-iterations 2 --timeout 5
```

**Expected Behavior**:
- ✅ Set reasonable timeout per subject
- ✅ Detect worker crashes
- ✅ Log which subject/atlas failed
- ✅ Retry or skip and continue
- ✅ Report summary at end

---

## Category 6: Parameter Edge Cases

### 6.1 Single Value Parameters (Fixed)
**Test**: Parameters with range of [value, value]
```bash
# Already tested but verify output clarity
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir test_output --config configs/sweep_nano.json --n-iterations 2
# Verify that min_length=10 shows as "(fixed)" not as optimized range
```

**Expected Behavior**:
- ✅ Display "(fixed)" for parameters with range [x, x]
- ✅ Don't attempt to optimize fixed parameters
- ✅ Show in final results clearly

---

### 6.2 Extreme Parameter Values
**Test**: Very large or very small values
```bash
# Config with extreme values
cat > /tmp/extreme_params.json << 'EOF'
{
  "atlases": ["FreeSurferDKT_Cortical"],
  "connectivity_values": ["count"],
  "tract_count": 10000,
  "tracking_parameters": {
    "fa_threshold": [0.0001, 0.99999],
    "turning_angle": [0.1, 179.9],
    "step_size": [0.001, 100.0],
    "tract_count": [1000, 10000000]
  }
}
EOF

python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir test_output --config /tmp/extreme_params.json --n-iterations 2
```

**Expected Behavior**:
- ✅ Accept valid ranges even if extreme
- ✅ Warn if values are outside typical ranges
- ✅ Let user decide to continue

---

## Category 7: Iteration & Progress Stress Tests

### 7.1 Zero or Negative Iterations
**Test**: Invalid iteration counts
```bash
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir test_output --config configs/sweep_nano.json --n-iterations 0

python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir test_output --config configs/sweep_nano.json --n-iterations -5
```

**Expected Behavior**:
- ✅ Validate: n-iterations >= 1
- ✅ Clear error: "Iterations must be >= 1"
- ✅ Show default suggestion (e.g., 5)

---

### 7.2 Very Large Iteration Count
**Test**: User accidentally requests huge number of iterations
```bash
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir test_output --config configs/sweep_nano.json --n-iterations 10000
```

**Expected Behavior**:
- ✅ Estimate computation time
- ✅ Show total time in summary
- ✅ Ask user to confirm before starting
- ✅ Allow cancellation with Ctrl+C

---

### 7.3 Interruption (Ctrl+C)
**Test**: User presses Ctrl+C during execution
```bash
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir test_output --config configs/sweep_nano.json --n-iterations 100
# Press Ctrl+C during iteration 5
```

**Expected Behavior**:
- ✅ Catch SIGINT gracefully
- ✅ Save all progress so far
- ✅ Save checkpoint for resume
- ✅ Message: "Interrupted. Results saved to..."
- ✅ Allow resume with: `--resume final_run_enhanced`

---

## Category 8: Validation & Format Stress Tests

### 8.1 Unusual Atlas Names
**Test**: Non-existent or misspelled atlas names
```bash
cat > /tmp/bad_atlas.json << 'EOF'
{
  "atlases": ["InvalidAtlas", "FreeSurferDKT_Cortical"],
  "connectivity_values": ["count"],
  "tract_count": 10000
}
EOF

python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir test_output --config /tmp/bad_atlas.json --n-iterations 1
```

**Expected Behavior**:
- ✅ Validate atlas names before starting
- ✅ List available atlases
- ✅ Did-you-mean suggestion: "Did you mean 'FreeSurferDKT_Cortical'?"
- ✅ Clear error with available options

---

### 8.2 Invalid Connectivity Metrics
**Test**: Non-existent connectivity metric
```bash
cat > /tmp/bad_metric.json << 'EOF'
{
  "atlases": ["FreeSurferDKT_Cortical"],
  "connectivity_values": ["invalid_metric", "count"],
  "tract_count": 10000
}
EOF

python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir test_output --config /tmp/bad_metric.json --n-iterations 1
```

**Expected Behavior**:
- ✅ Validate metrics before starting
- ✅ List available metrics: count, fa, qa, ncount2, md, ad, rd, iso
- ✅ Clear suggestion which metric was misspelled

---

## Summary: Bulletproofing Improvements Needed

### High Priority (Must Fix)
- [x] Configuration validation (required fields, valid values)
- [x] DSI Studio path resolution
- [x] Fixed vs optimized parameter handling
- [ ] Graceful handling of missing/corrupted input files
- [ ] Disk space pre-check
- [ ] Worker crash handling and retry logic
- [ ] Graceful Ctrl+C interruption with progress save

### Medium Priority (Should Fix)
- [ ] Ask before overwriting existing output
- [ ] Estimate computation time and confirm
- [ ] Better error messages with suggestions
- [ ] Validate parameter ranges (min < max)
- [ ] Auto-adjust workers if > subjects
- [ ] Did-you-mean suggestions for atlas/metric names

### Low Priority (Nice to Have)
- [ ] Resume from checkpoint
- [ ] Parallel execution safety checks
- [ ] Progress bar improvements
- [ ] Real-time result monitoring
- [ ] Email notifications on completion

---

## How to Run All Stress Tests

```bash
#!/bin/bash
cd /data/local/software/braingraph-pipeline
source braingraph_pipeline/bin/activate

# Run each test
echo "Testing missing data directory..."
python opticonn.py bayesian --data-dir /nonexistent/path/ --output-dir test_output \
  --config configs/sweep_nano.json --n-iterations 1

echo "Testing invalid config..."
# ... (run all tests from above)

echo "All stress tests completed!"
```

---

## User Experience Improvements

### Before (Current)
```
❌ Some error occurred
Traceback: ...
```

### After (Desired)
```
❌ Configuration Error
   Missing required field: "atlases"
   
   Example valid config:
   {
     "atlases": ["FreeSurferDKT_Cortical", "Brainnectome"],
     "connectivity_values": ["count", "fa"],
     "tract_count": 10000
   }
   
   ℹ️  For more info: python opticonn.py bayesian --help
```

