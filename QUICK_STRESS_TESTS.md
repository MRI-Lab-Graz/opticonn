# Quick Stress Test Execution Guide

## Setup

```bash
cd /data/local/software/braingraph-pipeline
source braingraph_pipeline/bin/activate

# Create test directories
mkdir -p /tmp/test_data
mkdir -p /tmp/test_configs
mkdir -p /tmp/test_output
```

---

## Quick Tests (5-10 minutes each)

### Test 1: Missing/Invalid Input Data ⚡
```bash
echo "=== Test 1a: Non-existent directory ==="
python opticonn.py bayesian --data-dir /nonexistent/path/ \
  --output-dir /tmp/test_output/test1a --config configs/sweep_nano.json --n-iterations 1

echo -e "\n=== Test 1b: Empty directory ==="
mkdir -p /tmp/empty_test
python opticonn.py bayesian --data-dir /tmp/empty_test \
  --output-dir /tmp/test_output/test1b --config configs/sweep_nano.json --n-iterations 1

echo -e "\n=== Test 1c: Wrong file types ==="
mkdir -p /tmp/wrong_files
touch /tmp/wrong_files/test.txt
python opticonn.py bayesian --data-dir /tmp/wrong_files \
  --output-dir /tmp/test_output/test1c --config configs/sweep_nano.json --n-iterations 1
```

**Expected Result**: Clear error messages about missing .fz files ✅

---

### Test 2: Configuration Validation ⚡
```bash
echo "=== Test 2a: Invalid JSON ==="
cat > /tmp/test_configs/bad.json << 'EOF'
{
  "atlases": ["FreeSurferDKT_Cortical",
  "connectivity_values": ["count"]
}
EOF
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/test2a --config /tmp/test_configs/bad.json --n-iterations 1

echo -e "\n=== Test 2b: Missing required field ==="
cat > /tmp/test_configs/incomplete.json << 'EOF'
{
  "connectivity_values": ["count"],
  "tract_count": 10000
}
EOF
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/test2b --config /tmp/test_configs/incomplete.json --n-iterations 1

echo -e "\n=== Test 2c: Invalid atlas name ==="
cat > /tmp/test_configs/bad_atlas.json << 'EOF'
{
  "atlases": ["InvalidAtlas"],
  "connectivity_values": ["count"],
  "tract_count": 10000
}
EOF
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/test2c --config /tmp/test_configs/bad_atlas.json --n-iterations 1
```

**Expected Result**: Clear validation errors with suggestions ✅

---

### Test 3: Parameter Validation ⚡
```bash
echo "=== Test 3a: Inverted parameter range (min > max) ==="
cat > /tmp/test_configs/inverted_range.json << 'EOF'
{
  "atlases": ["FreeSurferDKT_Cortical"],
  "connectivity_values": ["count"],
  "tract_count": 10000,
  "tracking_parameters": {
    "fa_threshold": [0.3, 0.05]
  }
}
EOF
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/test3a --config /tmp/test_configs/inverted_range.json --n-iterations 1

echo -e "\n=== Test 3b: All fixed parameters (no optimization) ==="
cat > /tmp/test_configs/all_fixed.json << 'EOF'
{
  "atlases": ["FreeSurferDKT_Cortical"],
  "connectivity_values": ["count"],
  "tract_count": 10000,
  "tracking_parameters": {
    "fa_threshold": [0.1],
    "turning_angle": [45.0]
  }
}
EOF
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/test3b --config /tmp/test_configs/all_fixed.json --n-iterations 1
```

**Expected Result**: Parameters validated, fixed params handled correctly ✅

---

### Test 4: DSI Studio Path Validation ⚡
```bash
echo "=== Test 4a: Missing DSI_STUDIO_PATH ==="
unset DSI_STUDIO_PATH
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/test4a --config configs/sweep_nano.json --n-iterations 1

# Re-export for other tests
export DSI_STUDIO_PATH=/data/local/software/dsi-studio/dsi_studio

echo -e "\n=== Test 4b: Wrong executable path ==="
export DSI_STUDIO_PATH=/bin/ls
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/test4b --config configs/sweep_nano.json --n-iterations 1

# Fix for remaining tests
export DSI_STUDIO_PATH=/data/local/software/dsi-studio/dsi_studio

echo -e "\n=== Test 4c: Non-executable file ==="
cp /data/local/software/dsi-studio/dsi_studio /tmp/dsi_noexec
chmod -x /tmp/dsi_noexec
export DSI_STUDIO_PATH=/tmp/dsi_noexec
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/test4c --config configs/sweep_nano.json --n-iterations 1

# Restore correct path
export DSI_STUDIO_PATH=/data/local/software/dsi-studio/dsi_studio
```

**Expected Result**: Clear errors about DSI Studio not being found/executable ✅

---

### Test 5: Invalid Iteration Counts ⚡
```bash
echo "=== Test 5a: Zero iterations ==="
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/test5a --config configs/sweep_nano.json --n-iterations 0

echo -e "\n=== Test 5b: Negative iterations ==="
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/test5b --config configs/sweep_nano.json --n-iterations -5
```

**Expected Result**: Validation error: iterations must be >= 1 ✅

---

### Test 6: Invalid Worker Counts ⚡
```bash
echo "=== Test 6a: Zero workers ==="
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/test6a --config configs/sweep_nano.json \
  --n-iterations 2 --max-workers 0

echo -e "\n=== Test 6b: Negative workers ==="
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/test6b --config configs/sweep_nano.json \
  --n-iterations 2 --max-workers -5

echo -e "\n=== Test 6c: Workers > available workers ==="
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/test6c --config configs/sweep_nano.json \
  --n-iterations 2 --max-workers 1000
```

**Expected Result**: Workers validated and adjusted if needed ✅

---

## Full Run Test (Verify Everything Works)

```bash
echo "=== Full Smoke Test: 2 iterations ==="
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/smoke_test --config configs/sweep_nano.json \
  --n-iterations 2 --max-workers 2 --verbose 2>&1 | tee /tmp/test_output/smoke_test.log

# Check results
echo -e "\n=== Check Results ==="
if [ -f /tmp/test_output/smoke_test/bayesian_optimization_results.json ]; then
  echo "✅ Results file created"
  cat /tmp/test_output/smoke_test/bayesian_optimization_results.json | python -m json.tool | head -30
else
  echo "❌ Results file not found"
fi
```

---

## Quick Validation Checks

```bash
# Check setup
echo "=== Setup Validation ==="
python scripts/validate_setup.py --config configs/sweep_nano.json

# Check data availability
echo -e "\n=== Data Availability ==="
ls -lh /data/local/Poly/derivatives/meta/fz/ | head -10
echo "Total subjects: $(ls /data/local/Poly/derivatives/meta/fz/*.fz 2>/dev/null | wc -l)"

# Check DSI Studio
echo -e "\n=== DSI Studio Check ==="
echo "DSI_STUDIO_PATH: $DSI_STUDIO_PATH"
$DSI_STUDIO_PATH --version 2>&1 | head -3
```

---

## Automated Test Runner Script

Save as `run_stress_tests.sh`:

```bash
#!/bin/bash
set -e

cd /data/local/software/braingraph-pipeline
source braingraph_pipeline/bin/activate

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_LOG="/tmp/test_output/stress_tests_${TIMESTAMP}.log"
mkdir -p /tmp/test_output

echo "Starting stress tests at $(date)" | tee "$TEST_LOG"

# Track results
PASS=0
FAIL=0

run_test() {
    local test_name="$1"
    local test_cmd="$2"
    local expect_fail="$3"
    
    echo -e "\n>>> Running: $test_name" | tee -a "$TEST_LOG"
    
    if eval "$test_cmd" &>> "$TEST_LOG"; then
        if [ "$expect_fail" = "true" ]; then
            echo "❌ FAIL - Expected to fail but succeeded" | tee -a "$TEST_LOG"
            ((FAIL++))
        else
            echo "✅ PASS" | tee -a "$TEST_LOG"
            ((PASS++))
        fi
    else
        if [ "$expect_fail" = "true" ]; then
            echo "✅ PASS - Failed as expected" | tee -a "$TEST_LOG"
            ((PASS++))
        else
            echo "❌ FAIL - Unexpected error" | tee -a "$TEST_LOG"
            ((FAIL++))
        fi
    fi
}

# Run tests
run_test "Missing directory" \
    "python opticonn.py bayesian --data-dir /nonexistent --output-dir /tmp/test_out_1 --config configs/sweep_nano.json --n-iterations 1" \
    "true"

run_test "Invalid config JSON" \
    "python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ --output-dir /tmp/test_out_2 --config /nonexistent/config.json --n-iterations 1" \
    "true"

run_test "Zero iterations" \
    "python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ --output-dir /tmp/test_out_3 --config configs/sweep_nano.json --n-iterations 0" \
    "true"

run_test "Valid run (2 iterations)" \
    "python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ --output-dir /tmp/test_out_4 --config configs/sweep_nano.json --n-iterations 2 --max-workers 2" \
    "false"

echo -e "\n==================" | tee -a "$TEST_LOG"
echo "Test Results:" | tee -a "$TEST_LOG"
echo "  PASSED: $PASS" | tee -a "$TEST_LOG"
echo "  FAILED: $FAIL" | tee -a "$TEST_LOG"
echo "  Log: $TEST_LOG" | tee -a "$TEST_LOG"
echo "==================" | tee -a "$TEST_LOG"

exit $FAIL
```

Run with:
```bash
chmod +x run_stress_tests.sh
./run_stress_tests.sh
```

---

## Test Results Tracking

After each test, check:

| Aspect | Check | Expected |
|--------|-------|----------|
| **Exit Code** | `echo $?` | 0 for success, 1 for expected errors |
| **Log Output** | `tail -20 /tmp/test_output/test.log` | Clear error messages, no tracebacks |
| **Error Message** | Read stderr | User-friendly, not technical |
| **Suggestions** | Look for hints | "Did you mean...", "Try this..." |
| **Recovery** | Can retry easily? | Yes, with simple command change |

---

## Current Status

### ✅ Already Working
- Basic pipeline execution
- DSI Studio path resolution
- Configuration validation (JSON parsing)
- Fixed parameter handling
- Results output with atlas info

### ⚠️ Needs Testing
- Error messages clarity
- Validation error reporting
- Edge cases (empty dirs, wrong files)
- Parameter range validation
- Worker count validation

### 🔴 Not Yet Implemented
- Input data pre-validation
- Graceful Ctrl+C handling
- Disk space check
- Worker crash recovery
- Did-you-mean suggestions

---

## Next Steps

After running these tests, create issues/improvements for:
1. Any unexpected errors or crashes
2. Unclear error messages
3. Missing validation checks
4. Improved user experience
5. Better documentation
