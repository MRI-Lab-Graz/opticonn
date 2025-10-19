# Stress Testing Plan Summary

## Overview

To make the pipeline bulletproof and user-friendly, we've identified **8 main stress test categories** with **30+ specific test cases** covering edge cases and error scenarios.

---

## The 8 Stress Test Categories

### 1️⃣ Input Data Issues (3 tests)
- Missing/non-existent directory
- Empty directory  
- Wrong file types
- Corrupted files

**Goal**: Pipeline handles missing/bad data gracefully

---

### 2️⃣ Configuration Issues (4 tests)
- Invalid JSON syntax
- Missing required fields
- Invalid parameter ranges
- All fixed parameters (no optimization)

**Goal**: User gets clear feedback on config problems

---

### 3️⃣ DSI Studio Path Issues (3 tests)
- Missing DSI_STUDIO_PATH
- Wrong executable path
- File not executable

**Goal**: User knows exactly how to fix DSI Studio problems

---

### 4️⃣ Output Directory Issues (3 tests)
- Directory already exists
- No write permissions
- Disk space full

**Goal**: Pipeline handles output gracefully

---

### 5️⃣ Parallel Execution Issues (3 tests)
- Too many workers
- Invalid worker count (0, negative)
- Worker crashes/timeouts

**Goal**: Robust parallel execution with graceful fallback

---

### 6️⃣ Parameter Validation (2 tests)
- Fixed vs optimized parameters
- Extreme parameter values

**Goal**: Parameters displayed correctly in results

---

### 7️⃣ Iteration & Progress Issues (3 tests)
- Zero/negative iterations
- Huge iteration count (time warning)
- User interruption (Ctrl+C)

**Goal**: User can run and stop runs easily

---

### 8️⃣ Format & Validation (2 tests)
- Invalid atlas names
- Invalid connectivity metrics

**Goal**: Did-you-mean suggestions for common errors

---

## Quick Start: Run the Tests

### Minimal Test (5 minutes)
```bash
cd /data/local/software/braingraph-pipeline
source braingraph_pipeline/bin/activate

# Test missing directory
python opticonn.py bayesian --data-dir /nonexistent --output-dir /tmp/out \
  --config configs/sweep_nano.json --n-iterations 1

# Test invalid config
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/out2 --config /nonexistent.json --n-iterations 1

# Test zero iterations
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/out3 --config configs/sweep_nano.json --n-iterations 0
```

### Full Test Suite (30 minutes)
See `QUICK_STRESS_TESTS.md` for the complete test runner

---

## Key Bulletproofing Improvements

### Must Have (High Priority)
- ✅ Configuration validation (done)
- ✅ DSI Studio path resolution (done)
- ❌ Input data validation (missing)
- ❌ Graceful error handling (needs improvement)
- ❌ Disk space pre-check (missing)
- ❌ Worker crash recovery (missing)

### Should Have (Medium Priority)
- ❌ Confirmation before overwriting
- ❌ Computation time estimate
- ❌ Better error messages with suggestions
- ❌ Parameter range validation
- ❌ Auto-adjust worker count

### Nice to Have (Low Priority)
- ❌ Checkpoint/resume feature
- ❌ Did-you-mean suggestions
- ❌ Real-time monitoring
- ❌ Email notifications

---

## What Each Test Checks

### Test Type 1: Error Messages
**Question**: Does user get a clear, actionable error message?

**Example**:
```
❌ BAD:
ERROR: No such file or directory
Traceback: ...

✅ GOOD:
❌ Input Error: No .fz files found in /nonexistent
   
   Please provide a directory containing .fz or .fib.gz files
   Example: /data/local/Poly/derivatives/meta/fz/
   
   📁 Current directory has:
      (none)
```

---

### Test Type 2: Validation Before Execution
**Question**: Does pipeline catch errors before wasting time?

**Example**:
```
✅ Pre-flight Checks:
   ✓ Configuration is valid JSON
   ✓ All required fields present
   ✓ Atlas names are valid
   ✓ Connectivity metrics are valid
   ✓ Parameter ranges are valid (min < max)
   ✓ Input data directory has files
   ✓ DSI Studio is executable
   ✓ Output directory is writable
   ✓ Sufficient disk space available
   
   🚀 All checks passed. Ready to start!
```

---

### Test Type 3: Graceful Degradation
**Question**: Does pipeline continue when non-critical issue occurs?

**Example**:
```
⚠️  Found 1 corrupt file: corrupted.fz
   Using remaining 304 files for optimization
   
   ℹ️  Consider removing: corrupted.fz
```

---

## Files Created

| File | Purpose |
|------|---------|
| `STRESS_TESTING_GUIDE.md` | Complete stress test specifications (8 categories) |
| `QUICK_STRESS_TESTS.md` | Quick runnable test cases with commands |
| This file | Summary and overview |

---

## How to Run Manually

```bash
# Setup
cd /data/local/software/braingraph-pipeline
source braingraph_pipeline/bin/activate

# Category 1: Input Data
echo "Test 1a: Missing directory"
python opticonn.py bayesian --data-dir /nonexistent --output-dir /tmp/test_1a \
  --config configs/sweep_nano.json --n-iterations 1

# Category 2: Configuration  
echo "Test 2a: Invalid config"
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_2a --config /bad/config.json --n-iterations 1

# Category 3: DSI Studio
echo "Test 3a: Missing DSI_STUDIO_PATH"
unset DSI_STUDIO_PATH
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_3a --config configs/sweep_nano.json --n-iterations 1

# Category 4: Output
echo "Test 4a: No write permissions"
mkdir -p /tmp/readonly_out
chmod 444 /tmp/readonly_out
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/readonly_out --config configs/sweep_nano.json --n-iterations 1

# Category 5: Parallel
echo "Test 5a: Zero workers"
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_5a --config configs/sweep_nano.json \
  --n-iterations 1 --max-workers 0

# Category 6: Parameters
echo "Test 6a: Fixed parameters"
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_6a --config configs/sweep_nano.json --n-iterations 1

# Category 7: Iterations
echo "Test 7a: Zero iterations"
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_7a --config configs/sweep_nano.json --n-iterations 0

# Category 8: Format
echo "Test 8a: Bad atlas name"
cat > /tmp/bad_atlas.json << 'EOF'
{
  "atlases": ["InvalidAtlas"],
  "connectivity_values": ["count"],
  "tract_count": 10000
}
EOF
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_8a --config /tmp/bad_atlas.json --n-iterations 1
```

---

## Expected Results

For each test, we expect:

1. **Clear error message** - User understands what went wrong
2. **Actionable suggestion** - User knows how to fix it
3. **No cryptic tracebacks** - Error is user-friendly, not technical
4. **Fast feedback** - Error caught before wasting time/resources
5. **Graceful exit** - Exit code 1, no incomplete files

---

## Improvement Checklist

After running tests, implement these improvements:

- [ ] Input data validation before starting
- [ ] Better error messages (no tracebacks)
- [ ] Validation summary before execution
- [ ] Disk space pre-check
- [ ] Worker crash recovery
- [ ] Graceful Ctrl+C handling with progress save
- [ ] Parameter range validation (min < max)
- [ ] Auto-adjustment of worker count
- [ ] Did-you-mean suggestions for common errors
- [ ] Documentation links in error messages

---

## Next: Your Turn! 🚀

1. **Read** `STRESS_TESTING_GUIDE.md` for detailed test specs
2. **Review** `QUICK_STRESS_TESTS.md` for runnable commands
3. **Run** the tests manually
4. **Document** any failures or unclear error messages
5. **Report** issues found (we can fix them!)

Would you like to start with any specific category? I'd recommend starting with:
1. **Configuration tests** (quick, most likely to fail)
2. **DSI Studio tests** (important for execution)
3. **Error message tests** (affects user experience)

