# Manual Testing Checklist

Use this checklist as you manually run each test to track results.

---

## Pre-Test Setup

- [ ] Terminal opened and shell is bash
- [ ] Working directory: `/data/local/software/braingraph-pipeline`
- [ ] Virtual environment activated: `source braingraph_pipeline/bin/activate`
- [ ] DSI_STUDIO_PATH set: `echo $DSI_STUDIO_PATH`
- [ ] Test data available: `ls /data/local/Poly/derivatives/meta/fz/ | wc -l`
- [ ] Test directories created: `mkdir -p /tmp/test_output /tmp/test_configs`

**Setup Status**: ⚪ Ready / ⚪ In Progress / ⚪ Complete

---

## Category 1: Input Data Validation

### Test 1.1: Non-existent Directory
```bash
python opticonn.py bayesian --data-dir /nonexistent/path \
  --output-dir /tmp/test_output/1.1 --config configs/sweep_nano.json --n-iterations 1
```

**Expected**: Error message about directory not found
- [ ] Exit code: 1 ✅
- [ ] Error message clear? ✅
- [ ] Suggests valid path? ✅
- [ ] No traceback? ✅

**Result**: ⚪ Pass / ⚪ Fail / ⚪ Partial

**Notes**: _______________________________________________________________

---

### Test 1.2: Empty Directory
```bash
mkdir -p /tmp/empty_test
python opticonn.py bayesian --data-dir /tmp/empty_test \
  --output-dir /tmp/test_output/1.2 --config configs/sweep_nano.json --n-iterations 1
```

**Expected**: Error about no .fz files found
- [ ] Exit code: 1 ✅
- [ ] Error message clear? ✅
- [ ] Suggests file format? ✅
- [ ] No traceback? ✅

**Result**: ⚪ Pass / ⚪ Fail / ⚪ Partial

**Notes**: _______________________________________________________________

---

### Test 1.3: Wrong File Types
```bash
mkdir -p /tmp/wrong_files && touch /tmp/wrong_files/test.txt
python opticonn.py bayesian --data-dir /tmp/wrong_files \
  --output-dir /tmp/test_output/1.3 --config configs/sweep_nano.json --n-iterations 1
```

**Expected**: Error about no valid fiber files
- [ ] Exit code: 1 ✅
- [ ] Error message clear? ✅
- [ ] Explains valid formats? ✅
- [ ] No traceback? ✅

**Result**: ⚪ Pass / ⚪ Fail / ⚪ Partial

**Notes**: _______________________________________________________________

---

## Category 2: Configuration Validation

### Test 2.1: Invalid JSON
```bash
cat > /tmp/test_configs/bad.json << 'EOF'
{
  "atlases": ["FreeSurferDKT_Cortical",
  "connectivity_values": ["count"]
}
EOF

python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/2.1 --config /tmp/test_configs/bad.json --n-iterations 1
```

**Expected**: JSON parsing error with line number
- [ ] Exit code: 1 ✅
- [ ] Points to error line? ✅
- [ ] Suggests valid JSON? ✅
- [ ] No traceback? ✅

**Result**: ⚪ Pass / ⚪ Fail / ⚪ Partial

**Notes**: _______________________________________________________________

---

### Test 2.2: Missing Required Fields
```bash
cat > /tmp/test_configs/incomplete.json << 'EOF'
{
  "connectivity_values": ["count"],
  "tract_count": 10000
}
EOF

python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/2.2 --config /tmp/test_configs/incomplete.json --n-iterations 1
```

**Expected**: List of missing fields
- [ ] Exit code: 1 ✅
- [ ] Lists missing field: "atlases"? ✅
- [ ] Shows example field value? ✅
- [ ] No traceback? ✅

**Result**: ⚪ Pass / ⚪ Fail / ⚪ Partial

**Notes**: _______________________________________________________________

---

### Test 2.3: Invalid Atlas Name
```bash
cat > /tmp/test_configs/bad_atlas.json << 'EOF'
{
  "atlases": ["InvalidAtlas"],
  "connectivity_values": ["count"],
  "tract_count": 10000
}
EOF

python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/2.3 --config /tmp/test_configs/bad_atlas.json --n-iterations 1
```

**Expected**: Invalid atlas error with suggestions
- [ ] Exit code: 1 ✅
- [ ] Lists valid atlases? ✅
- [ ] Did-you-mean suggestion? ✅
- [ ] No traceback? ✅

**Result**: ⚪ Pass / ⚪ Fail / ⚪ Partial

**Notes**: _______________________________________________________________

---

### Test 2.4: Invalid Connectivity Metric
```bash
cat > /tmp/test_configs/bad_metric.json << 'EOF'
{
  "atlases": ["FreeSurferDKT_Cortical"],
  "connectivity_values": ["invalid_metric"],
  "tract_count": 10000
}
EOF

python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/2.4 --config /tmp/test_configs/bad_metric.json --n-iterations 1
```

**Expected**: Invalid metric error with valid options
- [ ] Exit code: 1 ✅
- [ ] Lists valid metrics? ✅
- [ ] Clear error message? ✅
- [ ] No traceback? ✅

**Result**: ⚪ Pass / ⚪ Fail / ⚪ Partial

**Notes**: _______________________________________________________________

---

## Category 3: Parameter Validation

### Test 3.1: Inverted Range (min > max)
```bash
cat > /tmp/test_configs/inverted.json << 'EOF'
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
  --output-dir /tmp/test_output/3.1 --config /tmp/test_configs/inverted.json --n-iterations 1
```

**Expected**: Error about invalid range
- [ ] Exit code: 1 ✅
- [ ] Points to parameter: fa_threshold? ✅
- [ ] Explains min must be < max? ✅
- [ ] No traceback? ✅

**Result**: ⚪ Pass / ⚪ Fail / ⚪ Partial

**Notes**: _______________________________________________________________

---

### Test 3.2: All Fixed Parameters
```bash
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
  --output-dir /tmp/test_output/3.2 --config /tmp/test_configs/all_fixed.json --n-iterations 1
```

**Expected**: Error about no parameters to optimize
- [ ] Exit code: 1 ✅
- [ ] Clear message about no ranges? ✅
- [ ] Suggests adding parameter ranges? ✅
- [ ] No traceback? ✅

**Result**: ⚪ Pass / ⚪ Fail / ⚪ Partial

**Notes**: _______________________________________________________________

---

## Category 4: DSI Studio Path

### Test 4.1: Missing DSI_STUDIO_PATH
```bash
unset DSI_STUDIO_PATH
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/4.1 --config configs/sweep_nano.json --n-iterations 1
```

**Expected**: Error about missing DSI_STUDIO_PATH
- [ ] Exit code: 1 ✅
- [ ] Clear message? ✅
- [ ] Instructions to fix? ✅
- [ ] No traceback? ✅

**After test, restore**: `export DSI_STUDIO_PATH=/data/local/software/dsi-studio/dsi_studio`

**Result**: ⚪ Pass / ⚪ Fail / ⚪ Partial

**Notes**: _______________________________________________________________

---

### Test 4.2: Wrong Executable
```bash
export DSI_STUDIO_PATH=/bin/ls
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/4.2 --config configs/sweep_nano.json --n-iterations 1

export DSI_STUDIO_PATH=/data/local/software/dsi-studio/dsi_studio
```

**Expected**: Error about DSI Studio test failed
- [ ] Exit code: 1 ✅
- [ ] Catches wrong executable? ✅
- [ ] Suggests validation before running? ✅
- [ ] No traceback? ✅

**Result**: ⚪ Pass / ⚪ Fail / ⚪ Partial

**Notes**: _______________________________________________________________

---

### Test 4.3: Not Executable
```bash
cp /data/local/software/dsi-studio/dsi_studio /tmp/dsi_noexec
chmod -x /tmp/dsi_noexec
export DSI_STUDIO_PATH=/tmp/dsi_noexec
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/4.3 --config configs/sweep_nano.json --n-iterations 1

export DSI_STUDIO_PATH=/data/local/software/dsi-studio/dsi_studio
```

**Expected**: Error about permissions
- [ ] Exit code: 1 ✅
- [ ] Points to permission issue? ✅
- [ ] Suggests chmod +x? ✅
- [ ] No traceback? ✅

**Result**: ⚪ Pass / ⚪ Fail / ⚪ Partial

**Notes**: _______________________________________________________________

---

## Category 5: Iteration Count Validation

### Test 5.1: Zero Iterations
```bash
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/5.1 --config configs/sweep_nano.json --n-iterations 0
```

**Expected**: Error about iterations must be >= 1
- [ ] Exit code: 1 ✅
- [ ] Clear validation message? ✅
- [ ] Shows valid range? ✅
- [ ] No traceback? ✅

**Result**: ⚪ Pass / ⚪ Fail / ⚪ Partial

**Notes**: _______________________________________________________________

---

### Test 5.2: Negative Iterations
```bash
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/5.2 --config configs/sweep_nano.json --n-iterations -5
```

**Expected**: Error about iterations must be >= 1
- [ ] Exit code: 1 ✅
- [ ] Clear validation message? ✅
- [ ] No traceback? ✅

**Result**: ⚪ Pass / ⚪ Fail / ⚪ Partial

**Notes**: _______________________________________________________________

---

## Category 6: Worker Count Validation

### Test 6.1: Zero Workers
```bash
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/6.1 --config configs/sweep_nano.json \
  --n-iterations 1 --max-workers 0
```

**Expected**: Error about workers must be >= 1
- [ ] Exit code: 1 ✅
- [ ] Clear validation message? ✅
- [ ] No traceback? ✅

**Result**: ⚪ Pass / ⚪ Fail / ⚪ Partial

**Notes**: _______________________________________________________________

---

### Test 6.2: Negative Workers
```bash
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/6.2 --config configs/sweep_nano.json \
  --n-iterations 1 --max-workers -5
```

**Expected**: Error about workers must be >= 1
- [ ] Exit code: 1 ✅
- [ ] Clear validation message? ✅
- [ ] No traceback? ✅

**Result**: ⚪ Pass / ⚪ Fail / ⚪ Partial

**Notes**: _______________________________________________________________

---

## Category 7: Output Directory

### Test 7.1: Read-Only Directory
```bash
mkdir -p /tmp/readonly_out && chmod 444 /tmp/readonly_out
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/readonly_out --config configs/sweep_nano.json --n-iterations 1

chmod 755 /tmp/readonly_out 2>/dev/null
```

**Expected**: Error about write permissions
- [ ] Exit code: 1 ✅
- [ ] Clear permission error? ✅
- [ ] Suggests solution? ✅
- [ ] No traceback? ✅

**Result**: ⚪ Pass / ⚪ Fail / ⚪ Partial

**Notes**: _______________________________________________________________

---

## Category 8: Success Test (Smoke Test)

### Test 8.1: Valid Run (2 iterations)
```bash
python opticonn.py bayesian --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output-dir /tmp/test_output/8.1 --config configs/sweep_nano.json \
  --n-iterations 2 --max-workers 2 --verbose 2>&1 | tee /tmp/test_output/8.1.log
```

**Expected**: Successful optimization run
- [ ] Exit code: 0 ✅
- [ ] 2 iterations completed? ✅
- [ ] Results file created? ✅
- [ ] Best atlas shown in output? ✅
- [ ] Computation time shown? ✅
- [ ] No errors? ✅

**Result**: ⚪ Pass / ⚪ Fail / ⚪ Partial

**Notes**: _______________________________________________________________

---

## Summary

### Total Tests: 18
- [ ] Tests Passed: ___ / 18
- [ ] Tests Failed: ___ / 18
- [ ] Partial Pass: ___ / 18

### By Category
- [ ] Category 1 (Input Data): 3/3 ✅
- [ ] Category 2 (Configuration): 4/4 ✅
- [ ] Category 3 (Parameters): 2/2 ✅
- [ ] Category 4 (DSI Studio): 3/3 ✅
- [ ] Category 5 (Iterations): 2/2 ✅
- [ ] Category 6 (Workers): 2/2 ✅
- [ ] Category 7 (Output): 1/1 ✅
- [ ] Category 8 (Success): 1/1 ✅

### Critical Findings

**Most Important Issues**:
1. ___________________________________________________________________
2. ___________________________________________________________________
3. ___________________________________________________________________

**Quick Wins** (Easy Fixes):
1. ___________________________________________________________________
2. ___________________________________________________________________

**Long-Term Improvements**:
1. ___________________________________________________________________
2. ___________________________________________________________________

---

## Recommendations

✅ **READY FOR RELEASE** if:
- All critical category tests pass
- Error messages are clear and actionable
- No cryptic tracebacks
- Graceful exits with exit code 1

⚠️  **NEEDS WORK** if:
- Any critical test fails
- Error messages are unclear
- Tracebacks appear in user output
- Abnormal exit codes

🔴 **BLOCKERS** if:
- Pipeline crashes unexpectedly
- Corrupted output files
- Lost progress on failure
- Security issues

---

**Test Date**: _____________________
**Tester Name**: _____________________
**System**: _____________________
**DSI Studio Version**: _____________________
