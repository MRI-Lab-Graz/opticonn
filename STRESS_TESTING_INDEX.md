# Stress Testing Documentation Index

## Complete Stress Testing Suite for Bayesian Optimizer

This directory contains comprehensive stress testing documentation for the braingraph pipeline Bayesian optimizer. Use these guides to manually test and validate the pipeline.

---

## 📖 Files in This Suite

### 1. **STRESS_TEST_SUMMARY.md** (Start Here!)
- **Purpose**: High-level overview of all stress tests
- **Content**: 8 test categories, 30+ test cases, quick start guide
- **Reading Time**: 10 minutes
- **Best For**: Understanding the big picture

**Key Sections**:
- Overview of 8 categories
- Quick start instructions
- Testing matrix and results tracking
- Improvement checklist

---

### 2. **STRESS_TESTING_GUIDE.md** (Detailed Reference)
- **Purpose**: Complete specifications for each test
- **Content**: Detailed test descriptions, expected behaviors, root causes
- **Reading Time**: 30 minutes
- **Best For**: Understanding the why behind each test

**Key Sections**:
- Category 1-8 with detailed test specs
- Expected behavior for each test
- How to recognize success vs failure
- Common issues and their solutions

---

### 3. **QUICK_STRESS_TESTS.md** (Execute Tests)
- **Purpose**: Copy-paste ready test commands
- **Content**: All tests organized by category with bash commands
- **Reading Time**: 15 minutes
- **Best For**: Actually running the tests

**Key Sections**:
- Quick validation checks
- Tests by category with commands
- Full smoke test
- Automated test runner script (save as `run_stress_tests.sh`)

---

### 4. **MANUAL_TEST_CHECKLIST.md** (Track Progress)
- **Purpose**: Checkbox format for tracking test results
- **Content**: Setup instructions, per-test checklists, summary section
- **Reading Time**: 5 minutes to scan, use throughout testing
- **Best For**: Tracking which tests pass/fail and documenting issues

**Key Sections**:
- Pre-test setup checklist
- Per-test result tracking (pass/fail/partial)
- Notes section for each test
- Summary statistics
- Recommendations based on results

---

## 🚀 Quick Start (Choose Your Path)

### Path A: "Just Show Me the Tests" (30 minutes)
1. Read: **STRESS_TEST_SUMMARY.md** (10 min)
2. Copy: Commands from **QUICK_STRESS_TESTS.md** (5 min)
3. Run: Tests 1-5 from each category (15 min)

### Path B: "Thorough Testing" (60 minutes)
1. Read: **STRESS_TEST_SUMMARY.md** (10 min)
2. Read: **STRESS_TESTING_GUIDE.md** (20 min)
3. Use: **MANUAL_TEST_CHECKLIST.md** (30 min)
4. Run: All 18 core tests with tracking

### Path C: "Automated Testing" (15 minutes)
1. Read: **QUICK_STRESS_TESTS.md** "Automated Test Runner" section
2. Run: `bash run_stress_tests.sh` (from QUICK_STRESS_TESTS.md)
3. Review: Test summary report

---

## 📊 Test Categories at a Glance

| Category | Tests | Focus | Priority |
|----------|-------|-------|----------|
| Input Data | 3 | Missing/empty/wrong files | High |
| Configuration | 4 | JSON parsing, required fields | High |
| Parameters | 2 | Range validation, fixed params | High |
| DSI Studio | 3 | Path resolution, executability | High |
| Iterations | 2 | Count validation | Medium |
| Workers | 2 | Count validation | Medium |
| Output | 1 | Directory permissions | Low |
| Success | 1 | Smoke test | High |

**Total: 18 core tests**

---

## ✨ Key Testing Principles

Each test verifies:

1. **Error Message Clarity**
   - Is the error message clear and understandable?
   - Does it avoid technical jargon?
   - Does it suggest what to do next?

2. **Actionable Suggestions**
   - Can the user easily fix the issue?
   - Are there examples provided?
   - Is there documentation referenced?

3. **No Cryptic Tracebacks**
   - Does the error show code tracebacks to users?
   - Are cryptic Python errors hidden?
   - Is the user-facing message simple?

4. **Fast Error Detection**
   - Are errors caught before wasting time/resources?
   - Is there pre-flight validation?
   - Does the user get fast feedback?

5. **Graceful Exit Handling**
   - Does the program exit cleanly with code 1?
   - Are incomplete files not left behind?
   - Is progress saved if possible?

---

## 🧪 How to Run Tests

### Setup
```bash
cd /data/local/software/braingraph-pipeline
source braingraph_pipeline/bin/activate
mkdir -p /tmp/test_output /tmp/test_configs
```

### Example Test
```bash
# Test: Non-existent directory
python opticonn.py bayesian --data-dir /nonexistent \
  --output-dir /tmp/test_output/test_1 \
  --config configs/sweep_nano.json --n-iterations 1

# Expected: Error about directory not found
# Exit code: 1
# Message should be clear and actionable
```

### Tracking Results
Use **MANUAL_TEST_CHECKLIST.md** to track:
- [ ] Exit code correct?
- [ ] Error message clear?
- [ ] Suggestions provided?
- [ ] No traceback?

---

## 📋 Pre-Testing Checklist

Before running tests:

- [ ] Terminal opened and in bash
- [ ] Working directory: `/data/local/software/braingraph-pipeline`
- [ ] Virtual environment activated: `source braingraph_pipeline/bin/activate`
- [ ] DSI_STUDIO_PATH set: `echo $DSI_STUDIO_PATH` (should show path)
- [ ] Test data available: `ls /data/local/Poly/derivatives/meta/fz/ | head`
- [ ] Test directories ready: `mkdir -p /tmp/test_output /tmp/test_configs`

---

## 🎯 Success Criteria

### Ready for Release ✅
- [ ] All tests pass or fail gracefully
- [ ] Error messages are clear and actionable
- [ ] No cryptic tracebacks in output
- [ ] Program exits with code 1 on errors
- [ ] Smoke test (Category 8) succeeds

### Needs Work ⚠️
- [ ] Some tests fail with unclear errors
- [ ] Error messages could be improved
- [ ] Need more validation checks
- [ ] Documentation gaps

### Blockers 🔴
- [ ] Program crashes unexpectedly
- [ ] Corrupts data on failure
- [ ] Security vulnerabilities
- [ ] Loses progress without saving

---

## 📊 Test Results Summary Template

```
Date: _______________
Tester: _______________
System: _______________

Category 1 (Input Data): ___ / 3 passed
Category 2 (Config): ___ / 4 passed
Category 3 (Parameters): ___ / 2 passed
Category 4 (DSI Studio): ___ / 3 passed
Category 5 (Iterations): ___ / 2 passed
Category 6 (Workers): ___ / 2 passed
Category 7 (Output): ___ / 1 passed
Category 8 (Smoke): ___ / 1 passed

Total: ___ / 18 passed

Critical Issues Found:
1. _______________
2. _______________
3. _______________

Recommendations:
- Ready for release?
- Needs work?
- Any blockers?
```

---

## 🔧 If Tests Fail

### Common Issues

**Issue: "Configuration validation failed"**
- Check: Is JSON valid?
- Check: Are all required fields present?
- Check: Are atlas names valid?
- See: STRESS_TESTING_GUIDE.md Category 2

**Issue: "DSI Studio not found"**
- Check: Is DSI_STUDIO_PATH set? `echo $DSI_STUDIO_PATH`
- Check: Does file exist? `ls $DSI_STUDIO_PATH`
- Check: Is it executable? `$DSI_STUDIO_PATH --version`
- See: STRESS_TESTING_GUIDE.md Category 3

**Issue: "Input directory has no files"**
- Check: Does directory exist? `ls /data/local/Poly/derivatives/meta/fz/`
- Check: Are there .fz files? `ls *.fz`
- Check: Disk space? `df -h`
- See: STRESS_TESTING_GUIDE.md Category 1

---

## 📚 Related Documentation

| Document | Purpose |
|----------|---------|
| README.md | Main pipeline documentation |
| INSTALLATION_GUIDE.md | How to install and set up |
| DSI_STUDIO_PATH_RESOLUTION_FIX.md | DSI Studio configuration details |
| BAYESIAN OPTIMIZATION RESULTS.md | Understanding optimization results |

---

## 💡 Tips for Testing

1. **Start Simple**: Begin with Category 2 (Configuration) - most likely to reveal issues
2. **Test Edge Cases**: Invalid values teach us the most
3. **Document Everything**: Use MANUAL_TEST_CHECKLIST.md to track findings
4. **Note Surprises**: If behavior is unexpected, document it!
5. **Be Methodical**: Test one thing at a time
6. **Check Error Messages**: This is the most important part

---

## 🚀 Next Steps

1. **Read**: STRESS_TEST_SUMMARY.md (10 min)
2. **Review**: STRESS_TESTING_GUIDE.md (20 min)
3. **Plan**: Choose your testing path (A, B, or C)
4. **Execute**: Run tests using QUICK_STRESS_TESTS.md
5. **Track**: Use MANUAL_TEST_CHECKLIST.md
6. **Report**: Document findings
7. **Improve**: We'll implement fixes based on results!

---

## ❓ Questions?

- **What should I test?** → See STRESS_TEST_SUMMARY.md
- **How do I run a test?** → See QUICK_STRESS_TESTS.md
- **What should happen?** → See STRESS_TESTING_GUIDE.md
- **How do I track results?** → Use MANUAL_TEST_CHECKLIST.md

---

## 🎉 Thank You!

Your thorough testing will help make the pipeline bulletproof and user-friendly. Every test you run, every error you catch, and every suggestion you make helps improve the pipeline for everyone!

**Happy Testing!** 🚀

---

**Last Updated**: October 19, 2025
**Version**: 1.0
**Status**: Ready for Testing
