# OptiConn 3-Step Workflow

OptiConn provides a clean, user-friendly workflow for optimizing tractography parameters and analyzing brain connectivity.

## Overview

```
Step 1: SWEEP     → Compute metrics for parameter combinations
Step 2: REVIEW    → Interactively select best candidate  
Step 3: APPLY     → Run full analysis with selected parameters
```

---

## Step 1: Sweep - Parameter Exploration

**Purpose:** Test different parameter combinations on a small subset of data to find optimal settings.

```bash
opticonn sweep -i data/fib_samples -o studies/my_study --quick
```

**What it does:**
- Randomly selects a small subset of subjects (default: 3)
- Runs two validation waves with different parameter combinations
- Computes connectivity matrices and network metrics
- Generates Pareto plots showing trade-offs between objectives
- Saves all results without auto-selecting "best" parameters

**Output:**
```
studies/my_study/
└── sweep-<uuid>/
    └── optimize/
        ├── bootstrap_qa_wave_1/
        │   └── combos/
        │       ├── sweep_0001/
        │       ├── sweep_0002/
        │       └── ...
        ├── bootstrap_qa_wave_2/
        │   └── combos/...
        └── optimization_results/
            ├── pareto_front.csv
            ├── pareto_front.png
            └── all_candidates_ranked.json
```

**Key Options:**
- `--quick`: Use micro sweep config (faster, fewer combinations)
- `--verbose`: Show DSI Studio commands in log
- `--subjects N`: Number of subjects per wave (default: 3)
- `--auto-select`: Skip review step and auto-select top 3 (legacy mode)

---

## Step 2: Review - Interactive Selection

**Purpose:** Review sweep results and select the best parameter combination for your study.

```bash
opticonn review -o studies/my_study/sweep-<uuid>/optimize
```

**What it does:**
- Launches interactive Dash web app (http://localhost:8050)
- Displays Pareto front visualization
- Shows all candidates in sortable table with cross-wave metrics
- **Cross-Wave Analysis:**
  - Shows which candidates appear in multiple waves (consistency)
  - Calculates average QA scores across waves
  - Highlights candidates that are robust across validation
- Allows you to:
  - Compare different parameter combinations
  - Sort by QA score, wave consistency, or any metric
  - Filter candidates
  - Visualize trade-offs (e.g., density vs. efficiency)
  - Select optimal candidate based on your priorities
- Saves selection to `selected_candidate.json`

**How to use:**
1. Open browser to http://localhost:8050
2. Review the recommendation at the top (best QA + consistency)
3. Explore Pareto plot by selecting different axes
4. Review candidate table:
   - **Sort by `waves_present`** to find candidates validated across all waves
   - **Sort by `avg_qa_across_waves`** to find highest quality
   - **Sort by `qa_consistency`** to find most stable parameters
5. Click on a row to select it
6. Click "Select This Candidate for Apply" button
7. Copy the `opticonn apply` command shown
8. Close the Dash app (Ctrl+C)

**Automated Selection (No GUI):**
```bash
opticonn review -o studies/my_study/sweep-<uuid>/optimize --auto-select-best
```
Automatically selects the candidate with:
- Highest QA score among candidates present in ALL waves, OR
- If no candidate in all waves: highest overall QA score

**Understanding Wave Alignment:**
- **`waves_present`**: How many validation waves this candidate appeared in
- **`avg_qa_across_waves`**: Average QA score across all waves
- **`qa_consistency`**: How stable the QA score is (1.0 = perfectly consistent)
- **Best choice**: High QA + appears in all waves = robust, validated parameters

**Output:**
```
studies/my_study/sweep-<uuid>/optimize/
└── selected_candidate.json
```

---

## Step 3: Apply - Full Dataset Analysis

**Purpose:** Apply selected parameters to ALL subjects in your dataset.

```bash
opticonn apply -i data/fib_samples \
  --optimal-config studies/my_study/sweep-<uuid>/optimize/selected_candidate.json \
  -o studies/my_study/sweep-<uuid>
```

**What it does:**
- Reads selected parameter configuration
- Runs connectivity extraction on **all subjects** (not just the subset)
- Computes network metrics for entire dataset
- Generates final analysis outputs

**Output:**
```
studies/my_study/sweep-<uuid>/
└── selected/
    ├── extraction_from_selection.json
    ├── 01_connectivity/
    │   ├── <subject1>.gqi_<date>/
    │   ├── <subject2>.gqi_<date>/
    │   └── ...
    ├── 02_optimization/
    └── 03_analysis/
```

**Key Options:**
- `--skip-extraction`: Skip Step 01 if already done
- `--candidate-index N`: If using top3_candidates.json, select rank N (default: 1)
- `--quiet`: Suppress verbose output

---

## Alternative: Legacy Auto-Select Mode

If you want the old behavior (automatic selection without review):

```bash
opticonn sweep -i data/fib_samples -o studies/my_study --quick --auto-select
```

This will:
- Run sweep as normal
- Automatically aggregate and rank candidates
- Generate `top3_candidates.json`
- Allow you to skip straight to `opticonn apply`

Then apply:
```bash
opticonn apply -i data/fib_samples \
  --optimal-config studies/my_study/sweep-<uuid>/optimize/optimization_results/top3_candidates.json \
  -o studies/my_study/sweep-<uuid>
```

---

## Example End-to-End Session

```bash
# 1. Run quick sweep on demo data
opticonn sweep -i data/fib_samples -o demo_study --quick

# 2. Review and select best candidate interactively
opticonn review -o demo_study/sweep-<uuid>/optimize
# (Use web interface to select candidate)

# 3. Apply to full dataset
opticonn apply -i data/fib_samples \
  --optimal-config demo_study/sweep-<uuid>/optimize/selected_candidate.json \
  -o demo_study/sweep-<uuid>
```

---

## Tips

### Choosing Parameters in Review
- **High density + High efficiency**: Good connectivity, well-organized network
- **Low density + High efficiency**: Sparse but organized (may miss connections)
- **High density + Low efficiency**: Over-connected, noisy
- **Low density + Low efficiency**: Poor quality, likely bad parameters

### Quality Flags to Watch For
- `poor_small_world`: Network doesn't show expected small-world properties
- `extreme_score_artifact`: Suspicious metrics (possibly artifacts)
- Check QA penalties in candidate details

### When to Use Quick vs. Full Sweep
- **Quick (`--quick`)**: Testing, demos, initial exploration (~5-10 min)
- **Full sweep**: Production analysis, paper-quality results (hours)

### Reproducibility
- All parameter combinations are saved with metadata
- Each sweep has unique UUID for traceability
- Git commit hash embedded in outputs (when in git repo)

---

## Troubleshooting

**Sweep fails immediately:**
```bash
# Run validation to check setup
python scripts/validate_setup.py --config configs/braingraph_default_config.json --output-dir test_output --test-input data/fib_samples
```

**Review app won't start:**
```bash
# Check if Dash is installed
pip install dash plotly pandas

# Specify different port
opticonn review -o <path> --port 8051
```

**Can't find selected_candidate.json:**
- Make sure you clicked "Select This Candidate" button in review app
- Check the exact path shown in the app's success message
- File will be in `<optimize_dir>/selected_candidate.json`

---

For more help: `opticonn --help` or `opticonn <command> --help`
