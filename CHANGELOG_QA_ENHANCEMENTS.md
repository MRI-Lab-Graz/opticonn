# OptiConn Quality Assurance Enhancements

## Version 2.0 - October 6, 2025

### Critical Bug Fixes

#### 1. **Fixed tract_count Parameter Bug** 🐛
**Problem**: All sweep combinations were using 100k tracks regardless of configuration
- Sweep with 5M tract_count was actually running with 100k
- Resulted in identical network measures across different tract_count settings

**Root Cause**: Typo in `extract_connectivity_matrices.py`
- Code used `self.config["track_count"]` 
- But config files use `"tract_count"` (correct DSI Studio parameter name)

**Solution**:
```python
# Now handles both spellings for backward compatibility
tract_count = self.config.get('tract_count', self.config.get('track_count', 100000))
```

**Files Changed**:
- `scripts/extract_connectivity_matrices.py` (lines 374, 467, 525)

**Impact**: 🔴 **CRITICAL** - Previous sweeps may have invalid results if tract_count was varied

---

### User Experience Improvements

#### 2. **Reduced Verbose Output** 🔇
**Problem**: Every sweep combination logged detailed metrics, cluttering output

**Solution**:
- Per-combination logs only show with `--verbose` flag
- Added concise summary: "✅ Completed X/Y parameter combinations successfully"
- Removed unhelpful normalized quality score (always 1.0) from default output

**Usage**:
```bash
# Quiet mode (new default)
opticonn sweep --config sweep.json --data-dir data --output-dir out

# Verbose mode (shows all details)
opticonn sweep --config sweep.json --data-dir data --output-dir out --verbose
```

**Files Changed**:
- `scripts/cross_validation_bootstrap_optimizer.py`

---

### Enhanced QA Scoring

#### 3. **Expanded Network Topology Metrics** 📊

**New Metrics in QA Scoring**:

| Metric | Weight | Purpose | Threshold |
|--------|--------|---------|-----------|
| Small-worldness | 0.20 | Primary topology indicator | ≥ 1.0 |
| Global efficiency | 0.15 | Network integration | ≥ 0.3 |
| Clustering coefficient | 0.10 | Local connectivity | 0.2 - 0.9 |
| Assortativity | 0.10 | Degree correlation | ≤ 0.2 (disassortative) |
| Sparsity | 0.20 | Network density | 0.05 - 0.40 |
| Modularity | 0.15 | Community structure | - |
| Reliability | 0.10 | Cross-subject consistency | ≥ 0.60 |

**Enhanced Penalty System**:
- Poor sparsity: 50% penalty (was same)
- Poor small-worldness: 30% penalty (was same)
- **NEW**: Poor assortativity (>0.2): 20% penalty
- **NEW**: Poor efficiency (<0.3): 20% penalty
- **NEW**: Poor clustering (extreme values): 10% penalty

**Files Changed**:
- `scripts/optimal_selection.py` (enhanced `calculate_pure_qa_score`)
- `configs/qa_scoring_enhanced.py` (new documentation)

---

### New Configuration Files

#### 4. **Production-Ready Configs** 📝

**A. Full Atlas Coverage** (`configs/sweep_production_full.json`)
```json
{
  "atlases": [
    "FreeSurferDKT_Cortical",
    "FreeSurferDKT_Subcortical",
    "ATAG_basal_ganglia",
    "BrainSeg",
    "Brainnectome",
    "Brodmann",
    "Campbell",
    "Cerebellum-SUIT",
    "CerebrA",
    "HCP-MMP",
    "HCP842_tractography",
    "HCPex",
    "JulichBrain",
    "Kleist",
    "THOMAS"
  ],
  "connectivity_values": ["count", "fa", "qa", "nqa"],
  "sampling": {
    "method": "lhs",
    "n_samples": 50
  }
}
```

**B. Ultra-Fast Validation** (`configs/sweep_nano.json`)
- Only 4 combinations (2 fa_threshold × 2 tract_count)
- tract_count: 10k, 50k (very fast)
- Perfect for smoke testing

---

### Available Network Measures from DSI Studio

OptiConn now has access to 30 network measures per subject:

**Binary Graph Metrics:**
- density
- clustering_coeff_average
- transitivity
- network_characteristic_path_length
- small-worldness
- global_efficiency
- diameter_of_graph
- radius_of_graph
- assortativity_coefficient
- rich_club_k5, k10, k15, k20, k25

**Weighted Graph Metrics:**
- (Same as above, with weighted calculations)

**Node-Level Metrics** (available but not yet aggregated):
- degree
- strength
- cluster_coef
- local_efficiency
- betweenness_centrality
- eigenvector_centrality
- pagerank_centrality
- eccentricity

---

### Migration Guide

#### For Existing Sweeps

1. **Verify tract_count was used correctly**:
```bash
# Check if your sweep configs had tract_count variations
grep tract_count_range your_sweep_config.json

# If yes, results may be invalid - re-run with fixed version
```

2. **Re-run critical sweeps**:
```bash
# Use same config but will now use correct tract_count
opticonn sweep --config your_sweep.json --data-dir data --output-dir new_sweep
```

3. **Verbose output**:
```bash
# Old behavior (verbose by default) - add --verbose flag
opticonn sweep --config sweep.json --data-dir data --output-dir out --verbose
```

#### For New Projects

**Quick Validation** (2-5 minutes):
```bash
opticonn sweep --config configs/sweep_nano.json \
  --data-dir data/subjects \
  --output-dir validation_sweep \
  --subjects 1
```

**Micro Sweep** (10-30 minutes):
```bash
opticonn sweep --config configs/sweep_micro.json \
  --data-dir data/subjects \
  --output-dir micro_sweep \
  --subjects 3
```

**Full Production** (hours to days):
```bash
opticonn sweep --config configs/sweep_production_full.json \
  --data-dir data/subjects \
  --output-dir production_sweep \
  --subjects 10
```

---

### Breaking Changes

None - all changes are backward compatible:
- `tract_count` now preferred, but `track_count` still works
- Verbose output requires explicit `--verbose` flag (cleaner default)
- Enhanced QA scoring uses same methodology, just more metrics

---

### Testing Recommendations

1. **Validate tract_count fix**:
```bash
# Run small sweep with different tract_counts
# Verify network measures differ between conditions
```

2. **Compare QA scores**:
```bash
# Old QA scores should be similar to new ones
# New scores may be slightly lower due to additional penalties
```

3. **Check for assortativity warnings**:
```bash
# Brain networks are typically disassortative (negative assortativity)
# If you see many "poor_assortativity" penalties, parameters may need tuning
```

---

### Performance Notes

**Tract Count Impact** (approximate, per subject per atlas):
- 10k tracks: ~5 seconds
- 50k tracks: ~15 seconds  
- 100k tracks: ~30 seconds
- 500k tracks: ~2 minutes
- 1M tracks: ~4 minutes
- 5M tracks: ~20 minutes

**Recommended for sweeps**: Start with 100k for validation, use 1M+ for production

---

### Known Limitations

1. **Node-level metrics not yet used**: betweenness, eigenvector, pagerank centrality available but not aggregated
2. **Modularity not always computed**: Depends on network structure
3. **Rich club metrics**: Only k5, k10, k15, k20, k25 available (hardcoded in DSI Studio)

---

### Future Enhancements

- [ ] Add node-level centrality aggregation
- [ ] Compute modularity separately if not provided by DSI Studio
- [ ] Add reproducibility metrics across bootstrap samples
- [ ] Implement edge-weight distribution QA checks
- [ ] Add hemisphere asymmetry checks for validation

---

### Contributors

- QA Enhancement: Karl (Oct 6, 2025)
- tract_count bugfix: Karl (Oct 6, 2025)
- Verbose output reduction: Karl (Oct 6, 2025)
