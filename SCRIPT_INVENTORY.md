# Script Inventory and Usage Analysis

## Active Pipeline Scripts (Core Functionality)

These scripts are essential for the sweep/apply workflow and are actively used:

### Primary Entry Points
- **`opticonn.py`** ✅ Active
  - Main CLI launcher
  - Activates virtual environment
  - Routes to `opticonn_hub.py`

### Hub and Dispatchers
- **`scripts/opticonn_hub.py`** ✅ Active
  - Central command dispatcher
  - Handles: sweep, apply, review, validate commands
  - Coordinates workflow execution

### Optimization Core
- **`scripts/cross_validation_bootstrap_optimizer.py`** ✅ Active
  - Core sweep implementation
  - Two-wave bootstrap validation
  - Called by opticonn_hub for sweep command

- **`scripts/extract_connectivity_matrices.py`** ✅ Active
  - DSI Studio integration
  - Tractography execution
  - Matrix extraction
  - Called by cross_validation_bootstrap_optimizer

- **`scripts/optimal_selection.py`** ✅ Active
  - Parameter selection logic
  - Quality-based ranking
  - Dataset preparation
  - Called by cross_validation_bootstrap_optimizer as subprocess

### Results Processing
- **`scripts/aggregate_network_measures.py`** ✅ Active
  - Aggregates connectivity metrics
  - Generates statistical summaries
  - Called by cross_validation_bootstrap_optimizer

- **`scripts/quick_quality_check.py`** ✅ Active
  - Fast QA assessment
  - Issue detection
  - Called by opticonn_hub after sweep

- **`scripts/pareto_view.py`** ✅ Active
  - Pareto front analysis
  - Multi-objective visualization
  - Called by opticonn_hub for post-sweep analysis

### Subcommands
- **`scripts/subcommands/find_optimal_parameters.py`** ✅ Active
  - Alternative entry for optimization
  - Supports both Bayesian and sweep methods
  - Called by opticonn_hub

- **`scripts/subcommands/apply.py`** ✅ Active
  - Applies parameters to full dataset
  - Batch processing
  - Called by opticonn_hub for apply command

- **`scripts/subcommands/review.py`** ✅ Active
  - Results display and summary
  - Called by opticonn_hub for review command

### Utilities
- **`scripts/utils/runtime.py`** ✅ Active
  - Platform utilities
  - Environment setup
  - Path handling

- **`scripts/utils/validate_setup.py`** ✅ Active
  - Environment validation
  - Configuration checking
  - Dependency verification

- **`scripts/json_validator.py`** ✅ Active
  - Config validation
  - Error suggestions
- **`scripts/sweep_utils.py`** ✅ Active
  - Parameter grid generation
  - Sampling strategies (grid, random, LHS)

---

## Potentially Redundant Scripts

These scripts exist but are not actively used in the main sweep/apply workflow:

### 1. **`scripts/bayesian_optimizer.py`** ⚠️ POTENTIALLY UNUSED
- **Purpose**: Standalone Bayesian optimization
- **Status**: Code exists but may be superseded by find_optimal_parameters.py
- **Used By**: Not called in main workflows
- **Recommendation**: 
  - Check if functionality is replicated in find_optimal_parameters.py
  - If yes, consider consolidation or deprecation
  - If no, document as alternative method

### 2. **`scripts/run_parameter_sweep.py`** ⚠️ POTENTIALLY UNUSED
- **Purpose**: Parameter sweep execution
- **Status**: Functionality likely covered by cross_validation_bootstrap_optimizer.py
- **Used By**: Not found in current workflow
- **Recommendation**: 
  - Verify if this is legacy code
  - Consider archiving if replaced by cross_validation_bootstrap_optimizer

### 3. **`scripts/run_pipeline.py`** ⚠️ POTENTIALLY UNUSED
- **Purpose**: General pipeline runner
- **Status**: Replaced by modular command structure
- **Used By**: Not actively called
- **Recommendation**: 
  - Check if end-to-end pipeline execution is needed
  - Consider if replacement by separate sweep/apply is sufficient
  - May be legacy from earlier design

### 4. **`scripts/metric_optimizer.py`** ⚠️ POTENTIALLY UNUSED
- **Purpose**: Metric-specific optimization
- **Status**: Unclear if used
- **Used By**: Not found in sweep workflow
- **Recommendation**: 
  - Clarify purpose and relationship to main optimization
  - Check if metrics are handled by aggregate_network_measures
  - Consider merging into core workflow if needed

### 5. **`scripts/statistical_analysis.py`** ⚠️ POTENTIALLY UNUSED
- **Purpose**: Statistical analysis tools
- **Status**: May be utility functions not integrated
- **Used By**: Not actively called
- **Recommendation**: 
  - Extract functions used by other scripts
  - Archive remaining statistical functions
  - Consider as separate analysis tool

### 6. **`scripts/statistical_metric_comparator.py`** ⚠️ POTENTIALLY UNUSED
- **Purpose**: Compare metrics across conditions
- **Status**: Analysis tool, not pipeline operation
- **Used By**: Manual analysis only
- **Recommendation**: 
  - Keep for post-hoc analysis
  - Document as optional analysis tool
  - Consider moving to analysis/ subdirectory

### 7. **`scripts/qa_cross_validator.py`** ⚠️ POTENTIALLY UNUSED
- **Purpose**: Cross-validated quality assessment
- **Status**: Overlaps with bootstrap_qa_validator.py
- **Used By**: Not found in active workflow
- **Recommendation**: 
  - Compare with bootstrap_qa_validator
  - Consolidate if overlapping functionality
  - Document differences if both are needed

### 8. **`scripts/aggregate_wave_candidates.py`** ⚠️ POTENTIALLY UNUSED
- **Purpose**: Aggregates Wave candidates
- **Status**: Unclear if used or superseded
- **Used By**: Not actively called
- **Recommendation**: 
  - Check if functionality is in aggregate_network_measures
  - Verify if needed for Wave processing
  - Consider consolidation

### 9. **`scripts/sensitivity_analyzer.py`** ⚠️ POTENTIALLY UNUSED
- **Purpose**: Parameter sensitivity analysis
- **Status**: Advanced analysis feature
- **Used By**: Not in main workflow
- **Recommendation**: 
  - Useful for advanced users
  - Document as optional analysis
  - Consider adding to main workflow if valuable

### 10. **`scripts/verify_parameter_uniqueness.py`** ⚠️ POTENTIALLY UNUSED
- **Purpose**: Validates parameter uniqueness
- **Status**: Utility function
- **Used By**: Not called in workflow
- **Recommendation**: 
  - Move to utils/ if needed
  - Archive if not needed

### 11. **`scripts/pre_test_validation.py`** ⚠️ POTENTIALLY UNUSED
- **Purpose**: Pre-execution validation
- **Status**: Similar to validate_setup.py
- **Used By**: Not actively used
- **Recommendation**: 
  - Compare with validate_setup
  - Consolidate if duplicate
  - Keep one, archive other

### 12. **`scripts/check_script_compliance.py`** ⚠️ POTENTIALLY UNUSED
- **Purpose**: Code compliance checker
- **Status**: Development/maintenance tool
- **Used By**: Not part of pipeline
- **Recommendation**: 
  - Archive as development tool
  - Not needed for end users
  - Keep in repo for maintainers

---

## Test Scripts

These are for development/testing only:

- **`scripts/test_integrity_checks.py`** 📋 Development
- **`scripts/test/test_braingraph_features.py`** 📋 Development
- **`scripts/test/test_enhanced_conversion.py`** 📋 Development

**Recommendation**: Keep in separate `test/` directory, not included in distribution

---

## Duplicate Files

### Duplicate Utilities Detected

These utilities appear in two locations:

1. **`scripts/json_validator.py`** vs **`scripts/utils/json_validator.py`**
   - **Status**: Potential duplicate
   - **Recommendation**: Keep one, remove other
   - **Suggestion**: Keep in `scripts/utils/json_validator.py`

2. **`scripts/quick_quality_check.py`** vs **`scripts/utils/quick_quality_check.py`**
   - **Status**: Potential duplicate
   - **Recommendation**: Keep one, remove other
   - **Suggestion**: Keep in `scripts/utils/quick_quality_check.py`

3. **`scripts/validate_setup.py`** vs **`scripts/utils/validate_setup.py`**
   - **Status**: Potential duplicate
   - **Recommendation**: Keep one, remove other
   - **Suggestion**: Keep in `scripts/utils/validate_setup.py`

---

## Consolidation Recommendations

### Phase 1: Immediate Cleanup (No Risk)
1. **Remove duplicate utility files**
   - Keep only the version in `scripts/utils/`
   - Update imports in other files
   - Files to remove:
     - `scripts/json_validator.py`
     - `scripts/quick_quality_check.py`
     - `scripts/validate_setup.py`

2. **Archive development tools**
   - Move to `scripts/archive/` or `.deprecated/`
   - Files:
     - `scripts/check_script_compliance.py`
     - Test files (keep in `scripts/test/` but mark as optional)

### Phase 2: Analysis & Consolidation (Review First)
1. **Compare potentially redundant scripts**
   - `bayesian_optimizer.py` vs `find_optimal_parameters.py`
   - `qa_cross_validator.py` vs `bootstrap_qa_validator.py`
   - `run_parameter_sweep.py` vs `cross_validation_bootstrap_optimizer.py`
   - `pre_test_validation.py` vs `utils/validate_setup.py`

2. **Decision for each**
   - If duplicate: Keep one, archive other
   - If complementary: Document both
   - If specialized: Mark as optional/advanced feature

### Phase 3: Documentation
1. Update README with only active scripts
2. Document optional/advanced scripts separately
3. Provide migration guide if removing scripts

---

## Usage Count by Script Category

### Active Scripts (in main workflow)
- Entry points: 1
- Hub/Dispatcher: 1
- Optimization core: 3
- Results processing: 3
- Subcommands: 3
- Utilities: 4
- **Total: 15 scripts** (essential for functionality)

### Potentially Redundant Scripts
- **Count: 12 scripts** (may be inactive or redundant)

### Development/Test Scripts
- **Count: 3 scripts** (for testing, not user-facing)

### Duplicate Utilities
- **Count: 3 files** (exact or near duplicates)

---

## Recommendations Summary

| Action | Scripts | Effort | Risk |
|--------|---------|--------|------|
| Remove duplicate utils | 3 | Low | Very Low |
| Archive dev tools | 1 | Low | None |
| Consolidate redundant | 4-6 | Medium | Medium |
| Keep as optional | 3-4 | Low | Low |
| Archive legacy | 2-3 | Low | Medium |

**Estimated cleanup: 1-2 days of work with testing**

---

## Next Steps

1. **Verify usage** of potentially redundant scripts
2. **Test consolidation** in development environment
3. **Update documentation** to reflect changes
4. **Archive superseded scripts** with explanation
5. **Update imports** in remaining files
6. **Release new version** with cleaner codebase

---

## Questions for Code Review

Before implementing consolidation:

1. Is `bayesian_optimizer.py` used or tested anywhere?
2. Are there users depending on `run_parameter_sweep.py` directly?
3. What is the relationship between different "optimizer" scripts?
4. Should dual utilities in utils/ and root exist for backward compatibility?
5. Are there missing features in the new structure vs. old scripts?
