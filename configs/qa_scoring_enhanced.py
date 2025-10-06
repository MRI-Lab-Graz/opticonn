"""
Enhanced QA Scoring Configuration
==================================

This document defines the enhanced Pure QA scoring methodology that uses
additional network topology measures for more robust parameter selection.

Network Measures Used in QA Scoring:
------------------------------------

PRIMARY TOPOLOGY METRICS (High Weight):
- small_worldness: Brain networks should exhibit small-world properties (SW ≈ 1.0+)
- global_efficiency: Measure of network integration (higher = better connected)
- clustering_coefficient: Local connectivity/segregation
- density/sparsity: Network connection density (0.05-0.40 typical range)

SECONDARY TOPOLOGY METRICS (Medium Weight):
- assortativity: Degree correlation (-1 to 1, brain networks typically disassortative)
- transitivity: Alternative clustering measure
- modularity: Community structure (if computed)

TERTIARY METRICS (Low Weight - Quality Indicators):
- rich_club: Hub connectivity patterns
- diameter/radius: Network compactness metrics

QA Scoring Principles:
---------------------
1. METRIC-AGNOSTIC: Don't favor count vs fa vs qa - judge topology quality
2. STUDY-INDEPENDENT: No assumptions about group differences
3. TOPOLOGY-FOCUSED: Evaluate biologically plausible network properties
4. REPRODUCIBILITY-FIRST: Penalize unreliable or extreme values

Penalty System:
--------------
- Poor sparsity (< 0.05 or > 0.40): 50% penalty
- Poor small-worldness (< 1.0): 30% penalty  
- Poor assortativity (> 0.2 for brain networks): 20% penalty
- Extreme outlier scores (> 0.98): 10% penalty
- Low reliability (< 0.60): 20% penalty

Enhanced Scoring Formula:
------------------------
base_score = weighted_average(topology_metrics)
qa_score = base_score × penalty_factors
qa_score = clip(qa_score, 0.0, 1.0)

"""

# Enhanced QA Configuration
QA_CONFIG = {
    "version": "2.0",
    "description": "Enhanced Pure QA with expanded network topology metrics",
    
    "principles": {
        "connectivity_metric_agnostic": True,
        "study_design_independent": True,
        "network_topology_focused": True,
        "reproducibility_prioritized": True
    },
    
    "weights": {
        # Primary topology metrics (total: 0.6)
        "small_worldness": 0.20,
        "global_efficiency": 0.15,
        "clustering_coefficient": 0.15,
        "sparsity": 0.10,
        
        # Secondary metrics (total: 0.3)
        "transitivity": 0.10,
        "assortativity_coefficient": 0.10,
        "modularity": 0.10,
        
        # Tertiary metrics (total: 0.1)
        "rich_club_k10": 0.05,
        "network_characteristic_path_length": 0.05
    },
    
    "thresholds": {
        # Sparsity/density bounds
        "min_sparsity": 0.05,
        "max_sparsity": 0.40,
        
        # Small-worldness (brain networks should be > 1.0)
        "min_small_worldness": 1.0,
        "target_small_worldness": 1.5,
        
        # Assortativity (brain networks are typically disassortative)
        "min_assortativity": -0.5,
        "max_assortativity": 0.2,
        
        # Global efficiency (well-connected networks)
        "min_global_efficiency": 0.3,
        
        # Clustering (balanced segregation)
        "min_clustering": 0.2,
        "max_clustering": 0.9,
        
        # Reliability threshold
        "min_reliability": 0.60,
        
        # Outlier detection
        "exclude_extreme_outliers": True,
        "extreme_threshold": 0.98
    },
    
    "penalties": {
        "poor_sparsity": 0.5,          # 50% penalty
        "poor_small_world": 0.7,        # 30% penalty
        "poor_assortativity": 0.8,      # 20% penalty
        "poor_efficiency": 0.8,         # 20% penalty
        "poor_clustering": 0.9,         # 10% penalty
        "extreme_score_artifact": 0.9,  # 10% penalty
        "poor_reliability": 0.8         # 20% penalty
    }
}
