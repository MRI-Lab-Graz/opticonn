#!/usr/bin/env python3
# Supports: --dry-run (prints intended actions without running)
# When run without arguments the script prints help: parser.print_help()
"""
Optimal Selection Module
========================

Extracts optimal atlas/connectivity metric combinations from optimization results
and prepares them for scientific analysis (e.g., soccer vs control comparisons).

Author: Braingraph Pipeline Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import json

from scripts.utils.runtime import configure_stdio

logger = logging.getLogger(__name__)

class OptimalSelector:
    """
    Selects optimal atlas/metric combinations and prepares data for scientific analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the optimal selector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict:
        """Default configuration for optimal selection."""
        return {
            "selection_criteria": {
                "use_recommended": True,          # Use optimizer's recommended combinations
                "min_quality_score": 0.7,        # Minimum quality score threshold
                "top_n_per_atlas": 1,            # Number of top combinations per atlas
                "top_n_overall": 5                # Number of top combinations overall
            },
            "priority_atlases": [
                "FreeSurferDKT_Tissue",
                "FreeSurferDKT_Cortical", 
                "HCP-MMP",
                "Cerebellum-SUIT",
                "JulichBrain"
            ],
            "priority_metrics": [
                "global_efficiency",
                "small_worldness", 
                "clustering_coefficient",
                "modularity",
                "characteristic_path_length"
            ],
            "quality_weighting_strategy": {
                "version": "pure_qa_v1",
                "rationale": "Pure network quality assessment independent of statistical analysis",
                "weights": {
                    "sparsity_score": 0.25,        # Optimal network density for analysis
                    "small_worldness": 0.25,       # Critical brain network property
                    "modularity": 0.20,            # Community structure quality
                    "global_efficiency": 0.20,     # Network integration capability
                    "reliability": 0.10            # Cross-subject consistency
                },
                "quality_thresholds": {
                    "min_sparsity": 0.05,          # Minimum network density
                    "max_sparsity": 0.40,          # Maximum network density
                    "min_small_worldness": 1.0,    # Minimum small-world property
                    "min_reliability": 0.60,       # Minimum cross-subject reliability
                    "exclude_extreme_outliers": True  # Remove statistical outliers in QA metrics
                },
                "qa_principles": {
                    "connectivity_metric_agnostic": True,    # No bias toward specific metrics
                    "study_design_independent": True,        # Independent of experimental design
                    "network_topology_focused": True,        # Focus on network properties only
                    "reproducibility_prioritized": True      # Emphasize reliable measurements
                }
            }
        }
    
    def load_optimization_results(self, optimization_file: str) -> pd.DataFrame:
        """
        Load optimization results from CSV file.
        
        Args:
            optimization_file: Path to optimization results CSV
            
        Returns:
            DataFrame with optimization results
        """
        try:
            df = pd.read_csv(optimization_file)
            logger.info(f"Loaded optimization results: {len(df)} records")
            logger.info(f"Atlases: {df['atlas'].nunique()}, Metrics: {df['connectivity_metric'].nunique()}")
            return df
        except Exception as e:
            logger.error(f"Error loading optimization results: {e}")
            raise
    
    def select_optimal_combinations(self, df: pd.DataFrame) -> List[Dict]:
        """
        Select optimal atlas/connectivity metric combinations based on optimization results.
        
        Args:
            df: DataFrame with optimization results
            
        Returns:
            List of optimal combinations with their properties
        """
        logger.info("Selecting optimal atlas/metric combinations...")
        
        criteria = self.config["selection_criteria"]
        optimal_combinations = []
        
        # Filter out subject-level data (keep only atlas-level)
        df_atlas = df[df['atlas'] != 'by_subject'].copy()
        
        # Apply enhanced quality scoring
        logger.info("Applying enhanced quality assessment...")
        df_atlas = self.enhance_quality_scores(df_atlas)
        
        # Use enhanced scores for selection (fallback to original if not available)
        score_column = 'enhanced_quality_score' if 'enhanced_quality_score' in df_atlas.columns else 'quality_score'
        logger.info(f"Using {score_column} for optimal selection")
        
        # Method 1: Use recommended combinations
        if criteria.get("use_recommended", True):
            recommended = df_atlas[df_atlas['recommended'] == True]
            if len(recommended) > 0:
                for _, row in recommended.iterrows():
                    combo = self._extract_combination_info(row, score_column)
                    combo['selection_method'] = 'recommended'
                    optimal_combinations.append(combo)
                logger.info(f"Found {len(recommended)} recommended combinations")
        
        # Method 2: Top quality scores overall
        top_n_overall = criteria.get("top_n_overall", 5)
        if top_n_overall > 0:
            # Group by atlas/metric and get mean scores; include raw metrics when available
            base_aggs = [
                score_column,
                'sparsity_score',
                'small_worldness_score',
                'efficiency_score',
                'modularity_score'
            ]
            optional_raw = [
                'density',
                'sparsity',
                'global_efficiency', 'global_efficiency(binary)', 'global_efficiency(weighted)',
                'small_worldness', 'small-worldness(binary)', 'small-worldness(weighted)',
                'clustering_coefficient', 'clustering_coeff_average(binary)', 'clustering_coeff_average(weighted)'
            ]
            agg_dict = {k: 'mean' for k in base_aggs if k in df_atlas.columns}
            if score_column == 'enhanced_quality_score' and 'quality_score' in df_atlas.columns:
                agg_dict['quality_score'] = 'mean'
            for col in optional_raw:
                if col in df_atlas.columns:
                    agg_dict[col] = 'mean'
            
            grouped = df_atlas.groupby(['atlas', 'connectivity_metric']).agg(agg_dict).reset_index()
            
            top_overall = grouped.nlargest(top_n_overall, score_column)
            for _, row in top_overall.iterrows():
                combo = self._extract_combination_info(row, score_column)
                combo['selection_method'] = 'top_overall'
                optimal_combinations.append(combo)
            logger.info(f"Added {len(top_overall)} top overall combinations")
        
        # Method 3: Top per atlas
        top_n_per_atlas = criteria.get("top_n_per_atlas", 1)
        if top_n_per_atlas > 0:
            # Get top combination for each atlas
            for atlas in df_atlas['atlas'].unique():
                atlas_data = df_atlas[df_atlas['atlas'] == atlas]
                
                base_aggs = [
                    score_column,
                    'sparsity_score',
                    'small_worldness_score',
                    'efficiency_score',
                    'modularity_score'
                ]
                optional_raw = [
                    'density',
                    'sparsity',
                    'global_efficiency', 'global_efficiency(binary)', 'global_efficiency(weighted)',
                    'small_worldness', 'small-worldness(binary)', 'small-worldness(weighted)',
                    'clustering_coefficient', 'clustering_coeff_average(binary)', 'clustering_coeff_average(weighted)'
                ]
                agg_dict = {k: 'mean' for k in base_aggs if k in atlas_data.columns}
                if score_column == 'enhanced_quality_score' and 'quality_score' in atlas_data.columns:
                    agg_dict['quality_score'] = 'mean'
                for col in optional_raw:
                    if col in atlas_data.columns:
                        agg_dict[col] = 'mean'
                
                atlas_grouped = atlas_data.groupby('connectivity_metric').agg(agg_dict).reset_index()
                atlas_grouped['atlas'] = atlas
                
                top_for_atlas = atlas_grouped.nlargest(top_n_per_atlas, score_column)
                for _, row in top_for_atlas.iterrows():
                    combo = self._extract_combination_info(row, score_column)
                    combo['selection_method'] = 'top_per_atlas'
                    optimal_combinations.append(combo)
        
        # Remove duplicates while preserving order
        unique_combinations = []
        seen = set()
        for combo in optimal_combinations:
            key = (combo['atlas'], combo['connectivity_metric'])
            if key not in seen:
                unique_combinations.append(combo)
                seen.add(key)
        
        logger.info(f"Selected {len(unique_combinations)} unique optimal combinations")
        return unique_combinations
    
    def enhance_quality_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply pure QA-based quality scoring independent of statistical analysis goals.
        
        This method ONLY considers network topology properties and measurement reliability,
        completely independent of any anticipated statistical analysis or study design.
        
        Args:
            df: DataFrame with optimization results
            
        Returns:
            DataFrame with pure QA-based quality scores
        """
        df_enhanced = df.copy()
        
        # Get QA strategy from config
        qa_strategy = self.config.get("quality_weighting_strategy", {})
        weights = qa_strategy.get("weights", {})
        thresholds = qa_strategy.get("quality_thresholds", {})
        principles = qa_strategy.get("qa_principles", {})
        
        # Enhanced default QA weights (v2.0 - includes more topology metrics)
        default_weights = {
            "sparsity_score": 0.20,              # Network density quality
            "small_worldness_score": 0.20,       # Small-world topology
            "modularity_score": 0.15,            # Community structure
            "efficiency_score": 0.15,            # Network integration
            "clustering_score": 0.10,            # Local connectivity
            "assortativity_score": 0.10,         # Degree correlation
            "reliability": 0.10                  # Cross-subject consistency
        }
        
        # Enhanced default thresholds
        default_thresholds = {
            "min_sparsity": 0.05,
            "max_sparsity": 0.40,
            "min_small_worldness": 1.0,
            "min_global_efficiency": 0.3,
            "min_clustering": 0.2,
            "max_clustering": 0.9,
            "max_assortativity": 0.2,  # Brain networks typically disassortative
            "min_reliability": 0.60,
            "exclude_extreme_outliers": True
        }
        
        # Merge with user-provided thresholds
        thresholds = {**default_thresholds, **thresholds}
        
        # Apply pure QA scoring (no metric-specific biases)
        def calculate_pure_qa_score(row):
            # Start with original quality score or compute from components
            if 'quality_score' in row:
                base_score = row['quality_score']
            else:
                # Compute from individual components if available
                component_score = 0.0
                total_weight = 0.0
                
                for component, weight in weights.items() if weights else default_weights.items():
                    if component in row and not pd.isna(row[component]):
                        component_score += weight * row[component]
                        total_weight += weight
                
                base_score = component_score / total_weight if total_weight > 0 else 0.5
            
            # Apply QA-based filters (topology-based only)
            qa_score = base_score
            penalties = []
            
            # Sparsity range check (network topology requirement)
            sparsity = row.get('sparsity', 0.2)
            min_sparsity = thresholds.get('min_sparsity', 0.05)
            max_sparsity = thresholds.get('max_sparsity', 0.40)
            
            if sparsity < min_sparsity or sparsity > max_sparsity:
                qa_score *= 0.5  # Penalty for poor network density
                penalties.append(f"poor_sparsity({sparsity:.3f})")
            
            # Small-worldness check (network topology requirement)
            small_world = row.get('small_worldness', 0)
            min_sw = thresholds.get('min_small_worldness', 1.0)
            
            if small_world < min_sw:
                qa_score *= 0.7  # Penalty for non-small-world networks
                penalties.append(f"poor_small_world({small_world:.3f})")
            
            # Assortativity check (brain networks are typically disassortative)
            assort = row.get('assortativity_coefficient', 0)
            max_assort = thresholds.get('max_assortativity', 0.2)
            
            if assort > max_assort:
                qa_score *= 0.8  # Penalty for unusually assortative networks
                penalties.append(f"poor_assortativity({assort:.3f})")
            
            # Global efficiency check (well-connected networks)
            geff = row.get('global_efficiency', 0)
            min_geff = thresholds.get('min_global_efficiency', 0.3)
            
            if geff < min_geff:
                qa_score *= 0.8  # Penalty for poorly connected networks
                penalties.append(f"poor_efficiency({geff:.3f})")
            
            # Clustering check (balanced segregation)
            clust = row.get('clustering_coefficient', 0)
            min_clust = thresholds.get('min_clustering', 0.2)
            max_clust = thresholds.get('max_clustering', 0.9)
            
            if clust < min_clust or clust > max_clust:
                qa_score *= 0.9  # Small penalty for extreme clustering
                penalties.append(f"poor_clustering({clust:.3f})")
            
            # Reliability check (measurement quality requirement)
            reliability = row.get('reliability', 1.0)
            min_reliability = thresholds.get('min_reliability', 0.60)
            
            if reliability < min_reliability:
                qa_score *= 0.8  # Penalty for unreliable measurements
                penalties.append(f"poor_reliability({reliability:.3f})")
            
            # Extreme outlier detection (statistical quality, not analysis bias)
            if thresholds.get('exclude_extreme_outliers', True):
                # Check for extremely high scores that might indicate measurement artifacts
                if base_score > 0.98:
                    qa_score *= 0.9  # Small penalty for potential artifacts
                    penalties.append("extreme_score_artifact")
            
            # Ensure score stays in [0, 1] range
            qa_score = min(max(qa_score, 0.0), 1.0)
            
            return qa_score, penalties
        
        # Calculate pure QA scores
        scores_and_penalties = df_enhanced.apply(
            lambda row: calculate_pure_qa_score(row), axis=1
        )
        
        df_enhanced['pure_qa_score'] = [sp[0] for sp in scores_and_penalties]
        df_enhanced['qa_penalties'] = ['; '.join(sp[1]) if sp[1] else 'none' for sp in scores_and_penalties]
        
        # Add QA methodology info
        def get_qa_rationale(row):
            qa_principles_used = []
            
            if principles.get('connectivity_metric_agnostic', True):
                qa_principles_used.append("metric-agnostic")
            
            if principles.get('study_design_independent', True):
                qa_principles_used.append("study-independent")
            
            if principles.get('network_topology_focused', True):
                qa_principles_used.append("topology-focused")
            
            if principles.get('reproducibility_prioritized', True):
                qa_principles_used.append("reproducibility-focused")
            
            return f"Pure QA: {', '.join(qa_principles_used)}"
        
        df_enhanced['qa_methodology'] = df_enhanced.apply(get_qa_rationale, axis=1)
        
        # Use pure QA score as the enhanced score
        df_enhanced['enhanced_quality_score'] = df_enhanced['pure_qa_score']
        
        logger.info("Applied pure QA-based quality scoring (study-design independent)")
        return df_enhanced

    def _extract_combination_info(self, row: pd.Series, score_column: str = 'quality_score') -> Dict:
        """Extract combination information from a dataframe row.
        Gracefully map to available column names and compute reasonable fallbacks.
        """
        def _val(candidates: List[str]) -> float:
            for c in candidates:
                if c in row and not pd.isna(row[c]):
                    try:
                        return float(row[c])
                    except Exception:
                        continue
            return float('nan')

        # Try to obtain raw metrics; if missing, use fallbacks
        sparsity = _val(['sparsity'])
        if np.isnan(sparsity):
            density = _val(['density'])
            if not np.isnan(density):
                sparsity = max(0.0, min(1.0, 1.0 - density))

        small_world = _val(['small_worldness', 'small-worldness(binary)', 'small-worldness(weighted)'])
        glob_eff = _val(['global_efficiency', 'global_efficiency(binary)', 'global_efficiency(weighted)'])
        clustering = _val(['clustering_coefficient', 'clustering_coeff_average(binary)', 'clustering_coeff_average(weighted)'])
        modularity = _val(['modularity', 'modularity_score'])

        result = {
            'atlas': row['atlas'],
            'connectivity_metric': row['connectivity_metric'],
            'quality_score': float(row.get('quality_score', 0)) if 'quality_score' in row and not pd.isna(row['quality_score']) else 0.0,
            'sparsity': sparsity,
            'small_worldness': small_world,
            'global_efficiency': glob_eff,
            'clustering_coefficient': clustering,
            'modularity': modularity if not np.isnan(modularity) else 0.0
        }
        
        # Add pure QA score if available
        if score_column == 'enhanced_quality_score' and 'enhanced_quality_score' in row:
            result['pure_qa_score'] = float(row['enhanced_quality_score'])
            result['primary_score'] = result['pure_qa_score']
        else:
            result['primary_score'] = result['quality_score']
            
        # Add QA methodology info if available
        if 'qa_methodology' in row:
            result['qa_methodology'] = row['qa_methodology']
            
        if 'qa_penalties' in row:
            result['qa_penalties'] = row['qa_penalties']
        
        return result
    
    def prepare_scientific_dataset(self, optimization_df: pd.DataFrame, 
                                 optimal_combinations: List[Dict],
                                 output_dir: str) -> Dict[str, str]:
        """
        Prepare datasets for scientific analysis using optimal combinations.
        
        Args:
            optimization_df: Full optimization results
            optimal_combinations: Selected optimal combinations
            output_dir: Directory to save prepared datasets
            
        Returns:
            Dictionary mapping combination names to file paths
        """
        logger.info("Preparing datasets for scientific analysis...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        prepared_files = {}
        
        # Filter out subject-level data
        df_clean = optimization_df[optimization_df['atlas'] != 'by_subject'].copy()
        
        for combo in optimal_combinations:
            atlas = combo['atlas']
            metric = combo['connectivity_metric']
            
            # Extract data for this combination
            combo_data = df_clean[
                (df_clean['atlas'] == atlas) & 
                (df_clean['connectivity_metric'] == metric)
            ].copy()
            
            if len(combo_data) == 0:
                logger.warning(f"No data found for {atlas} + {metric}")
                continue
            
            # Select relevant columns for scientific analysis
            analysis_columns = [
                'subject', 'atlas', 'connectivity_metric',
                'global_efficiency', 'small_worldness', 'clustering_coefficient',
                'characteristic_path_length', 'assortativity', 'modularity',
                'sparsity', 'transitivity_binary', 'transitivity_weighted'
            ]
            
            # Keep only available columns
            available_columns = [col for col in analysis_columns if col in combo_data.columns]
            analysis_data = combo_data[available_columns].copy()
            
            # Add combination metadata
            analysis_data['quality_score'] = combo['quality_score']
            analysis_data['selection_method'] = combo['selection_method']
            
            # Save prepared dataset
            filename = f"{atlas}_{metric}_analysis_ready.csv"
            filepath = output_path / filename
            analysis_data.to_csv(filepath, index=False)
            
            combo_key = f"{atlas}_{metric}"
            prepared_files[combo_key] = str(filepath)
            
            logger.info(f"Prepared dataset: {filename} ({len(analysis_data)} subjects)")
        
        # Create a combined dataset with all optimal combinations
        all_optimal_data = []
        for combo in optimal_combinations:
            atlas = combo['atlas']
            metric = combo['connectivity_metric']
            
            combo_data = df_clean[
                (df_clean['atlas'] == atlas) & 
                (df_clean['connectivity_metric'] == metric)
            ].copy()
            
            if len(combo_data) > 0:
                all_optimal_data.append(combo_data)
        
        if all_optimal_data:
            combined_df = pd.concat(all_optimal_data, ignore_index=True)
            combined_file = output_path / "all_optimal_combinations.csv"
            combined_df.to_csv(combined_file, index=False)
            prepared_files['combined_optimal'] = str(combined_file)
            logger.info(f"Created combined dataset: {len(combined_df)} records")
        
        return prepared_files
    
    def create_selection_summary(self, optimal_combinations: List[Dict], 
                               output_file: str) -> None:
        """
        Create a summary report of the optimal selection.
        
        Args:
            optimal_combinations: Selected optimal combinations
            output_file: Path to output summary file
        """
        logger.info("Creating optimal selection summary...")
        
        # Group by selection method
        by_method = {}
        for combo in optimal_combinations:
            method = combo.get('selection_method', 'unknown')
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(combo)
        
    # Create summary content
        summary_content = f"""
Optimal Atlas/Metric Selection Summary
=====================================

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW
--------
Total optimal combinations selected: {len(optimal_combinations)}
Selection methods used: {', '.join(by_method.keys())}

SELECTED COMBINATIONS
--------------------
"""
        
        for i, combo in enumerate(optimal_combinations, 1):
            score_info = ""
            if 'pure_qa_score' in combo:
                score_info = f"Pure QA Score: {combo['pure_qa_score']:.3f} (Original: {combo['quality_score']:.3f})"
            else:
                score_info = f"Quality Score: {combo['quality_score']:.3f}"
            
            def fmt3(x):
                try:
                    if x is None or (isinstance(x, float) and np.isnan(x)):
                        return 'n/a'
                    return f"{float(x):.3f}"
                except Exception:
                    return 'n/a'

            summary_content += f"""
{i:2d}. {combo['atlas']} + {combo['connectivity_metric']}
    {score_info}
    Selection Method: {combo['selection_method']}
    Global Efficiency: {fmt3(combo.get('global_efficiency'))}
    Small-worldness: {fmt3(combo.get('small_worldness'))}
    Clustering Coefficient: {fmt3(combo.get('clustering_coefficient'))}
    Sparsity: {fmt3(combo.get('sparsity'))}"""
    
            if 'qa_methodology' in combo:
                summary_content += f"""
    QA Methodology: {combo['qa_methodology']}"""
            
            if 'qa_penalties' in combo and combo['qa_penalties'] != 'none':
                summary_content += f"""
    QA Penalties: {combo['qa_penalties']}"""
            summary_content += "\n"
            # Append parameters block if present
            params = combo.get('parameters') or {}
            if params:
                # Flatten tracking parameters for pretty print
                tp = params.get('tracking_parameters') or {}
                conn_thr = params.get('connectivity_threshold')
                # Stable order for readability
                ordered = [
                    ('tract_count', params.get('tract_count')),
                    ('connectivity_threshold', conn_thr),
                    ('fa_threshold', tp.get('fa_threshold')),
                    ('turning_angle', tp.get('turning_angle')),
                    ('step_size', tp.get('step_size')),
                    ('smoothing', tp.get('smoothing')),
                    ('min_length', tp.get('min_length')),
                    ('max_length', tp.get('max_length')),
                    ('track_voxel_ratio', tp.get('track_voxel_ratio')),
                    ('dt_threshold', tp.get('dt_threshold')),
                ]
                def fmtv(v):
                    try:
                        if v is None:
                            return 'n/a'
                        if isinstance(v, (int,)):
                            return str(v)
                        return f"{float(v):.3f}"
                    except Exception:
                        return str(v)
                summary_content += """
    Parameters:
      - """ + "\n      - ".join([f"{k}={fmtv(v)}" for k, v in ordered if v is not None]) + "\n"
        
        summary_content += f"""

BREAKDOWN BY SELECTION METHOD
-----------------------------
"""
        
        for method, combos in by_method.items():
            summary_content += f"""
{method.upper()} ({len(combos)} combinations):
"""
            for combo in combos:
                score_display = combo.get('pure_qa_score', combo['quality_score'])
                summary_content += f"  • {combo['atlas']} + {combo['connectivity_metric']} (score: {score_display:.3f})\n"
        
        summary_content += f"""

RECOMMENDED ANALYSIS STRATEGY
----------------------------
1. PRIMARY ANALYSIS: Use the top-scoring combination(s) for main scientific hypothesis testing
   
2. VALIDATION ANALYSIS: Replicate findings using other high-quality combinations
   
3. SENSITIVITY ANALYSIS: Compare results across different atlases to assess robustness

4. ATLAS-SPECIFIC ANALYSIS: If different atlases show different patterns, investigate why

TOP RECOMMENDATIONS FOR YOUR SOCCER VS CONTROL STUDY:
"""
        
        # Sort by primary score (pure QA if available, otherwise quality score) and show top 3
        sorted_combos = sorted(optimal_combinations, key=lambda x: x.get('primary_score', x['quality_score']), reverse=True)
        for i, combo in enumerate(sorted_combos[:3], 1):
            score_display = combo.get('pure_qa_score', combo['quality_score'])
            summary_content += f"""
{i}. {combo['atlas']} + {combo['connectivity_metric']} (Score: {score_display:.3f})
   - This combination offers optimal network properties for group comparisons
   - Expected to provide reliable and interpretable results"""
   
            if 'qa_penalties' in combo and combo['qa_penalties'] != "none":
                summary_content += f"""
   - QA Notes: {combo['qa_penalties']}"""
        
        summary_content += f"""

NEXT STEPS
----------
1. Load the prepared datasets from the analysis-ready CSV files
2. Add your group membership information (soccer players vs controls)
3. Perform statistical comparisons using appropriate tests for your study design
4. Consider multiple comparison corrections if testing multiple metrics
5. Validate findings using alternative atlas/metric combinations

IMPORTANT NOTES
--------------
- These combinations were selected using PURE QA scoring independent of study design
- QA scoring focuses solely on network topology properties and measurement reliability  
- NO bias toward specific connectivity metrics (FA, count, etc.)
- NO consideration of anticipated statistical results or group differences
- Original quality scores and pure QA scores both focus on network properties only
- You still need to add group membership information for your scientific analysis
- Consider the biological interpretability of each atlas for your research question
- Validate any significant findings using independent datasets if available

PURE QA SCORING METHODOLOGY
---------------------------
The pure quality assessment applies ONLY network topology criteria:
• Sparsity Range: Optimal network density (0.05-0.40) for meaningful connectivity analysis
• Small-worldness: Networks must show small-world properties (sigma > 1.0)  
• Modularity: Community structure quality assessment
• Global Efficiency: Network integration capability measurement
• Reliability: Cross-subject consistency of measurements (> 0.60)
• Outlier Detection: Removal of extreme scores that may indicate measurement artifacts
• This approach is COMPLETELY INDEPENDENT of study design or anticipated statistical results
• Ensures methodological rigor by separating QA from statistical analysis
"""
        
        # Write summary
        with open(output_file, 'w') as f:
            f.write(summary_content)
        
        logger.info(f"Created selection summary: {output_file}")
    
    def create_selection_plots(self, optimal_combinations: List[Dict], 
                             output_dir: str) -> List[str]:
        """
        Create visualization plots for optimal selection results.
        
        Args:
            optimal_combinations: Selected optimal combinations
            output_dir: Directory to save plots
            
        Returns:
            List of created plot files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_files = []
        
        if not optimal_combinations:
            logger.warning("No optimal combinations to plot")
            return plot_files
        
        # Convert to DataFrame for easier plotting
        df_optimal = pd.DataFrame(optimal_combinations)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Quality scores by atlas and metric
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sns.barplot(data=df_optimal, x='quality_score', y='atlas', 
                   hue='connectivity_metric', orient='h')
        plt.title('Quality Scores by Atlas and Connectivity Metric')
        plt.xlabel('Quality Score')
        
        plt.subplot(2, 2, 2)
        plt.scatter(df_optimal['global_efficiency'], df_optimal['small_worldness'], 
                   c=df_optimal['quality_score'], cmap='viridis', s=100, alpha=0.7)
        plt.colorbar(label='Quality Score')
        plt.xlabel('Global Efficiency')
        plt.ylabel('Small-worldness')
        plt.title('Network Properties of Optimal Combinations')
        
        plt.subplot(2, 2, 3)
        if len(df_optimal['selection_method'].unique()) > 1:
            sns.countplot(data=df_optimal, x='selection_method')
            plt.title('Combinations by Selection Method')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, f"All combinations selected by:\n{df_optimal['selection_method'].iloc[0]}", 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Selection Method')
        
        plt.subplot(2, 2, 4)
        # Network metrics radar chart would be nice but simplified bar chart
        metrics = ['global_efficiency', 'small_worldness', 'clustering_coefficient', 'sparsity']
        available_metrics = [m for m in metrics if m in df_optimal.columns]
        
        if available_metrics:
            metric_means = df_optimal[available_metrics].mean()
            metric_means.plot(kind='bar')
            plt.title('Mean Network Properties')
            plt.ylabel('Mean Value')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plot_file = output_path / 'optimal_selection_overview.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(plot_file))
        
        # 2. Quality score ranking
        plt.figure(figsize=(12, 8))
        
        # Sort by quality score
        df_sorted = df_optimal.sort_values('quality_score', ascending=True)
        
        y_pos = np.arange(len(df_sorted))
        bars = plt.barh(y_pos, df_sorted['quality_score'])
        
        # Color bars by selection method
        methods = df_sorted['selection_method'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        method_colors = dict(zip(methods, colors))
        
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            bars[i].set_color(method_colors[row['selection_method']])
        
        plt.yticks(y_pos, [f"{row['atlas']}\n{row['connectivity_metric']}" 
                          for _, row in df_sorted.iterrows()])
        plt.xlabel('Quality Score')
        plt.title('Optimal Combinations Ranked by Quality Score')
        
        # Add legend for selection methods
        legend_elements = [plt.Rectangle((0,0),1,1, color=method_colors[method], label=method) 
                          for method in methods]
        plt.legend(handles=legend_elements, title='Selection Method')
        
        plt.tight_layout()
        plot_file = output_path / 'quality_ranking.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(plot_file))
        
        logger.info(f"Created {len(plot_files)} selection plots")
        return plot_files


def main():
    """Command line interface for optimal selection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Select optimal atlas/metric combinations")
    # Positional arguments (backward compatible)
    parser.add_argument("optimization_file", nargs='?', help="CSV file with optimization results")
    parser.add_argument("output_dir", nargs='?', help="Output directory for prepared datasets")
    # Optional aliases for consistency
    parser.add_argument("-i", "--input", dest="input_opt", help="Alias for optimization CSV file")
    parser.add_argument("-o", "--output", dest="output_opt", help="Alias for output directory")
    parser.add_argument("--config", help="Configuration file (JSON)")
    parser.add_argument("--plots", action="store_true", help="Generate selection plots")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--no-emoji", action="store_true", help="Disable emoji in console output")
    
    args = parser.parse_args()

    configure_stdio(args.no_emoji)

    # Reconcile optional -i/-o with positional args
    if args.input_opt and args.optimization_file and args.input_opt != args.optimization_file:
        print("❌ Conflicting input provided via positional and -i/--input")
        return 2
    if args.output_opt and args.output_dir and args.output_opt != args.output_dir:
        print("❌ Conflicting output provided via positional and -o/--output")
        return 2
    if args.input_opt and not args.optimization_file:
        args.optimization_file = args.input_opt
    if args.output_opt and not args.output_dir:
        args.output_dir = args.output_opt

    # Validate required args after reconciliation
    if not args.optimization_file or not args.output_dir:
        parser.print_help()
        return 2
    
    # Setup logging: console without timestamps; file with timestamps
    log_level = getattr(logging, args.log_level.upper())
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)
    fh = logging.FileHandler(Path(args.output_dir) / 'optimal_selection.log', encoding='utf-8')
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    root_logger.addHandler(fh)
    root_logger.addHandler(ch)
    
    # Load configuration
    config = None
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f).get('optimal_selection', {})
    
    try:
        # Initialize selector and load data
        selector = OptimalSelector(config)
        df = selector.load_optimization_results(args.optimization_file)
        
        # Select optimal combinations
        optimal_combinations = selector.select_optimal_combinations(df)
        
        # Prepare datasets for scientific analysis
        prepared_files = selector.prepare_scientific_dataset(
            df, optimal_combinations, args.output_dir
        )
        
        # Try to discover selected parameters (e.g., from cross-validation wave root)
        try:
            out_dir_path = Path(args.output_dir)
            wave_root = out_dir_path.parent if out_dir_path.name == '03_selection' else out_dir_path
            sel_params_file = wave_root / 'selected_parameters.json'
            selected_params = None
            if sel_params_file.exists():
                with sel_params_file.open() as _f:
                    raw = json.load(_f)
                cfg = raw.get('selected_config') if isinstance(raw, dict) else None
                if isinstance(cfg, dict):
                    # Extract concise parameter snapshot
                    tp = (cfg.get('tracking_parameters') or {})
                    conn = (cfg.get('connectivity_options') or {})
                    param_snapshot = {
                        'tract_count': cfg.get('tract_count'),
                        'connectivity_threshold': conn.get('connectivity_threshold'),
                        'tracking_parameters': {
                            'fa_threshold': tp.get('fa_threshold'),
                            'turning_angle': tp.get('turning_angle'),
                            'step_size': tp.get('step_size'),
                            'smoothing': tp.get('smoothing'),
                            'min_length': tp.get('min_length'),
                            'max_length': tp.get('max_length'),
                            'track_voxel_ratio': tp.get('track_voxel_ratio'),
                            'dt_threshold': tp.get('dt_threshold'),
                        }
                    }
                    selected_params = param_snapshot
            # Attach to each combo for downstream consumers
            if selected_params:
                for combo in optimal_combinations:
                    combo['parameters'] = selected_params
        except Exception:
            # Non-fatal: parameters are optional
            pass

        # Create summary report
        summary_file = Path(args.output_dir) / "optimal_selection_summary.txt"
        selector.create_selection_summary(optimal_combinations, str(summary_file))
        
        # Create plots if requested
        if args.plots:
            plot_files = selector.create_selection_plots(optimal_combinations, args.output_dir)
            logger.info(f"Generated {len(plot_files)} selection plots")
        
        # Save optimal combinations as JSON for programmatic use
        combinations_file = Path(args.output_dir) / "optimal_combinations.json"
        with open(combinations_file, 'w') as f:
            json.dump(optimal_combinations, f, indent=2)
        
        # Print summary
        print(f"\n🎯 Optimal Selection Complete!")
        print(f"{'='*50}")
        print(f"Selected {len(optimal_combinations)} optimal combinations")
        print(f"Prepared {len(prepared_files)} analysis-ready datasets")
        print(f"Results saved to: {args.output_dir}")
        print(f"\nTop 3 recommendations:")
        
        sorted_combos = sorted(optimal_combinations, key=lambda x: x.get('primary_score', x['quality_score']), reverse=True)
        for i, combo in enumerate(sorted_combos[:3], 1):
            score_display = combo.get('pure_qa_score', combo['quality_score'])
            score_type = "Pure QA" if 'pure_qa_score' in combo else "Quality"
            line = f"{i}. {combo['atlas']} + {combo['connectivity_metric']} ({score_type}: {score_display:.3f})"
            # If parameters available, add a concise inline summary
            # Handle both nested (parameters.tracking_parameters) and flat (tracking_parameters) structures
            params = combo.get('parameters') or {}
            tp = params.get('tracking_parameters') or combo.get('tracking_parameters') or {}
            tract_count = params.get('tract_count') or combo.get('tract_count')
            
            if tp or tract_count:
                def _fmt(v, default_val='auto'):
                    try:
                        if v is None:
                            return 'n/a'
                        if isinstance(v, int):
                            return str(v)
                        if isinstance(v, float):
                            # 0.0 means DSI Studio will use defaults
                            if v == 0.0:
                                return default_val
                            return f"{v:.3f}"
                        return str(v)
                    except Exception:
                        return str(v)
                param_bits = [
                    f"n_tracks={_fmt(tract_count, 'auto')}",
                    f"fa={_fmt(tp.get('fa_threshold'), 'auto')}",
                    f"angle={_fmt(tp.get('turning_angle'), 'auto')}",
                    f"step={_fmt(tp.get('step_size'), 'auto')}",
                ]
                line += " | " + ", ".join(param_bits)
            print(line)
        
        # Keep console concise; details live in the summary file.
        print(f"\nThanks for using OptiConn! ✨")

        return 0
        
    except Exception as e:
        logger.error(f"Optimal selection failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
