#!/usr/bin/env python3
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
            ]
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
        
        # Method 1: Use recommended combinations
        if criteria.get("use_recommended", True):
            recommended = df_atlas[df_atlas['recommended'] == True]
            if len(recommended) > 0:
                for _, row in recommended.iterrows():
                    combo = self._extract_combination_info(row)
                    combo['selection_method'] = 'recommended'
                    optimal_combinations.append(combo)
                logger.info(f"Found {len(recommended)} recommended combinations")
        
        # Method 2: Top quality scores overall
        top_n_overall = criteria.get("top_n_overall", 5)
        if top_n_overall > 0:
            # Group by atlas/metric and get mean quality score
            grouped = df_atlas.groupby(['atlas', 'connectivity_metric']).agg({
                'quality_score': 'mean',
                'sparsity': 'mean',
                'small_worldness': 'mean',
                'global_efficiency': 'mean',
                'clustering_coefficient': 'mean',
                'modularity': 'mean'
            }).reset_index()
            
            top_overall = grouped.nlargest(top_n_overall, 'quality_score')
            for _, row in top_overall.iterrows():
                combo = self._extract_combination_info(row)
                combo['selection_method'] = 'top_overall'
                optimal_combinations.append(combo)
            logger.info(f"Added {len(top_overall)} top overall combinations")
        
        # Method 3: Top per atlas
        top_n_per_atlas = criteria.get("top_n_per_atlas", 1)
        if top_n_per_atlas > 0:
            # Get top combination for each atlas
            for atlas in df_atlas['atlas'].unique():
                atlas_data = df_atlas[df_atlas['atlas'] == atlas]
                atlas_grouped = atlas_data.groupby('connectivity_metric').agg({
                    'quality_score': 'mean',
                    'sparsity': 'mean',
                    'small_worldness': 'mean',
                    'global_efficiency': 'mean',
                    'clustering_coefficient': 'mean',
                    'modularity': 'mean'
                }).reset_index()
                atlas_grouped['atlas'] = atlas
                
                top_for_atlas = atlas_grouped.nlargest(top_n_per_atlas, 'quality_score')
                for _, row in top_for_atlas.iterrows():
                    combo = self._extract_combination_info(row)
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
    
    def _extract_combination_info(self, row: pd.Series) -> Dict:
        """Extract combination information from a dataframe row."""
        return {
            'atlas': row['atlas'],
            'connectivity_metric': row['connectivity_metric'],
            'quality_score': float(row['quality_score']),
            'sparsity': float(row.get('sparsity', 0)),
            'small_worldness': float(row.get('small_worldness', 0)),
            'global_efficiency': float(row.get('global_efficiency', 0)),
            'clustering_coefficient': float(row.get('clustering_coefficient', 0)),
            'modularity': float(row.get('modularity', 0)) if 'modularity' in row and not pd.isna(row['modularity']) else 0.0
        }
    
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
            summary_content += f"""
{i:2d}. {combo['atlas']} + {combo['connectivity_metric']}
    Quality Score: {combo['quality_score']:.3f}
    Selection Method: {combo['selection_method']}
    Global Efficiency: {combo['global_efficiency']:.3f}
    Small-worldness: {combo['small_worldness']:.3f}
    Clustering Coefficient: {combo['clustering_coefficient']:.3f}
    Sparsity: {combo['sparsity']:.3f}
"""
        
        summary_content += f"""

BREAKDOWN BY SELECTION METHOD
-----------------------------
"""
        
        for method, combos in by_method.items():
            summary_content += f"""
{method.upper()} ({len(combos)} combinations):
"""
            for combo in combos:
                summary_content += f"  • {combo['atlas']} + {combo['connectivity_metric']} (score: {combo['quality_score']:.3f})\n"
        
        summary_content += f"""

RECOMMENDED ANALYSIS STRATEGY
----------------------------
1. PRIMARY ANALYSIS: Use the top-scoring combination(s) for main scientific hypothesis testing
   
2. VALIDATION ANALYSIS: Replicate findings using other high-quality combinations
   
3. SENSITIVITY ANALYSIS: Compare results across different atlases to assess robustness

4. ATLAS-SPECIFIC ANALYSIS: If different atlases show different patterns, investigate why

TOP RECOMMENDATIONS FOR YOUR SOCCER VS CONTROL STUDY:
"""
        
        # Sort by quality score and show top 3
        sorted_combos = sorted(optimal_combinations, key=lambda x: x['quality_score'], reverse=True)
        for i, combo in enumerate(sorted_combos[:3], 1):
            summary_content += f"""
{i}. {combo['atlas']} + {combo['connectivity_metric']} (Quality: {combo['quality_score']:.3f})
   - This combination offers optimal network properties for group comparisons
   - Expected to provide reliable and interpretable results
"""
        
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
- These combinations were selected based on network quality properties
- You still need to add group membership information for your scientific analysis
- Consider the biological interpretability of each atlas for your research question
- Validate any significant findings using independent datasets if available
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
    parser.add_argument("optimization_file", help="CSV file with optimization results")
    parser.add_argument("output_dir", help="Output directory for prepared datasets")
    parser.add_argument("--config", help="Configuration file (JSON)")
    parser.add_argument("--plots", action="store_true", help="Generate selection plots")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(args.output_dir) / 'optimal_selection.log'),
            logging.StreamHandler()
        ]
    )
    
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
        
        sorted_combos = sorted(optimal_combinations, key=lambda x: x['quality_score'], reverse=True)
        for i, combo in enumerate(sorted_combos[:3], 1):
            print(f"{i}. {combo['atlas']} + {combo['connectivity_metric']} (quality: {combo['quality_score']:.3f})")
        
        print(f"\n📊 Next steps:")
        print(f"1. Add group membership (soccer vs control) to your datasets")
        print(f"2. Use prepared CSV files for your scientific analysis")
        print(f"3. Start with the highest quality combination for primary analysis")
        
        return 0
        
    except Exception as e:
        logger.error(f"Optimal selection failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
