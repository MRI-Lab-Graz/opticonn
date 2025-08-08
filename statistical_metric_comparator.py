#!/usr/bin/env python3
"""
Statistical Connectivity Metric Comparator
==========================================

Performs statistical comparisons between different connectivity metrics 
(count vs ncount2, fa vs qa) to identify which metrics show significant
differences in network properties for brain network analysis.

Author: Braingraph Pipeline Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, shapiro, levene
from statsmodels.stats.multitest import multipletests
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ConnectivityMetricComparator:
    """
    Statistical comparison of connectivity metrics for brain network analysis.
    
    This class performs statistical tests to determine which connectivity metrics
    (count, ncount2, fa, qa) provide significantly different network properties
    for the same subjects and atlases.
    """
    
    def __init__(self, alpha: float = 0.05, correction_method: str = 'fdr_bh'):
        """
        Initialize the comparator.
        
        Args:
            alpha: Significance level for statistical tests
            correction_method: Multiple comparison correction method
        """
        self.alpha = alpha
        self.correction_method = correction_method
    
    def compare_metrics(self, df: pd.DataFrame, 
                       atlas_filter: Optional[List[str]] = None,
                       metrics_to_compare: Optional[List[str]] = None,
                       target_metrics: Optional[List[str]] = None) -> Dict:
        """
        Perform statistical comparisons between connectivity metrics.
        
        Args:
            df: DataFrame with graph metrics (one row per subject/atlas/metric)
            atlas_filter: List of atlases to include (None = all)
            metrics_to_compare: Connectivity metrics to compare (default: ['count', 'ncount2', 'fa', 'qa'])
            target_metrics: Network metrics to analyze (default: brain network metrics)
            
        Returns:
            Dictionary with statistical comparison results
        """
        logger.info("🔬 Starting statistical comparison of connectivity metrics...")
        
        # Set defaults
        if metrics_to_compare is None:
            metrics_to_compare = ['count', 'ncount2', 'fa', 'qa']
        
        if target_metrics is None:
            target_metrics = [
                'small_worldness', 'global_efficiency', 'clustering_coefficient',
                'characteristic_path_length', 'assortativity', 'sparsity'
            ]
        
        # Filter available metrics
        available_conn_metrics = [m for m in metrics_to_compare if m in df['connectivity_metric'].unique()]
        available_target_metrics = [m for m in target_metrics if m in df.columns]
        
        logger.info(f"📊 Comparing connectivity metrics: {available_conn_metrics}")
        logger.info(f"🧠 Analyzing network metrics: {available_target_metrics}")
        
        results = {
            'comparisons': [],
            'summary_stats': {},
            'effect_sizes': {},
            'best_metrics_per_atlas': {},
            'recommendations': {}
        }
        
        # Filter atlases if specified
        if atlas_filter:
            df_filtered = df[df['atlas'].isin(atlas_filter)]
            logger.info(f"🎯 Filtered to atlases: {atlas_filter}")
        else:
            df_filtered = df.copy()
        
        # Remove subject organization if present
        df_filtered = df_filtered[df_filtered['atlas'] != 'by_subject']
        
        # Perform comparisons for each atlas
        for atlas in df_filtered['atlas'].unique():
            atlas_data = df_filtered[df_filtered['atlas'] == atlas]
            atlas_results = {'atlas': atlas, 'metric_comparisons': {}}
            
            logger.info(f"🔍 Analyzing atlas: {atlas}")
            
            # For each target metric, compare across connectivity metrics
            for target_metric in available_target_metrics:
                if target_metric not in atlas_data.columns:
                    continue
                    
                metric_data = {}
                
                # Collect data for each connectivity metric
                for conn_metric in available_conn_metrics:
                    conn_data = atlas_data[atlas_data['connectivity_metric'] == conn_metric]
                    if len(conn_data) > 0:
                        values = conn_data[target_metric].dropna()
                        if len(values) > 3:  # Minimum sample size
                            metric_data[conn_metric] = values.values
                
                if len(metric_data) < 2:  # Need at least 2 groups to compare
                    continue
                
                # Perform pairwise comparisons
                comparison_results = self._perform_pairwise_comparisons(
                    metric_data, target_metric, atlas
                )
                
                if comparison_results:
                    atlas_results['metric_comparisons'][target_metric] = comparison_results
            
            if atlas_results['metric_comparisons']:
                results['comparisons'].append(atlas_results)
        
        # Generate summary statistics
        results['summary_stats'] = self._generate_summary(results['comparisons'])
        
        # Calculate effect sizes
        results['effect_sizes'] = self._calculate_effect_sizes(
            df_filtered, available_conn_metrics, available_target_metrics
        )
        
        # Identify best metrics per atlas
        results['best_metrics_per_atlas'] = self._identify_best_metrics(results['comparisons'])
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        logger.info("✅ Statistical comparison completed!")
        return results
    
    def _perform_pairwise_comparisons(self, metric_data: Dict[str, np.ndarray], 
                                    target_metric: str, atlas: str) -> List[Dict]:
        """Perform pairwise statistical comparisons between connectivity metrics."""
        comparisons = []
        metric_names = list(metric_data.keys())
        
        for i in range(len(metric_names)):
            for j in range(i + 1, len(metric_names)):
                metric1, metric2 = metric_names[i], metric_names[j]
                data1, data2 = metric_data[metric1], metric_data[metric2]
                
                comparison = self._statistical_test(data1, data2, metric1, metric2)
                comparison['target_metric'] = target_metric
                comparison['atlas'] = atlas
                
                comparisons.append(comparison)
        
        return comparisons
    
    def _statistical_test(self, data1: np.ndarray, data2: np.ndarray, 
                         metric1: str, metric2: str) -> Dict:
        """Perform appropriate statistical test based on data properties."""
        
        comparison = {
            'metric1': metric1,
            'metric2': metric2,
            'n1': len(data1),
            'n2': len(data2),
            'mean1': np.mean(data1),
            'mean2': np.mean(data2),
            'std1': np.std(data1, ddof=1),
            'std2': np.std(data2, ddof=1),
            'median1': np.median(data1),
            'median2': np.median(data2)
        }
        
        # Test assumptions
        normal1 = normal2 = equal_var = False
        
        # Normality tests (if sample size >= 3)
        if len(data1) >= 3:
            _, p_norm1 = shapiro(data1)
            comparison['normality_p1'] = p_norm1
            normal1 = p_norm1 > 0.05
        
        if len(data2) >= 3:
            _, p_norm2 = shapiro(data2)
            comparison['normality_p2'] = p_norm2
            normal2 = p_norm2 > 0.05
        
        # Equal variance test
        if len(data1) >= 2 and len(data2) >= 2:
            _, p_levene = levene(data1, data2)
            comparison['equal_var_p'] = p_levene
            equal_var = p_levene > 0.05
        
        # Choose appropriate test
        use_parametric = normal1 and normal2 and equal_var
        
        if use_parametric:
            # Independent t-test
            stat, p_value = ttest_ind(data1, data2, equal_var=True)
            comparison['test_used'] = 't-test'
            comparison['test_type'] = 'parametric'
        else:
            # Mann-Whitney U test
            stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
            comparison['test_used'] = 'mann-whitney'
            comparison['test_type'] = 'non-parametric'
        
        comparison['statistic'] = stat
        comparison['p_value'] = p_value
        
        # Effect size (Cohen's d)
        cohens_d = self._calculate_cohens_d(data1, data2)
        comparison['cohens_d'] = cohens_d
        comparison['effect_size_category'] = self._categorize_effect_size(abs(cohens_d))
        
        # Practical significance
        comparison['practically_significant'] = abs(cohens_d) >= 0.5
        
        return comparison
    
    def _calculate_cohens_d(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(data1), len(data2)
        
        if n1 <= 1 or n2 <= 1:
            return 0.0
        
        # Pooled standard deviation
        s1, s2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(data1) - np.mean(data2)) / pooled_std
    
    def _categorize_effect_size(self, abs_cohens_d: float) -> str:
        """Categorize effect size magnitude."""
        if abs_cohens_d < 0.2:
            return 'negligible'
        elif abs_cohens_d < 0.5:
            return 'small'
        elif abs_cohens_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _generate_summary(self, comparisons: List[Dict]) -> Dict:
        """Generate summary statistics for all comparisons."""
        all_comparisons = []
        
        for atlas_result in comparisons:
            for target_metric, comps in atlas_result['metric_comparisons'].items():
                all_comparisons.extend(comps)
        
        if not all_comparisons:
            return {}
        
        df_comps = pd.DataFrame(all_comparisons)
        
        # Apply multiple comparison correction
        if 'p_value' in df_comps.columns and len(df_comps) > 0:
            p_values = df_comps['p_value'].dropna()
            if len(p_values) > 0:
                _, p_corrected, _, _ = multipletests(p_values, method=self.correction_method)
                df_comps['p_corrected'] = p_corrected
        
        summary = {
            'total_comparisons': len(df_comps),
            'significant_uncorrected': len(df_comps[df_comps['p_value'] < self.alpha]),
            'significant_corrected': len(df_comps[df_comps.get('p_corrected', pd.Series([1])) < self.alpha]),
            'mean_effect_size': df_comps['cohens_d'].abs().mean(),
            'median_effect_size': df_comps['cohens_d'].abs().median(),
            'large_effects': len(df_comps[df_comps['effect_size_category'] == 'large']),
            'medium_effects': len(df_comps[df_comps['effect_size_category'] == 'medium']),
            'small_effects': len(df_comps[df_comps['effect_size_category'] == 'small']),
            'parametric_tests': len(df_comps[df_comps['test_type'] == 'parametric']),
            'non_parametric_tests': len(df_comps[df_comps['test_type'] == 'non-parametric'])
        }
        
        return summary
    
    def _calculate_effect_sizes(self, df: pd.DataFrame, conn_metrics: List[str], 
                              target_metrics: List[str]) -> Dict:
        """Calculate effect sizes between connectivity metrics."""
        effect_sizes = {}
        
        for atlas in df['atlas'].unique():
            atlas_data = df[df['atlas'] == atlas]
            atlas_effects = {}
            
            for target_metric in target_metrics:
                if target_metric not in atlas_data.columns:
                    continue
                
                metric_stats = {}
                for conn_metric in conn_metrics:
                    conn_data = atlas_data[atlas_data['connectivity_metric'] == conn_metric]
                    if len(conn_data) > 0:
                        values = conn_data[target_metric].dropna()
                        if len(values) > 0:
                            metric_stats[conn_metric] = {
                                'mean': np.mean(values),
                                'std': np.std(values, ddof=1),
                                'median': np.median(values),
                                'n': len(values)
                            }
                
                if len(metric_stats) > 1:
                    atlas_effects[target_metric] = metric_stats
            
            if atlas_effects:
                effect_sizes[atlas] = atlas_effects
        
        return effect_sizes
    
    def _identify_best_metrics(self, comparisons: List[Dict]) -> Dict:
        """Identify the best connectivity metrics for each atlas."""
        best_metrics = {}
        
        for atlas_result in comparisons:
            atlas = atlas_result['atlas']
            metric_scores = {}
            
            # Score metrics based on statistical performance
            for target_metric, comps in atlas_result['metric_comparisons'].items():
                for comp in comps:
                    metric1, metric2 = comp['metric1'], comp['metric2']
                    
                    # Initialize scores
                    if metric1 not in metric_scores:
                        metric_scores[metric1] = []
                    if metric2 not in metric_scores:
                        metric_scores[metric2] = []
                    
                    # Score based on statistical significance and effect size
                    if comp.get('p_corrected', comp.get('p_value', 1)) < self.alpha:
                        effect_size = abs(comp.get('cohens_d', 0))
                        
                        # Award points for being the higher-performing metric
                        if comp['mean1'] > comp['mean2']:
                            metric_scores[metric1].append(effect_size)
                            metric_scores[metric2].append(0)
                        else:
                            metric_scores[metric2].append(effect_size)
                            metric_scores[metric1].append(0)
                    else:
                        # No significant difference - neutral score
                        metric_scores[metric1].append(0.1)  # Small positive for being included
                        metric_scores[metric2].append(0.1)
            
            # Calculate overall scores
            metric_overall_scores = {}
            for metric, scores in metric_scores.items():
                if scores:
                    metric_overall_scores[metric] = {
                        'mean_score': np.mean(scores),
                        'total_score': np.sum(scores),
                        'consistency': np.std(scores) if len(scores) > 1 else 0
                    }
            
            if metric_overall_scores:
                # Best metric is the one with highest mean score
                best_metric = max(metric_overall_scores, 
                                key=lambda x: metric_overall_scores[x]['mean_score'])
                
                best_metrics[atlas] = {
                    'best_metric': best_metric,
                    'score': metric_overall_scores[best_metric]['mean_score'],
                    'all_scores': metric_overall_scores
                }
        
        return best_metrics
    
    def _generate_recommendations(self, results: Dict) -> Dict:
        """Generate recommendations based on statistical analysis."""
        recommendations = {
            'overall_best_metrics': [],
            'atlas_specific': {},
            'target_metric_specific': {},
            'guidelines': []
        }
        
        # Overall best metrics across all atlases
        if results.get('best_metrics_per_atlas'):
            metric_votes = {}
            for atlas_info in results['best_metrics_per_atlas'].values():
                metric = atlas_info['best_metric']
                score = atlas_info['score']
                if metric not in metric_votes:
                    metric_votes[metric] = []
                metric_votes[metric].append(score)
            
            # Rank by average performance
            metric_rankings = []
            for metric, scores in metric_votes.items():
                avg_score = np.mean(scores)
                consistency = 1 / (1 + np.std(scores))  # Higher is more consistent
                metric_rankings.append({
                    'metric': metric,
                    'avg_score': avg_score,
                    'consistency': consistency,
                    'n_atlases': len(scores)
                })
            
            metric_rankings.sort(key=lambda x: x['avg_score'], reverse=True)
            recommendations['overall_best_metrics'] = metric_rankings
        
        # Generate guidelines
        summary = results.get('summary_stats', {})
        if summary:
            guidelines = []
            
            if summary.get('significant_corrected', 0) > 0:
                guidelines.append(
                    f"Found {summary['significant_corrected']} significant differences "
                    f"after multiple comparison correction"
                )
            
            if summary.get('large_effects', 0) > 0:
                guidelines.append(
                    f"{summary['large_effects']} comparisons showed large effect sizes (>0.8), "
                    "indicating practically meaningful differences"
                )
            
            if summary.get('mean_effect_size', 0) > 0.5:
                guidelines.append(
                    "Overall medium-to-large effect sizes suggest connectivity metrics "
                    "produce meaningfully different network properties"
                )
            else:
                guidelines.append(
                    "Small overall effect sizes suggest connectivity metrics are "
                    "relatively similar in their network properties"
                )
            
            recommendations['guidelines'] = guidelines
        
        return recommendations
    
    def create_comparison_plots(self, results: Dict, output_dir: str) -> List[str]:
        """Create visualization plots for metric comparisons."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_files = []
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Summary statistics plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract all comparisons
        all_comps = []
        for atlas_result in results.get('comparisons', []):
            for target_metric, comps in atlas_result['metric_comparisons'].items():
                for comp in comps:
                    comp_data = comp.copy()
                    comp_data['comparison'] = f"{comp['metric1']} vs {comp['metric2']}"
                    all_comps.append(comp_data)
        
        if all_comps:
            df_comps = pd.DataFrame(all_comps)
            
            # Plot 1: Effect size distribution
            axes[0, 0].hist(df_comps['cohens_d'].abs(), bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(0.2, color='green', linestyle='--', label='Small (0.2)')
            axes[0, 0].axvline(0.5, color='orange', linestyle='--', label='Medium (0.5)')
            axes[0, 0].axvline(0.8, color='red', linestyle='--', label='Large (0.8)')
            axes[0, 0].set_xlabel('|Cohen\'s d|')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Effect Sizes')
            axes[0, 0].legend()
            
            # Plot 2: Significance by comparison type
            if 'comparison' in df_comps.columns:
                comp_sig = df_comps.groupby('comparison')['p_value'].apply(lambda x: (x < 0.05).mean())
                comp_sig.plot(kind='bar', ax=axes[0, 1], alpha=0.7)
                axes[0, 1].set_xlabel('Comparison Type')
                axes[0, 1].set_ylabel('Proportion Significant')
                axes[0, 1].set_title('Significance Rate by Comparison')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Target metric performance
            if 'target_metric' in df_comps.columns:
                target_sig = df_comps.groupby('target_metric')['p_value'].apply(lambda x: (x < 0.05).mean())
                target_sig.plot(kind='bar', ax=axes[1, 0], alpha=0.7)
                axes[1, 0].set_xlabel('Target Network Metric')
                axes[1, 0].set_ylabel('Proportion Significant')
                axes[1, 0].set_title('Discriminative Power by Network Metric')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot 4: Atlas performance
            if 'atlas' in df_comps.columns:
                atlas_sig = df_comps.groupby('atlas')['p_value'].apply(lambda x: (x < 0.05).mean())
                atlas_sig.plot(kind='bar', ax=axes[1, 1], alpha=0.7)
                axes[1, 1].set_xlabel('Atlas')
                axes[1, 1].set_ylabel('Proportion Significant')
                axes[1, 1].set_title('Sensitivity by Atlas')
                axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_file = output_path / 'statistical_comparison_summary.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(plot_file))
        
        # 2. Best metrics per atlas
        if results.get('best_metrics_per_atlas'):
            plt.figure(figsize=(12, 8))
            
            atlases = []
            best_metrics = []
            scores = []
            
            for atlas, info in results['best_metrics_per_atlas'].items():
                atlases.append(atlas)
                best_metrics.append(info['best_metric'])
                scores.append(info['score'])
            
            # Create grouped bar plot
            unique_metrics = list(set(best_metrics))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_metrics)))
            metric_colors = {metric: colors[i] for i, metric in enumerate(unique_metrics)}
            
            bar_colors = [metric_colors[metric] for metric in best_metrics]
            
            bars = plt.bar(range(len(atlases)), scores, color=bar_colors, alpha=0.8, edgecolor='black')
            plt.xlabel('Atlas')
            plt.ylabel('Performance Score')
            plt.title('Best Connectivity Metric per Atlas (Statistical Performance)')
            plt.xticks(range(len(atlases)), atlases, rotation=45)
            
            # Add metric labels on bars
            for i, (bar, metric) in enumerate(zip(bars, best_metrics)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        metric, ha='center', va='bottom', rotation=90, fontsize=8)
            
            # Add legend
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=metric_colors[metric], 
                                           label=metric) for metric in unique_metrics]
            plt.legend(handles=legend_elements, title='Best Metric', 
                      bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plot_file = output_path / 'best_metrics_statistical.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(str(plot_file))
        
        logger.info(f"📊 Created {len(plot_files)} statistical comparison plots")
        return plot_files
    
    def generate_report(self, results: Dict, output_file: str) -> None:
        """Generate a comprehensive statistical comparison report."""
        
        report_content = f"""
Statistical Comparison of Connectivity Metrics
==============================================

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
-----------------
This analysis compares different connectivity metrics (count, ncount2, fa, qa) 
to determine which produce significantly different brain network properties.

"""
        
        # Summary statistics
        summary = results.get('summary_stats', {})
        if summary:
            report_content += f"""
SUMMARY STATISTICS
-----------------
Total statistical comparisons: {summary.get('total_comparisons', 0)}
Significant (uncorrected p<0.05): {summary.get('significant_uncorrected', 0)}
Significant (FDR corrected p<0.05): {summary.get('significant_corrected', 0)}
Correction rate: {summary.get('significant_corrected', 0) / max(summary.get('significant_uncorrected', 1), 1):.2%}

EFFECT SIZES
-----------
Mean |Cohen's d|: {summary.get('mean_effect_size', 0):.3f}
Median |Cohen's d|: {summary.get('median_effect_size', 0):.3f}
Large effects (|d|>0.8): {summary.get('large_effects', 0)}
Medium effects (0.5<|d|≤0.8): {summary.get('medium_effects', 0)}
Small effects (0.2<|d|≤0.5): {summary.get('small_effects', 0)}

TEST DISTRIBUTION
----------------
Parametric tests (t-test): {summary.get('parametric_tests', 0)}
Non-parametric tests (Mann-Whitney): {summary.get('non_parametric_tests', 0)}
"""
        
        # Best metrics per atlas
        if results.get('best_metrics_per_atlas'):
            report_content += """
BEST CONNECTIVITY METRIC PER ATLAS
----------------------------------
(Based on statistical performance across network metrics)

"""
            for atlas, info in results['best_metrics_per_atlas'].items():
                report_content += f"{atlas:30s}: {info['best_metric']:8s} (score: {info['score']:6.3f})\n"
        
        # Overall recommendations
        if results.get('recommendations', {}).get('overall_best_metrics'):
            report_content += """

OVERALL METRIC RANKINGS
-----------------------
(Ranked by average performance across all atlases)

"""
            for rank, metric_info in enumerate(results['recommendations']['overall_best_metrics'], 1):
                report_content += f"{rank}. {metric_info['metric']:8s} - Score: {metric_info['avg_score']:.3f} " \
                                f"(consistency: {metric_info['consistency']:.3f}, atlases: {metric_info['n_atlases']})\n"
        
        # Guidelines
        if results.get('recommendations', {}).get('guidelines'):
            report_content += """

KEY FINDINGS
-----------
"""
            for guideline in results['recommendations']['guidelines']:
                report_content += f"• {guideline}\n"
        
        # Detailed results
        if results.get('comparisons'):
            report_content += """

DETAILED STATISTICAL RESULTS
============================
"""
            
            for atlas_result in results['comparisons']:
                atlas = atlas_result['atlas']
                report_content += f"\n{atlas}\n" + "="*50 + "\n"
                
                for target_metric, comps in atlas_result['metric_comparisons'].items():
                    if not comps:
                        continue
                        
                    report_content += f"\nNetwork Metric: {target_metric}\n" + "-"*30 + "\n"
                    
                    for comp in comps:
                        # Significance markers
                        p_val = comp.get('p_corrected', comp.get('p_value', 1))
                        if p_val < 0.001:
                            sig_marker = "***"
                        elif p_val < 0.01:
                            sig_marker = "**"
                        elif p_val < 0.05:
                            sig_marker = "*"
                        else:
                            sig_marker = "ns"
                        
                        report_content += f"{comp['metric1']} vs {comp['metric2']}:\n"
                        report_content += f"  Means: {comp['mean1']:.4f} vs {comp['mean2']:.4f}\n"
                        report_content += f"  Test: {comp['test_used']} | p = {comp.get('p_value', 0):.4f} {sig_marker}\n"
                        report_content += f"  Effect size: d = {comp.get('cohens_d', 0):.3f} ({comp.get('effect_size_category', 'unknown')})\n"
                        report_content += f"  Sample sizes: n1={comp['n1']}, n2={comp['n2']}\n\n"
        
        report_content += f"""

METHODOLOGY
===========
Statistical Tests:
- Normality: Shapiro-Wilk test
- Equal variances: Levene's test  
- Parametric: Independent t-test (if assumptions met)
- Non-parametric: Mann-Whitney U test (robust alternative)

Multiple Comparison Correction:
- Method: {self.correction_method.upper()}
- Controls False Discovery Rate (FDR)
- More powerful than Bonferroni correction

Effect Size Interpretation:
- Cohen's d: Standardized mean difference
- Small: |d| = 0.2, Medium: |d| = 0.5, Large: |d| = 0.8
- Practical significance threshold: |d| ≥ 0.5

RECOMMENDATIONS FOR YOUR STUDY
=============================
🏈 Sport vs. Control Group Analysis:

1. METRIC SELECTION
   - Use the best-performing metric for each atlas
   - Focus on combinations with large effect sizes
   - Consider reliability across different network metrics

2. STATISTICAL APPROACH
   - Apply the same statistical methods used here
   - Include both parametric and non-parametric options
   - Always correct for multiple comparisons

3. INTERPRETATION
   - Report both p-values and effect sizes
   - Effect sizes more important than significance for practical meaning
   - Consider minimum effect size for biological relevance

4. NEXT STEPS
   - Validate findings with cross-validation
   - Consider mixed-effects models for subject-level analysis
   - Explore relationship with training/performance variables

SIGNIFICANCE LEVELS
==================
*** p < 0.001 (highly significant)
**  p < 0.01  (very significant)
*   p < 0.05  (significant)  
ns  p ≥ 0.05  (not significant)
"""
        
        # Write report
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"📋 Generated statistical comparison report: {output_file}")


def main():
    """Command line interface for connectivity metric comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Statistical comparison of connectivity metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.csv output/ --metrics count,ncount2,fa,qa
  %(prog)s data.csv output/ --atlases FreeSurferDKT_Subcortical,HCP-MMP --plots
        """
    )
    
    parser.add_argument("input_file", help="CSV file with graph metrics")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--metrics", help="Comma-separated connectivity metrics to compare")
    parser.add_argument("--atlases", help="Comma-separated atlas names to include")
    parser.add_argument("--target-metrics", help="Comma-separated network metrics to analyze")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--correction", default="fdr_bh", 
                       choices=["fdr_bh", "bonferroni", "holm"], 
                       help="Multiple comparison correction method")
    parser.add_argument("--plots", action="store_true", help="Generate comparison plots")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(args.output_dir) / 'comparison.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Load data
        df = pd.read_csv(args.input_file)
        logger.info(f"📊 Loaded {len(df)} records from {args.input_file}")
        
        # Validate required columns
        required_cols = ['subject', 'atlas', 'connectivity_metric']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"❌ Missing required columns: {missing_cols}")
            return 1
        
        # Parse optional arguments
        metrics_to_compare = args.metrics.split(',') if args.metrics else None
        atlas_filter = args.atlases.split(',') if args.atlases else None
        target_metrics = args.target_metrics.split(',') if args.target_metrics else None
        
        # Initialize comparator
        comparator = ConnectivityMetricComparator(
            alpha=args.alpha, 
            correction_method=args.correction
        )
        
        # Perform comparison
        results = comparator.compare_metrics(
            df, 
            atlas_filter=atlas_filter,
            metrics_to_compare=metrics_to_compare,
            target_metrics=target_metrics
        )
        
        # Save results
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate report
        report_file = output_path / "statistical_comparison_report.txt"
        comparator.generate_report(results, str(report_file))
        
        # Generate plots if requested
        if args.plots:
            plot_files = comparator.create_comparison_plots(results, str(output_path))
            logger.info(f"📈 Generated {len(plot_files)} plots")
        
        # Print summary
        summary = results.get('summary_stats', {})
        recommendations = results.get('recommendations', {})
        
        print(f"\n🎯 Statistical Comparison Summary:")
        print(f"{'='*50}")
        print(f"Total comparisons: {summary.get('total_comparisons', 0)}")
        print(f"Significant (corrected): {summary.get('significant_corrected', 0)}")
        print(f"Mean effect size: {summary.get('mean_effect_size', 0):.3f}")
        print(f"Large effects: {summary.get('large_effects', 0)}")
        
        if recommendations.get('overall_best_metrics'):
            print(f"\n🏆 Top connectivity metrics:")
            for i, metric in enumerate(recommendations['overall_best_metrics'][:3], 1):
                print(f"  {i}. {metric['metric']} (score: {metric['avg_score']:.3f})")
        
        print(f"\n📁 Results saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
