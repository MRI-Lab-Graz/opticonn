#!/usr/bin/env python3
"""
Metric Optimizer Module
=======================

Connectivity metric optimization based on network quality properties.
Identifies the most promising atlas/metric combinations for analysis.

Author: Braingraph Pipeline Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import logging
from pathlib import Path
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.stats import shapiro, levene
from statsmodels.stats.multitest import multipletests
import warnings

from scripts.utils.runtime import configure_stdio

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class MetricOptimizer:
    """
    Connectivity metric optimization based on network quality properties.

    This class evaluates different atlas/metric combinations based on:
    - Sparsity levels (optimal range for analysis)
    - Small-world properties
    - Modularity
    - Global efficiency
    - Network reliability and consistency
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the metric optimizer.

        Args:
            config: Configuration dictionary with optimization parameters
        """
        self.config = config or self._default_config()

        # Extract configuration parameters
        self.weight_factors = self.config.get("weight_factors", {})
        self.sparsity_range = self.config.get("sparsity_range", [0.05, 0.4])
        self.quality_threshold = self.config.get("quality_threshold", 0.7)
        self.metrics_to_evaluate = self.config.get("metrics_to_evaluate", [])

        # Normalization parameters (will be computed from data)
        self.normalization_params = {}

    def _default_config(self) -> Dict:
        """Default optimization configuration."""
        return {
            "metrics_to_evaluate": [
                "sparsity",
                "small_worldness",
                "modularity",
                "global_efficiency",
                "characteristic_path_length",
                "clustering_coefficient",
                "assortativity",
            ],
            "weight_factors": {
                "sparsity_score": 0.25,  # Optimal sparsity range
                "small_worldness_score": 0.25,  # Small-world coefficient
                "modularity_score": 0.20,  # Community structure
                "efficiency_score": 0.20,  # Network efficiency
                "reliability_score": 0.10,  # Cross-subject consistency
            },
            "sparsity_range": [0.05, 0.4],  # Optimal sparsity range
            "small_world_range": [0.0, 0.5],  # Adjusted for actual data range
            "quality_threshold": 0.65,  # Minimum quality score
            "reliability_threshold": 0.7,  # Minimum reliability score
        }

    def compute_sparsity_score(self, sparsity_values: np.ndarray) -> np.ndarray:
        """
        Compute sparsity quality score.

        Networks should be neither too sparse nor too dense for optimal analysis.

        Args:
            sparsity_values: Array of sparsity values

        Returns:
            Sparsity quality scores (0-1, higher is better)
        """
        min_sparsity, max_sparsity = self.sparsity_range

        # Optimal sparsity is in the middle of the range
        optimal_sparsity = (min_sparsity + max_sparsity) / 2
        sparsity_width = max_sparsity - min_sparsity

        # Score based on distance from optimal sparsity
        distance_from_optimal = np.abs(sparsity_values - optimal_sparsity)
        normalized_distance = distance_from_optimal / (sparsity_width / 2)

        # Convert to quality score (closer to optimal = higher score)
        scores = np.maximum(0, 1 - normalized_distance)

        # Penalty for values outside acceptable range
        out_of_range = (sparsity_values < min_sparsity) | (
            sparsity_values > max_sparsity
        )
        scores[out_of_range] *= 0.1  # Heavy penalty for out-of-range values

        return scores

    def compute_small_world_score(self, small_world_values: np.ndarray) -> np.ndarray:
        """
        Compute small-worldness quality score.

        Good small-world networks have sigma > 1 (ideally 1-3).

        Args:
            small_world_values: Array of small-worldness values

        Returns:
            Small-world quality scores (0-1, higher is better)
        """
        # Handle NaN values
        valid_mask = ~np.isnan(small_world_values)
        scores = np.zeros_like(small_world_values)

        if np.any(valid_mask):
            valid_values = small_world_values[valid_mask]

            # Use absolute target range for comparability across combos
            min_sw, max_sw = self.config.get("small_world_range", [0.0, 0.5])
            # Map values to 0..1 using configured range (clipped)
            denom = (max_sw - min_sw) if (max_sw - min_sw) != 0 else 1.0
            valid_scores = (valid_values - min_sw) / denom
            # Slight bonus for values in upper half of configured range
            mid_sw = (min_sw + max_sw) / 2.0
            with np.errstate(invalid="ignore"):
                bonus_mask = valid_values > mid_sw
            valid_scores[bonus_mask] *= 1.1
            valid_scores = np.clip(valid_scores, 0.0, 1.0)

            scores[valid_mask] = valid_scores

        return scores

    def compute_modularity_score(self, modularity_values: np.ndarray) -> np.ndarray:
        """
        Compute modularity quality score.

        Higher modularity indicates better community structure.

        Args:
            modularity_values: Array of modularity values

        Returns:
            Modularity quality scores (0-1, higher is better)
        """
        # Handle NaN values
        valid_mask = ~np.isnan(modularity_values)
        scores = np.zeros_like(modularity_values)

        if np.any(valid_mask):
            valid_values = modularity_values[valid_mask]

            # Modularity typically ranges from 0 to ~0.7
            # Higher values indicate better community structure
            normalized_values = np.clip(valid_values / 0.7, 0, 1)
            scores[valid_mask] = normalized_values

        return scores

    def compute_efficiency_score(self, efficiency_values: np.ndarray) -> np.ndarray:
        """
        Compute global efficiency quality score.

        Higher efficiency indicates better network integration.

        Args:
            efficiency_values: Array of global efficiency values

        Returns:
            Efficiency quality scores (0-1, higher is better)
        """
        # Handle NaN values
        valid_mask = ~np.isnan(efficiency_values)
        scores = np.zeros_like(efficiency_values)

        if np.any(valid_mask):
            valid_values = efficiency_values[valid_mask]

            # Global efficiency typically ranges from 0 to 1
            scores[valid_mask] = np.clip(valid_values, 0, 1)

        return scores

    def compute_reliability_score(
        self, df: pd.DataFrame, groupby_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Compute reliability scores based on cross-subject consistency.

        Args:
            df: DataFrame with metrics
            groupby_cols: Columns to group by (e.g., ['atlas', 'connectivity_metric'])

        Returns:
            DataFrame with reliability scores added
        """
        if groupby_cols is None:
            groupby_cols = ["atlas", "connectivity_metric"]

        # Metrics to evaluate for reliability
        reliability_metrics = [
            "global_efficiency(binary)",
            "global_efficiency(weighted)",
            "small_worldness(binary)",
            "small_worldness(weighted)",
            "clustering_coeff_average(binary)",
            "clustering_coeff_average(weighted)",
            "network_characteristic_path_length(binary)",
            "network_characteristic_path_length(weighted)",
        ]

        # Compute coefficient of variation (CV) for each group
        reliability_scores = []

        for name, group in df.groupby(groupby_cols):
            scores = {}
            scores.update(dict(zip(groupby_cols, name)))

            # Compute reliability for each metric
            metric_reliabilities = []

            for metric in reliability_metrics:
                if metric in group.columns and len(group) > 1:
                    values = group[metric].dropna()
                    if len(values) > 1 and np.std(values) > 0:
                        # Coefficient of variation (lower = more reliable)
                        cv = (
                            np.std(values) / np.abs(np.mean(values))
                            if np.mean(values) != 0
                            else np.inf
                        )
                        # Convert to reliability score (higher = more reliable)
                        reliability = np.exp(-cv)  # Exponential decay of CV
                        metric_reliabilities.append(reliability)

            # Overall reliability is mean of individual metric reliabilities
            if metric_reliabilities:
                scores["reliability_score"] = np.mean(metric_reliabilities)
            else:
                scores["reliability_score"] = 0.0

            reliability_scores.append(scores)

        reliability_df = pd.DataFrame(reliability_scores)

        # Merge reliability scores back to original dataframe
        df_with_reliability = df.merge(reliability_df, on=groupby_cols, how="left")

        return df_with_reliability

    def compute_quality_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute overall quality scores for each atlas/metric combination.

        Args:
            df: DataFrame with graph metrics

        Returns:
            DataFrame with quality scores added
        """
        logger.info("Computing quality scores for atlas/metric combinations...")

        # Add reliability scores
        df = self.compute_reliability_score(df)

        # Compute component scores from raw network measures
        quality_components = {}

        # Sparsity score (use density as proxy for sparsity)
        if "density" in df.columns:
            # Density is the opposite of sparsity, so invert it
            density_values = df["density"].values
            sparsity_values = 1 - density_values  # Convert density to sparsity
            quality_components["sparsity_score"] = self.compute_sparsity_score(
                sparsity_values
            )
        else:
            quality_components["sparsity_score"] = np.zeros(len(df))

        # Small-worldness score
        if "small-worldness(weighted)" in df.columns:
            sw_values = df["small-worldness(weighted)"].values
            quality_components["small_worldness_score"] = (
                self.compute_small_world_score(sw_values)
            )
        elif "small-worldness(binary)" in df.columns:
            sw_values = df["small-worldness(binary)"].values
            quality_components["small_worldness_score"] = (
                self.compute_small_world_score(sw_values)
            )
        else:
            quality_components["small_worldness_score"] = np.zeros(len(df))

        # Modularity score (use clustering coefficient as proxy)
        if "clustering_coeff_average(weighted)" in df.columns:
            mod_values = df["clustering_coeff_average(weighted)"].values
            quality_components["modularity_score"] = self.compute_modularity_score(
                mod_values
            )
        elif "clustering_coeff_average(binary)" in df.columns:
            mod_values = df["clustering_coeff_average(binary)"].values
            quality_components["modularity_score"] = self.compute_modularity_score(
                mod_values
            )
        else:
            quality_components["modularity_score"] = np.zeros(len(df))

        # Efficiency score
        if "global_efficiency(weighted)" in df.columns:
            eff_values = df["global_efficiency(weighted)"].values
            quality_components["efficiency_score"] = self.compute_efficiency_score(
                eff_values
            )
        elif "global_efficiency(binary)" in df.columns:
            eff_values = df["global_efficiency(binary)"].values
            quality_components["efficiency_score"] = self.compute_efficiency_score(
                eff_values
            )
        else:
            quality_components["efficiency_score"] = np.zeros(len(df))

        # Reliability score (already computed above)
        if "reliability_score" not in df.columns:
            quality_components["reliability_score"] = np.zeros(len(df))
        else:
            quality_components["reliability_score"] = df["reliability_score"].values

        # Add component scores to dataframe
        for component, scores in quality_components.items():
            df[component] = scores

        # Compute weighted overall quality score
        weights = self.weight_factors
        # Start from zero and accumulate weighted component scores (components are already 0-1 scaled)
        quality_score_raw = np.zeros(len(df))

        for component, weight in weights.items():
            if component in df.columns:
                component_values = df[component].values
                # Ensure component is within [0,1]
                component_values = np.clip(component_values, 0.0, 1.0)
                quality_score_raw += weight * component_values

        # Persist raw (absolute-scale) and normalized scores
        df["quality_score_raw"] = quality_score_raw
        quality_score = quality_score_raw.copy()
        if np.max(quality_score) > np.min(quality_score):
            quality_score = (quality_score - np.min(quality_score)) / (
                np.max(quality_score) - np.min(quality_score)
            )
        else:
            quality_score = np.ones_like(quality_score)
        df["quality_score"] = quality_score

        return df

    def optimize_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize connectivity metrics based on quality scores.

        Args:
            df: DataFrame with graph metrics

        Returns:
            DataFrame with quality scores and optimization results
        """
        logger.info("Optimizing connectivity metrics...")

        # Compute quality scores
        df_with_scores = self.compute_quality_scores(df)

        # Filter high-quality combinations
        high_quality = df_with_scores[
            df_with_scores["quality_score"] >= self.quality_threshold
        ]

        if len(high_quality) == 0:
            logger.warning(
                f"No combinations meet quality threshold {self.quality_threshold}"
            )
            logger.info(f"Lowering threshold to include top 10% of combinations")
            threshold = np.percentile(df_with_scores["quality_score"], 90)
            high_quality = df_with_scores[df_with_scores["quality_score"] >= threshold]

        # Add optimization flags
        df_with_scores["meets_quality_threshold"] = (
            df_with_scores["quality_score"] >= self.quality_threshold
        )
        df_with_scores["recommended"] = False

        # Mark recommended combinations (top scoring in each atlas)
        for atlas in df_with_scores["atlas"].unique():
            atlas_data = df_with_scores[df_with_scores["atlas"] == atlas]
            if len(atlas_data) > 0:
                best_idx = atlas_data["quality_score"].idxmax()
                df_with_scores.loc[best_idx, "recommended"] = True

        # Sort by quality score
        df_with_scores = df_with_scores.sort_values("quality_score", ascending=False)

        logger.info(
            f"Optimization complete. {len(high_quality)} high-quality combinations found."
        )

        return df_with_scores

    def generate_optimization_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the optimization.

        Args:
            df: DataFrame with optimization results

        Returns:
            Dictionary with summary statistics
        """
        summary = {}

        # Overall statistics
        summary["total_combinations"] = len(df)
        summary["high_quality_combinations"] = len(df[df["meets_quality_threshold"]])
        summary["recommended_combinations"] = len(df[df["recommended"]])
        summary["mean_quality_score"] = float(df["quality_score"].mean())
        summary["max_quality_score"] = float(df["quality_score"].max())
        if "quality_score_raw" in df.columns:
            summary["mean_quality_score_raw"] = float(df["quality_score_raw"].mean())
            summary["max_quality_score_raw"] = float(df["quality_score_raw"].max())

        # Best combinations
        top_10 = df.nlargest(10, "quality_score")[
            ["atlas", "connectivity_metric", "quality_score"]
        ]
        summary["top_10_combinations"] = top_10.to_dict("records")

        # Atlas-wise best
        atlas_best = df.loc[df.groupby("atlas")["quality_score"].idxmax()][
            ["atlas", "connectivity_metric", "quality_score"]
        ]
        summary["best_per_atlas"] = atlas_best.to_dict("records")

        # Metric-wise best
        metric_best = df.loc[
            df.groupby("connectivity_metric")["quality_score"].idxmax()
        ][["atlas", "connectivity_metric", "quality_score"]]
        summary["best_per_metric"] = metric_best.to_dict("records")

        # Quality component analysis
        component_cols = [
            col
            for col in df.columns
            if col.endswith("_score") and col != "quality_score"
        ]
        if component_cols:
            summary["component_correlations"] = {}
            for component in component_cols:
                corr = df[component].corr(df["quality_score"])
                summary["component_correlations"][component] = (
                    float(corr) if not np.isnan(corr) else 0.0
                )

        return summary

    def generate_report(self, df: pd.DataFrame, output_file: str) -> None:
        """
        Generate a detailed optimization report.

        Args:
            df: DataFrame with optimization results
            output_file: Path to output report file
        """
        summary = self.generate_optimization_summary(df)

        # Prepare optional raw score lines
        raw_lines = ""
        if "mean_quality_score_raw" in summary:
            raw_lines += (
                f"Mean quality score (raw): {summary['mean_quality_score_raw']:.3f}\n"
            )
        if "max_quality_score_raw" in summary:
            raw_lines += (
                f"Maximum quality score (raw): {summary['max_quality_score_raw']:.3f}\n"
            )

        report_content = f"""
Connectivity Metric Optimization Report
======================================

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS
-----------------
Total atlas/metric combinations: {summary['total_combinations']}
High-quality combinations: {summary['high_quality_combinations']}
Recommended combinations: {summary['recommended_combinations']}
Mean quality score (normalized): {summary['mean_quality_score']:.3f}
Maximum quality score (normalized): {summary['max_quality_score']:.3f}
{raw_lines}

TOP 10 COMBINATIONS
------------------
"""

        for i, combo in enumerate(summary["top_10_combinations"], 1):
            atlas = str(combo["atlas"])
            metric = str(combo["connectivity_metric"])
            score = combo["quality_score"]
            report_content += (
                f"{i:2d}. {atlas:15s} + {metric:10s} (score: {score:.3f})\n"
            )

        report_content += f"""

BEST COMBINATION PER ATLAS
--------------------------
"""
        for combo in summary["best_per_atlas"]:
            atlas = str(combo["atlas"])
            metric = str(combo["connectivity_metric"])
            score = combo["quality_score"]
            report_content += f"{atlas:15s}: {metric:10s} (score: {score:.3f})\n"

        report_content += f"""

BEST COMBINATION PER METRIC
---------------------------
"""
        for combo in summary["best_per_metric"]:
            metric = str(combo["connectivity_metric"])
            atlas = str(combo["atlas"])
            score = combo["quality_score"]
            report_content += f"{metric:10s}: {atlas:15s} (score: {score:.3f})\n"

        if "component_correlations" in summary:
            report_content += f"""

QUALITY COMPONENT CORRELATIONS
------------------------------
"""
            for component, corr in summary["component_correlations"].items():
                component_name = str(component)
                report_content += f"{component_name:20s}: {corr:6.3f}\n"

        report_content += f"""

OPTIMIZATION CONFIGURATION
--------------------------
Weight factors:
"""
        for factor, weight in self.weight_factors.items():
            factor_name = str(factor)
            report_content += f"  {factor_name:20s}: {weight:.2f}\n"

        report_content += f"""
Sparsity range: {self.sparsity_range[0]:.2f} - {self.sparsity_range[1]:.2f}
Quality threshold: {self.quality_threshold:.2f}

RECOMMENDATIONS
--------------
1. Use combinations marked as 'recommended' for main analysis
2. Consider combinations with quality_score > {self.quality_threshold:.2f}
3. Validate results with cross-validation on independent data
4. Check individual quality components for specific requirements

NOTES
-----
- Higher quality scores indicate better network properties for analysis
- Sparsity scores favor networks in the optimal density range
- Small-worldness scores favor networks with good small-world properties
- Modularity scores favor networks with clear community structure
- Efficiency scores favor networks with good global integration
- Reliability scores favor consistent results across subjects
"""

        # Write report
        with open(output_file, "w") as f:
            f.write(report_content)

        logger.info(f"Generated optimization report: {output_file}")

    def create_optimization_plots(self, df: pd.DataFrame, output_dir: str) -> List[str]:
        """
        Create visualization plots for optimization results.

        Args:
            df: DataFrame with optimization results
            output_dir: Directory to save plots

        Returns:
            List of created plot files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        plot_files = []

        # Set style
        plt.style.use("default")
        sns.set_palette("husl")

        # 1. Quality score distribution
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.hist(df["quality_score"], bins=30, alpha=0.7, edgecolor="black")
        plt.axvline(
            self.quality_threshold,
            color="red",
            linestyle="--",
            label=f"Threshold ({self.quality_threshold})",
        )
        plt.xlabel("Quality Score")
        plt.ylabel("Frequency")
        plt.title("Distribution of Quality Scores")
        plt.legend()

        # 2. Quality scores by atlas
        plt.subplot(2, 2, 2)
        atlas_order = (
            df.groupby("atlas")["quality_score"]
            .mean()
            .sort_values(ascending=False)
            .index
        )
        sns.boxplot(data=df, x="atlas", y="quality_score", order=atlas_order)
        plt.xticks(rotation=45)
        plt.ylabel("Quality Score")
        plt.title("Quality Scores by Atlas")

        # 3. Quality scores by connectivity metric
        plt.subplot(2, 2, 3)
        metric_order = (
            df.groupby("connectivity_metric")["quality_score"]
            .mean()
            .sort_values(ascending=False)
            .index
        )
        sns.boxplot(
            data=df, x="connectivity_metric", y="quality_score", order=metric_order
        )
        plt.xticks(rotation=45)
        plt.ylabel("Quality Score")
        plt.title("Quality Scores by Connectivity Metric")

        # 4. Component contributions
        plt.subplot(2, 2, 4)
        component_cols = [
            col
            for col in df.columns
            if col.endswith("_score") and col != "quality_score"
        ]
        if component_cols:
            component_means = df[component_cols].mean()
            component_means.plot(kind="bar")
            plt.xticks(rotation=45)
            plt.ylabel("Mean Score")
            plt.title("Quality Component Contributions")

        plt.tight_layout()
        plot_file = output_path / "optimization_overview.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        plot_files.append(str(plot_file))

        # 2. Heatmap of quality scores
        plt.figure(figsize=(14, 10))

        # Create pivot table for heatmap
        pivot_data = df.pivot_table(
            values="quality_score",
            index="atlas",
            columns="connectivity_metric",
            aggfunc="mean",
        )

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            cbar_kws={"label": "Quality Score"},
        )
        plt.title("Quality Scores: Atlas × Connectivity Metric")
        plt.xlabel("Connectivity Metric")
        plt.ylabel("Atlas")

        plot_file = output_path / "quality_heatmap.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        plot_files.append(str(plot_file))

        # 3. Component correlation plot
        component_cols = [col for col in df.columns if col.endswith("_score")]
        if len(component_cols) > 1:
            plt.figure(figsize=(10, 8))

            corr_matrix = df[component_cols].corr()
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".3f",
                cmap="coolwarm",
                center=0,
                cbar_kws={"label": "Correlation"},
            )
            plt.title("Quality Component Correlations")

            plot_file = output_path / "component_correlations.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            plot_files.append(str(plot_file))

        # 4. Scatter plot of key metrics
        if "sparsity" in df.columns and "small_worldness" in df.columns:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            scatter = plt.scatter(
                df["sparsity"],
                df["small_worldness"],
                c=df["quality_score"],
                cmap="viridis",
                alpha=0.7,
            )
            plt.colorbar(scatter, label="Quality Score")
            plt.xlabel("Sparsity")
            plt.ylabel("Small-worldness")
            plt.title("Sparsity vs Small-worldness")

            plt.subplot(1, 2, 2)
            if "modularity" in df.columns and "global_efficiency" in df.columns:
                scatter = plt.scatter(
                    df["modularity"],
                    df["global_efficiency"],
                    c=df["quality_score"],
                    cmap="viridis",
                    alpha=0.7,
                )
                plt.colorbar(scatter, label="Quality Score")
                plt.xlabel("Modularity")
                plt.ylabel("Global Efficiency")
                plt.title("Modularity vs Global Efficiency")

            plt.tight_layout()
            plot_file = output_path / "metric_relationships.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            plot_files.append(str(plot_file))

        logger.info(f"Created {len(plot_files)} optimization plots in {output_dir}")
        return plot_files

    def _perform_pairwise_comparisons(
        self, metric_data: Dict[str, np.ndarray], target_metric: str, atlas: str
    ) -> Dict:
        """Perform pairwise statistical comparisons between connectivity metrics."""
        comparisons = []
        metric_names = list(metric_data.keys())

        for i in range(len(metric_names)):
            for j in range(i + 1, len(metric_names)):
                metric1, metric2 = metric_names[i], metric_names[j]
                data1, data2 = metric_data[metric1], metric_data[metric2]

                # Perform statistical tests
                comparison = {
                    "metric1": metric1,
                    "metric2": metric2,
                    "n1": len(data1),
                    "n2": len(data2),
                    "mean1": np.mean(data1),
                    "mean2": np.mean(data2),
                    "std1": np.std(data1),
                    "std2": np.std(data2),
                }

                # Test for normality
                if len(data1) >= 3:
                    _, p_norm1 = shapiro(data1)
                    comparison["normality_p1"] = p_norm1
                else:
                    comparison["normality_p1"] = np.nan

                if len(data2) >= 3:
                    _, p_norm2 = shapiro(data2)
                    comparison["normality_p2"] = p_norm2
                else:
                    comparison["normality_p2"] = np.nan

                # Test for equal variances
                if len(data1) >= 2 and len(data2) >= 2:
                    _, p_levene = levene(data1, data2)
                    comparison["equal_var_p"] = p_levene
                else:
                    comparison["equal_var_p"] = np.nan

                # Choose appropriate test
                use_parametric = (
                    comparison["normality_p1"] > 0.05
                    and comparison["normality_p2"] > 0.05
                    and comparison["equal_var_p"] > 0.05
                )

                if use_parametric:
                    # t-test for independent samples
                    stat, p_value = ttest_ind(data1, data2, equal_var=True)
                    comparison["test_used"] = "t-test"
                else:
                    # Mann-Whitney U test
                    stat, p_value = mannwhitneyu(data1, data2, alternative="two-sided")
                    comparison["test_used"] = "mann-whitney"

                comparison["statistic"] = stat
                comparison["p_value"] = p_value

                # Effect size (Cohen's d)
                pooled_std = np.sqrt(
                    (
                        (len(data1) - 1) * np.var(data1)
                        + (len(data2) - 1) * np.var(data2)
                    )
                    / (len(data1) + len(data2) - 2)
                )

                if pooled_std > 0:
                    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
                    comparison["cohens_d"] = cohens_d

                    # Effect size interpretation
                    if abs(cohens_d) < 0.2:
                        comparison["effect_size"] = "negligible"
                    elif abs(cohens_d) < 0.5:
                        comparison["effect_size"] = "small"
                    elif abs(cohens_d) < 0.8:
                        comparison["effect_size"] = "medium"
                    else:
                        comparison["effect_size"] = "large"
                else:
                    comparison["cohens_d"] = 0.0
                    comparison["effect_size"] = "negligible"

                comparisons.append(comparison)

        return comparisons

    def _generate_comparison_summary(self, comparisons: List[Dict]) -> Dict:
        """Generate summary statistics for all comparisons."""
        all_comparisons = []

        for atlas_result in comparisons:
            atlas = atlas_result["atlas"]
            for target_metric, comps in atlas_result["metric_comparisons"].items():
                for comp in comps:
                    comp_summary = comp.copy()
                    comp_summary["atlas"] = atlas
                    comp_summary["target_metric"] = target_metric
                    all_comparisons.append(comp_summary)

        if not all_comparisons:
            return {}

        df_comps = pd.DataFrame(all_comparisons)

        # Apply multiple comparison correction
        if "p_value" in df_comps.columns:
            p_values = df_comps["p_value"].dropna()
            if len(p_values) > 0:
                _, p_corrected, _, _ = multipletests(p_values, method="fdr_bh")
                df_comps.loc[df_comps["p_value"].notna(), "p_corrected"] = p_corrected

        summary = {
            "total_comparisons": len(df_comps),
            "significant_uncorrected": (
                len(df_comps[df_comps["p_value"] < 0.05])
                if "p_value" in df_comps.columns
                else 0
            ),
            "significant_corrected": (
                len(df_comps[df_comps["p_corrected"] < 0.05])
                if "p_corrected" in df_comps.columns
                else 0
            ),
            "mean_effect_size": (
                df_comps["cohens_d"].abs().mean()
                if "cohens_d" in df_comps.columns
                else 0
            ),
            "large_effects": (
                len(df_comps[df_comps["effect_size"] == "large"])
                if "effect_size" in df_comps.columns
                else 0
            ),
        }

        return summary

    def _calculate_effect_sizes(
        self, df: pd.DataFrame, conn_metrics: List[str], target_metrics: List[str]
    ) -> Dict:
        """Calculate effect sizes between connectivity metrics."""
        effect_sizes = {}

        for atlas in df["atlas"].unique():
            if atlas == "by_subject":
                continue

            atlas_data = df[df["atlas"] == atlas]
            atlas_effects = {}

            for target_metric in target_metrics:
                if target_metric not in atlas_data.columns:
                    continue

                metric_means = {}
                for conn_metric in conn_metrics:
                    conn_data = atlas_data[
                        atlas_data["connectivity_metric"] == conn_metric
                    ]
                    if len(conn_data) > 0:
                        values = conn_data[target_metric].dropna()
                        if len(values) > 0:
                            metric_means[conn_metric] = np.mean(values)

                if len(metric_means) > 1:
                    atlas_effects[target_metric] = metric_means

            if atlas_effects:
                effect_sizes[atlas] = atlas_effects

        return effect_sizes

    def _identify_best_metrics(self, comparisons: List[Dict]) -> Dict:
        """Identify the best connectivity metrics for each atlas based on statistical properties."""
        best_metrics = {}

        for atlas_result in comparisons:
            atlas = atlas_result["atlas"]
            metric_scores = {}

            for target_metric, comps in atlas_result["metric_comparisons"].items():
                for comp in comps:
                    metric1, metric2 = comp["metric1"], comp["metric2"]

                    # Initialize scores
                    if metric1 not in metric_scores:
                        metric_scores[metric1] = []
                    if metric2 not in metric_scores:
                        metric_scores[metric2] = []

                    # Score based on effect size and significance
                    if comp.get("p_corrected", 1.0) < 0.05:  # Significant difference
                        effect_size = abs(comp.get("cohens_d", 0))

                        # Higher mean gets higher score
                        if comp["mean1"] > comp["mean2"]:
                            metric_scores[metric1].append(effect_size)
                            metric_scores[metric2].append(-effect_size)
                        else:
                            metric_scores[metric2].append(effect_size)
                            metric_scores[metric1].append(-effect_size)
                    else:
                        # No significant difference
                        metric_scores[metric1].append(0)
                        metric_scores[metric2].append(0)

            # Calculate overall scores
            metric_overall_scores = {}
            for metric, scores in metric_scores.items():
                if scores:
                    metric_overall_scores[metric] = np.mean(scores)

            if metric_overall_scores:
                best_metric = max(metric_overall_scores, key=metric_overall_scores.get)
                best_metrics[atlas] = {
                    "best_metric": best_metric,
                    "score": metric_overall_scores[best_metric],
                    "all_scores": metric_overall_scores,
                }

        return best_metrics


def main():
    """Command line interface for metric optimization."""
    import argparse

    parser = argparse.ArgumentParser(description="Connectivity metric optimization")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform a dry-run: process metadata and show expected outputs without writing",
    )
    # Print help when called with no args
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    # Positional arguments (backward compatible)
    parser.add_argument("input_file", nargs="?", help="CSV file with graph metrics")
    parser.add_argument("output_dir", nargs="?", help="Output directory for results")
    # Optional aliases for consistency
    parser.add_argument(
        "-i", "--input", dest="input_opt", help="Alias for input CSV file"
    )
    parser.add_argument(
        "-o", "--output", dest="output_opt", help="Alias for output directory"
    )
    parser.add_argument("--config", help="Configuration file (JSON)")
    parser.add_argument(
        "--plots", action="store_true", help="Generate optimization plots"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--no-emoji",
        action="store_true",
        default=None,
        help="Disable emoji in console output (useful for limited terminals)",
    )

    args = parser.parse_args()

    configure_stdio(args.no_emoji)

    # Reconcile optional -i/-o with positional args
    if args.input_opt and args.input_file and args.input_opt != args.input_file:
        print("❌ Conflicting input provided via positional and -i/--input")
        return 2
    if args.output_opt and args.output_dir and args.output_opt != args.output_dir:
        print("❌ Conflicting output provided via positional and -o/--output")
        return 2
    if args.input_opt and not args.input_file:
        args.input_file = args.input_opt
    if args.output_opt and not args.output_dir:
        args.output_dir = args.output_opt

    # Validate required args after reconciliation
    if not args.input_file or not args.output_dir:
        parser.print_help()
        return 2

    # Setup logging: console without timestamps; file with timestamps
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_level = getattr(logging, args.log_level.upper())
    logger_root = logging.getLogger()
    logger_root.handlers.clear()
    logger_root.setLevel(log_level)
    fh = logging.FileHandler(output_path / "optimization.log", encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger_root.addHandler(fh)
    logger_root.addHandler(ch)

    # Load configuration
    config = None
    if args.config and Path(args.config).exists():
        import json

        with open(args.config, "r") as f:
            config = json.load(f).get("optimization", {})

    # Load data
    try:
        df = pd.read_csv(args.input_file)
        logger.info(f"Loaded {len(df)} records from {args.input_file}")

        # Validate required columns
        required_cols = ["atlas", "connectivity_metric"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            logger.error(f"Available columns: {list(df.columns)}")
            return 1

        logger.info(f"Found {df['atlas'].nunique()} unique atlases")
        logger.info(
            f"Found {df['connectivity_metric'].nunique()} unique connectivity metrics"
        )

    except Exception as e:
        logger.error(f"Error loading data from {args.input_file}: {e}")
        return 1

    # Initialize optimizer and run optimization
    try:
        optimizer = MetricOptimizer(config)
        optimized_df = optimizer.optimize_metrics(df)

        # Save results
        # Save optimized metrics
        output_file = output_path / "optimized_metrics.csv"
        optimized_df.to_csv(output_file, index=False)
        logger.info(f"Saved optimized metrics to {output_file}")

        # Generate report
        report_file = output_path / "optimization_report.txt"
        optimizer.generate_report(optimized_df, str(report_file))

        # Generate plots if requested
        if args.plots:
            plot_files = optimizer.create_optimization_plots(
                optimized_df, str(output_path)
            )
            logger.info(f"Generated {len(plot_files)} optimization plots")

        # Print summary
        summary = optimizer.generate_optimization_summary(optimized_df)
        print(f"\n🎯 Optimization Summary:")
        print(f"{'='*50}")
        print(f"Total combinations: {summary['total_combinations']}")
        print(f"High-quality combinations: {summary['high_quality_combinations']}")
        print(f"Recommended combinations: {summary['recommended_combinations']}")
        print(f"Mean quality score (normalized): {summary['mean_quality_score']:.3f}")
        print(f"Max quality score (normalized): {summary['max_quality_score']:.3f}")
        if "mean_quality_score_raw" in summary:
            print(f"Mean quality score (raw): {summary['mean_quality_score_raw']:.3f}")
            print(f"Max quality score (raw): {summary['max_quality_score_raw']:.3f}")
        print(f"Results saved to {output_path}")

        return 0

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
