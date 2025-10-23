#!/usr/bin/env python3
# Supports: --dry-run (prints intended actions without running)
# When run without arguments the script prints help: parser.print_help()
"""
Statistical Analysis Module
===========================

Statistical modeling and analysis for structural connectivity data.
Supports flexible model formulations and comprehensive result reporting.

Author: Braingraph Pipeline Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import warnings
from statsmodels.formula.api import mixedlm, ols
from statsmodels.stats.multitest import multipletests
import json

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class StatisticalAnalysis:
    """
    Statistical analysis for structural connectivity data.

    This class provides:
    - Flexible statistical model formulations
    - Mixed-effects modeling for repeated measures
    - Multiple comparison corrections
    - Effect size calculations
    - Comprehensive visualization
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the statistical analyzer.

        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config or self._default_config()
        self.results = {}

    def _default_config(self) -> Dict:
        """Default statistical analysis configuration."""
        return {
            "statistical_model": "{metric} ~ timepoint + (1|subject)",
            "metrics_to_analyze": [
                "global_efficiency",
                "characteristic_path_length",
                "modularity",
                "clustering_coefficient",
                "assortativity",
                "small_worldness",
                "sparsity",
            ],
            "grouping_variables": ["atlas", "connectivity_metric"],
            "multiple_comparison_correction": "bonferroni",
            "significance_level": 0.05,
            "effect_size_threshold": 0.2,
            "model_type": "mixed",  # 'mixed', 'ols', 'anova'
            "random_effects": True,
            "generate_plots": True,
            "save_individual_results": True,
        }

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for statistical analysis.

        Args:
            df: DataFrame with connectivity metrics

        Returns:
            Prepared DataFrame
        """
        logger.info("Preparing data for statistical analysis...")

        # Create a copy to avoid modifying original data
        data = df.copy()

        # Ensure subject_id is string type
        if "subject_id" in data.columns:
            data["subject_id"] = data["subject_id"].astype(str)

        # Create timepoint variable if not present
        if "timepoint" not in data.columns:
            if "session" in data.columns:
                data["timepoint"] = data["session"]
            else:
                # Try to extract from subject_id or filename
                logger.warning(
                    "No timepoint/session variable found. Creating dummy timepoint."
                )
                data["timepoint"] = 1

        # Ensure numeric columns are properly typed
        numeric_columns = self.config.get("metrics_to_analyze", [])
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        # Remove rows with all NaN values in metrics
        metric_cols = [col for col in numeric_columns if col in data.columns]
        if metric_cols:
            data = data.dropna(subset=metric_cols, how="all")

        logger.info(
            f"Prepared data: {len(data)} observations, {data['subject_id'].nunique()} subjects"
        )

        return data

    def fit_statistical_model(
        self, data: pd.DataFrame, metric: str, formula: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fit statistical model for a specific metric.

        Args:
            data: DataFrame with the data
            metric: Name of the metric to analyze
            formula: Model formula (if None, uses config default)

        Returns:
            Dictionary with model results
        """
        if formula is None:
            formula = self.config["statistical_model"].format(metric=metric)

        # Remove missing values for this metric
        model_data = data.dropna(subset=[metric])

        if len(model_data) < 3:
            logger.warning(f"Insufficient data for {metric} (n={len(model_data)})")
            return {"error": "Insufficient data", "n_obs": len(model_data)}

        try:
            model_type = self.config.get("model_type", "mixed")

            if model_type == "mixed" and "(" in formula and "|" in formula:
                # Mixed-effects model
                try:
                    # Extract grouping variable from formula
                    groups_part = formula.split("(")[1].split(")")[0]
                    if "|" in groups_part:
                        group_var = groups_part.split("|")[1].strip()
                        model = mixedlm(
                            formula, model_data, groups=model_data[group_var]
                        )
                        result = model.fit(method="lbfgs")
                    else:
                        # Fall back to OLS if no grouping variable
                        simple_formula = formula.split("(")[0].strip()
                        if simple_formula.endswith("+"):
                            simple_formula = simple_formula[:-1].strip()
                        model = ols(simple_formula, data=model_data)
                        result = model.fit()
                except Exception as e:
                    logger.warning(f"Mixed-effects model failed for {metric}: {e}")
                    # Fall back to OLS
                    simple_formula = formula.split("(")[0].strip()
                    if simple_formula.endswith("+"):
                        simple_formula = simple_formula[:-1].strip()
                    model = ols(simple_formula, data=model_data)
                    result = model.fit()
            else:
                # Regular OLS model
                clean_formula = formula.split("(")[0].strip()
                if clean_formula.endswith("+"):
                    clean_formula = clean_formula[:-1].strip()
                model = ols(clean_formula, data=model_data)
                result = model.fit()

            # Extract results
            model_results = {
                "model_type": type(result).__name__,
                "formula": formula,
                "n_obs": result.nobs,
                "params": result.params.to_dict(),
                "pvalues": result.pvalues.to_dict(),
                "tvalues": getattr(result, "tvalues", pd.Series()).to_dict(),
                "conf_int": (
                    result.conf_int().to_dict() if hasattr(result, "conf_int") else {}
                ),
                "rsquared": getattr(result, "rsquared", np.nan),
                "rsquared_adj": getattr(result, "rsquared_adj", np.nan),
                "aic": result.aic,
                "bic": result.bic,
                "summary": str(result.summary()),
                "model_object": result,  # Store for further analysis
            }

            return model_results

        except Exception as e:
            logger.error(f"Model fitting failed for {metric}: {e}")
            return {"error": str(e), "n_obs": len(model_data)}

    def compute_effect_sizes(
        self, data: pd.DataFrame, metric: str, grouping_var: str = "timepoint"
    ) -> Dict[str, float]:
        """
        Compute effect sizes for group comparisons.

        Args:
            data: DataFrame with the data
            metric: Name of the metric
            grouping_var: Variable defining groups

        Returns:
            Dictionary with effect sizes
        """
        effect_sizes = {}

        try:
            if grouping_var in data.columns and metric in data.columns:
                groups = data[grouping_var].unique()
                metric_data = data.dropna(subset=[metric])

                if len(groups) >= 2:
                    # Cohen's d for pairwise comparisons
                    for i, group1 in enumerate(groups):
                        for group2 in groups[i + 1 :]:
                            group1_data = metric_data[
                                metric_data[grouping_var] == group1
                            ][metric]
                            group2_data = metric_data[
                                metric_data[grouping_var] == group2
                            ][metric]

                            if len(group1_data) > 0 and len(group2_data) > 0:
                                # Cohen's d
                                pooled_std = np.sqrt(
                                    (
                                        (len(group1_data) - 1)
                                        * np.var(group1_data, ddof=1)
                                        + (len(group2_data) - 1)
                                        * np.var(group2_data, ddof=1)
                                    )
                                    / (len(group1_data) + len(group2_data) - 2)
                                )

                                if pooled_std > 0:
                                    cohens_d = (
                                        np.mean(group1_data) - np.mean(group2_data)
                                    ) / pooled_std
                                    effect_sizes[f"cohens_d_{group1}_vs_{group2}"] = (
                                        cohens_d
                                    )

                # Eta-squared (proportion of variance explained)
                if len(groups) > 1:
                    try:
                        from scipy.stats import f_oneway

                        group_data = [
                            metric_data[metric_data[grouping_var] == group][
                                metric
                            ].values
                            for group in groups
                            if len(metric_data[metric_data[grouping_var] == group]) > 0
                        ]

                        if len(group_data) > 1:
                            f_stat, p_val = f_oneway(*group_data)

                            # Calculate eta-squared
                            ss_total = np.sum(
                                [
                                    (x - np.mean(metric_data[metric])) ** 2
                                    for x in metric_data[metric]
                                ]
                            )
                            ss_between = np.sum(
                                [
                                    len(group)
                                    * (np.mean(group) - np.mean(metric_data[metric]))
                                    ** 2
                                    for group in group_data
                                ]
                            )

                            if ss_total > 0:
                                eta_squared = ss_between / ss_total
                                effect_sizes["eta_squared"] = eta_squared
                    except Exception as e:
                        logger.warning(
                            f"Could not compute eta-squared for {metric}: {e}"
                        )

        except Exception as e:
            logger.warning(f"Effect size computation failed for {metric}: {e}")

        return effect_sizes

    def correct_multiple_comparisons(
        self, pvalues: List[float], method: str = None
    ) -> Tuple[List[bool], List[float]]:
        """
        Apply multiple comparison correction.

        Args:
            pvalues: List of p-values
            method: Correction method ('bonferroni', 'fdr_bh', etc.)

        Returns:
            Tuple of (reject_null, corrected_pvalues)
        """
        if method is None:
            method = self.config.get("multiple_comparison_correction", "bonferroni")

        if len(pvalues) == 0:
            return [], []

        # Filter out NaN values
        valid_pvalues = [p for p in pvalues if not np.isnan(p)]

        if len(valid_pvalues) == 0:
            return [False] * len(pvalues), [np.nan] * len(pvalues)

        try:
            rejected, corrected_pvalues, _, _ = multipletests(
                valid_pvalues,
                alpha=self.config.get("significance_level", 0.05),
                method=method,
            )

            # Map back to original indices
            result_rejected = []
            result_corrected = []
            valid_idx = 0

            for p in pvalues:
                if np.isnan(p):
                    result_rejected.append(False)
                    result_corrected.append(np.nan)
                else:
                    result_rejected.append(rejected[valid_idx])
                    result_corrected.append(corrected_pvalues[valid_idx])
                    valid_idx += 1

            return result_rejected, result_corrected

        except Exception as e:
            logger.warning(f"Multiple comparison correction failed: {e}")
            alpha = self.config.get("significance_level", 0.05)
            return [p < alpha for p in pvalues], pvalues

    def analyze_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis on connectivity metrics.

        Args:
            df: DataFrame with connectivity metrics

        Returns:
            Dictionary with all analysis results
        """
        logger.info("Starting statistical analysis...")

        # Prepare data
        data = self.prepare_data(df)

        # Get metrics to analyze
        metrics_to_analyze = self.config.get("metrics_to_analyze", [])
        available_metrics = [m for m in metrics_to_analyze if m in data.columns]

        if not available_metrics:
            logger.error("No specified metrics found in data")
            return {"error": "No metrics to analyze"}

        logger.info(f"Analyzing {len(available_metrics)} metrics: {available_metrics}")

        # Group by atlas and connectivity metric for separate analyses
        grouping_vars = self.config.get(
            "grouping_variables", ["atlas", "connectivity_metric"]
        )
        available_grouping = [var for var in grouping_vars if var in data.columns]

        all_results = {}
        all_pvalues = []
        analysis_info = []

        # Analyze each group separately
        for group_name, group_data in data.groupby(available_grouping):
            if len(available_grouping) == 1:
                group_key = f"{available_grouping[0]}_{group_name}"
            else:
                group_key = "_".join([str(gn) for gn in group_name])

            logger.info(f"Analyzing group: {group_key} (n={len(group_data)})")

            group_results = {}

            for metric in available_metrics:
                if metric not in group_data.columns:
                    continue

                # Check if there's enough data
                metric_data = group_data.dropna(subset=[metric])
                if len(metric_data) < 3:
                    logger.warning(f"Insufficient data for {metric} in {group_key}")
                    continue

                # Fit statistical model
                model_results = self.fit_statistical_model(metric_data, metric)

                # Compute effect sizes
                effect_sizes = self.compute_effect_sizes(metric_data, metric)

                # Combine results
                analysis_result = {
                    "metric": metric,
                    "group": group_key,
                    "model_results": model_results,
                    "effect_sizes": effect_sizes,
                    "descriptive_stats": {
                        "mean": float(metric_data[metric].mean()),
                        "std": float(metric_data[metric].std()),
                        "median": float(metric_data[metric].median()),
                        "min": float(metric_data[metric].min()),
                        "max": float(metric_data[metric].max()),
                        "n_obs": len(metric_data),
                    },
                }

                group_results[metric] = analysis_result

                # Collect p-values for multiple comparison correction
                if "pvalues" in model_results and isinstance(
                    model_results["pvalues"], dict
                ):
                    for param, pval in model_results["pvalues"].items():
                        if not np.isnan(pval):
                            all_pvalues.append(pval)
                            analysis_info.append((group_key, metric, param))

            all_results[group_key] = group_results

        # Apply multiple comparison correction
        if all_pvalues:
            rejected, corrected_pvalues = self.correct_multiple_comparisons(all_pvalues)

            # Update results with corrected p-values
            for i, (group_key, metric, param) in enumerate(analysis_info):
                if group_key in all_results and metric in all_results[group_key]:
                    if "corrected_pvalues" not in all_results[group_key][metric]:
                        all_results[group_key][metric]["corrected_pvalues"] = {}

                    all_results[group_key][metric]["corrected_pvalues"][param] = {
                        "corrected_pvalue": corrected_pvalues[i],
                        "significant_corrected": rejected[i],
                        "original_pvalue": all_pvalues[i],
                    }

        # Generate summary
        summary = self._generate_analysis_summary(all_results)

        # Compile final results
        final_results = {
            "analysis_results": all_results,
            "summary": summary,
            "configuration": self.config,
            "n_groups": len(all_results),
            "n_metrics": len(available_metrics),
            "n_comparisons": len(all_pvalues),
            "multiple_comparison_method": self.config.get(
                "multiple_comparison_correction", "bonferroni"
            ),
        }

        self.results = final_results
        logger.info("Statistical analysis completed")

        return final_results

    def _generate_analysis_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate summary of analysis results."""
        summary = {
            "significant_effects": [],
            "large_effects": [],
            "group_summaries": {},
            "overall_statistics": {},
        }

        effect_threshold = self.config.get("effect_size_threshold", 0.2)

        all_effects = []
        significant_count = 0
        total_tests = 0

        for group_key, group_results in results.items():
            group_summary = {
                "n_metrics": len(group_results),
                "significant_effects": 0,
                "large_effects": 0,
            }

            for metric, metric_results in group_results.items():
                # Check for significant effects
                if "corrected_pvalues" in metric_results:
                    for param, correction_info in metric_results[
                        "corrected_pvalues"
                    ].items():
                        total_tests += 1
                        if correction_info["significant_corrected"]:
                            significant_count += 1
                            group_summary["significant_effects"] += 1
                            summary["significant_effects"].append(
                                f"{group_key} - {metric} - {param}: p={correction_info['corrected_pvalue']:.4f}"
                            )

                # Check for large effect sizes
                if "effect_sizes" in metric_results:
                    for effect_name, effect_value in metric_results[
                        "effect_sizes"
                    ].items():
                        if not np.isnan(effect_value):
                            all_effects.append(abs(effect_value))
                            if abs(effect_value) >= effect_threshold:
                                group_summary["large_effects"] += 1
                                summary["large_effects"].append(
                                    f"{group_key} - {metric} - {effect_name}: {effect_value:.3f}"
                                )

            summary["group_summaries"][group_key] = group_summary

        # Overall statistics
        summary["overall_statistics"] = {
            "total_tests": total_tests,
            "significant_tests": significant_count,
            "proportion_significant": (
                significant_count / total_tests if total_tests > 0 else 0
            ),
            "mean_effect_size": np.mean(all_effects) if all_effects else 0,
            "large_effects_count": len(summary["large_effects"]),
        }

        return summary

    def generate_plots(self, df: pd.DataFrame, output_dir: str) -> List[str]:
        """
        Generate visualization plots for statistical results.

        Args:
            df: DataFrame with connectivity metrics
            output_dir: Directory to save plots

        Returns:
            List of created plot files
        """
        logger.info("Generating statistical analysis plots...")

        output_path = Path(output_dir)
        plots_dir = output_path / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_files = []

        # Set style
        plt.style.use("default")
        sns.set_palette("husl")

        # Prepare data
        data = self.prepare_data(df)
        metrics_to_plot = [
            m for m in self.config.get("metrics_to_analyze", []) if m in data.columns
        ]

        # 1. Effect size overview
        if hasattr(self, "results") and "analysis_results" in self.results:
            effect_sizes_data = []

            for group_key, group_results in self.results["analysis_results"].items():
                for metric, metric_results in group_results.items():
                    if "effect_sizes" in metric_results:
                        for effect_name, effect_value in metric_results[
                            "effect_sizes"
                        ].items():
                            if not np.isnan(effect_value):
                                effect_sizes_data.append(
                                    {
                                        "group": group_key,
                                        "metric": metric,
                                        "effect_type": effect_name,
                                        "effect_size": effect_value,
                                        "abs_effect_size": abs(effect_value),
                                    }
                                )

            if effect_sizes_data:
                effect_df = pd.DataFrame(effect_sizes_data)

                plt.figure(figsize=(14, 8))

                # Plot effect sizes
                plt.subplot(2, 2, 1)
                sns.boxplot(data=effect_df, x="metric", y="abs_effect_size")
                plt.xticks(rotation=45)
                plt.ylabel("Absolute Effect Size")
                plt.title("Effect Sizes by Metric")
                plt.axhline(y=0.2, color="red", linestyle="--", label="Small effect")
                plt.axhline(
                    y=0.5, color="orange", linestyle="--", label="Medium effect"
                )
                plt.axhline(y=0.8, color="green", linestyle="--", label="Large effect")
                plt.legend()

                # Plot by group
                plt.subplot(2, 2, 2)
                if (
                    len(effect_df["group"].unique()) <= 10
                ):  # Only if not too many groups
                    sns.boxplot(data=effect_df, x="group", y="abs_effect_size")
                    plt.xticks(rotation=45)
                else:
                    plt.hist(effect_df["abs_effect_size"], bins=30, alpha=0.7)
                    plt.xlabel("Absolute Effect Size")
                    plt.ylabel("Frequency")
                plt.ylabel("Absolute Effect Size")
                plt.title("Effect Sizes by Group")

                # Histogram of effect sizes
                plt.subplot(2, 2, 3)
                plt.hist(
                    effect_df["abs_effect_size"], bins=30, alpha=0.7, edgecolor="black"
                )
                plt.axvline(x=0.2, color="red", linestyle="--", label="Small effect")
                plt.axvline(
                    x=0.5, color="orange", linestyle="--", label="Medium effect"
                )
                plt.axvline(x=0.8, color="green", linestyle="--", label="Large effect")
                plt.xlabel("Absolute Effect Size")
                plt.ylabel("Frequency")
                plt.title("Distribution of Effect Sizes")
                plt.legend()

                # Top effects
                plt.subplot(2, 2, 4)
                top_effects = effect_df.nlargest(15, "abs_effect_size")
                y_pos = np.arange(len(top_effects))
                plt.barh(y_pos, top_effects["abs_effect_size"])
                plt.yticks(
                    y_pos,
                    [f"{row['metric'][:10]}..." for _, row in top_effects.iterrows()],
                )
                plt.xlabel("Absolute Effect Size")
                plt.title("Top 15 Effect Sizes")

                plt.tight_layout()
                plot_file = plots_dir / "effect_sizes_overview.png"
                plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                plt.close()
                plot_files.append(str(plot_file))

        # 2. Metric distributions by timepoint/condition
        if "timepoint" in data.columns and len(data["timepoint"].unique()) > 1:
            n_metrics = len(metrics_to_plot)
            if n_metrics > 0:
                n_cols = min(3, n_metrics)
                n_rows = (n_metrics + n_cols - 1) // n_cols

                plt.figure(figsize=(5 * n_cols, 4 * n_rows))

                for i, metric in enumerate(metrics_to_plot[:12]):  # Limit to 12 metrics
                    plt.subplot(n_rows, n_cols, i + 1)

                    metric_data = data.dropna(subset=[metric])
                    if len(metric_data) > 0:
                        sns.boxplot(data=metric_data, x="timepoint", y=metric)
                        plt.title(f"{metric}")
                        plt.xticks(rotation=45)

                plt.tight_layout()
                plot_file = plots_dir / "metric_distributions.png"
                plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                plt.close()
                plot_files.append(str(plot_file))

        # 3. Correlation matrix of metrics
        correlation_metrics = [
            m for m in metrics_to_plot if data[m].notna().sum() > 10
        ][
            :15
        ]  # Limit to 15 metrics

        if len(correlation_metrics) > 1:
            plt.figure(figsize=(12, 10))

            corr_data = data[correlation_metrics].corr()
            mask = np.triu(np.ones_like(corr_data, dtype=bool))

            sns.heatmap(
                corr_data,
                mask=mask,
                annot=True,
                fmt=".2",
                cmap="coolwarm",
                center=0,
                square=True,
                cbar_kws={"label": "Correlation"},
            )
            plt.title("Metric Correlations")

            plot_file = plots_dir / "metric_correlations.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            plot_files.append(str(plot_file))

        # 4. Summary statistics plot
        if hasattr(self, "results") and "summary" in self.results:
            summary = self.results["summary"]

            plt.figure(figsize=(12, 8))

            # Overall statistics
            plt.subplot(2, 2, 1)
            stats_data = summary.get("overall_statistics", {})
            categories = list(stats_data.keys())
            values = list(stats_data.values())

            if categories and values:
                plt.bar(categories, values)
                plt.xticks(rotation=45)
                plt.title("Overall Statistics")

            # Significant effects by group
            plt.subplot(2, 2, 2)
            group_summaries = summary.get("group_summaries", {})
            if group_summaries:
                groups = list(group_summaries.keys())[:10]  # Limit to 10 groups
                sig_effects = [
                    group_summaries[g].get("significant_effects", 0) for g in groups
                ]

                plt.bar(range(len(groups)), sig_effects)
                plt.xticks(
                    range(len(groups)),
                    [g[:15] + "..." if len(g) > 15 else g for g in groups],
                    rotation=45,
                )
                plt.ylabel("Significant Effects")
                plt.title("Significant Effects by Group")

            # Effect sizes by group
            plt.subplot(2, 2, 3)
            if group_summaries:
                large_effects = [
                    group_summaries[g].get("large_effects", 0) for g in groups
                ]

                plt.bar(range(len(groups)), large_effects)
                plt.xticks(
                    range(len(groups)),
                    [g[:15] + "..." if len(g) > 15 else g for g in groups],
                    rotation=45,
                )
                plt.ylabel("Large Effects")
                plt.title("Large Effects by Group")

            plt.tight_layout()
            plot_file = plots_dir / "analysis_summary.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            plot_files.append(str(plot_file))

        logger.info(f"Generated {len(plot_files)} statistical analysis plots")
        return plot_files

    def save_results(self, output_dir: str) -> Dict[str, str]:
        """
        Save all analysis results to files.

        Args:
            output_dir: Directory to save results

        Returns:
            Dictionary with paths to saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        if not hasattr(self, "results") or not self.results:
            logger.warning("No results to save")
            return saved_files

        # Save main results as JSON
        results_file = output_path / "statistical_results.json"

        # Create a copy of results without model objects (not JSON serializable)
        results_copy = {}
        for key, value in self.results.items():
            if key == "analysis_results":
                results_copy[key] = {}
                for group_key, group_results in value.items():
                    results_copy[key][group_key] = {}
                    for metric, metric_results in group_results.items():
                        results_copy[key][group_key][metric] = {}
                        for res_key, res_value in metric_results.items():
                            if res_key == "model_results" and isinstance(
                                res_value, dict
                            ):
                                # Remove model object
                                model_copy = {
                                    k: v
                                    for k, v in res_value.items()
                                    if k != "model_object"
                                }
                                results_copy[key][group_key][metric][
                                    res_key
                                ] = model_copy
                            else:
                                results_copy[key][group_key][metric][
                                    res_key
                                ] = res_value
            else:
                results_copy[key] = value

        with open(results_file, "w") as f:
            json.dump(results_copy, f, indent=2, default=str)
        saved_files["results"] = str(results_file)

        # Save summary as text
        summary_file = output_path / "analysis_summary.txt"
        self._write_summary_report(summary_file)
        saved_files["summary"] = str(summary_file)

        # Save detailed results as CSV
        csv_file = output_path / "detailed_results.csv"
        self._save_results_csv(csv_file)
        saved_files["csv"] = str(csv_file)

        logger.info(f"Saved statistical results to {output_path}")
        return saved_files

    def _write_summary_report(self, output_file: str) -> None:
        """Write a human-readable summary report."""
        if not hasattr(self, "results") or not self.results:
            return

        summary = self.results.get("summary", {})

        with open(output_file, "w") as f:
            f.write("Statistical Analysis Summary Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(
                f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            # Overall statistics
            overall_stats = summary.get("overall_statistics", {})
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 18 + "\n")
            for stat, value in overall_stats.items():
                f.write(f"{stat}: {value}\n")
            f.write("\n")

            # Significant effects
            sig_effects = summary.get("significant_effects", [])
            f.write(f"SIGNIFICANT EFFECTS ({len(sig_effects)} found)\n")
            f.write("-" * 25 + "\n")
            for effect in sig_effects:
                f.write(f"• {effect}\n")
            f.write("\n")

            # Large effects
            large_effects = summary.get("large_effects", [])
            f.write(f"LARGE EFFECT SIZES ({len(large_effects)} found)\n")
            f.write("-" * 23 + "\n")
            for effect in large_effects:
                f.write(f"• {effect}\n")
            f.write("\n")

            # Group summaries
            group_summaries = summary.get("group_summaries", {})
            f.write("GROUP SUMMARIES\n")
            f.write("-" * 15 + "\n")
            for group, group_summary in group_summaries.items():
                f.write(f"{group}:\n")
                f.write(f"  Metrics analyzed: {group_summary.get('n_metrics', 0)}\n")
                f.write(
                    f"  Significant effects: {group_summary.get('significant_effects', 0)}\n"
                )
                f.write(f"  Large effects: {group_summary.get('large_effects', 0)}\n")
                f.write("\n")

    def _save_results_csv(self, output_file: str) -> None:
        """Save detailed results as CSV."""
        if not hasattr(self, "results") or "analysis_results" not in self.results:
            return

        rows = []

        for group_key, group_results in self.results["analysis_results"].items():
            for metric, metric_results in group_results.items():
                base_row = {
                    "group": group_key,
                    "metric": metric,
                    "n_obs": metric_results.get("descriptive_stats", {}).get(
                        "n_obs", np.nan
                    ),
                    "mean": metric_results.get("descriptive_stats", {}).get(
                        "mean", np.nan
                    ),
                    "std": metric_results.get("descriptive_stats", {}).get(
                        "std", np.nan
                    ),
                    "median": metric_results.get("descriptive_stats", {}).get(
                        "median", np.nan
                    ),
                }

                # Add model results
                model_results = metric_results.get("model_results", {})
                if "params" in model_results:
                    for param, value in model_results["params"].items():
                        base_row[f"param_{param}"] = value

                if "pvalues" in model_results:
                    for param, value in model_results["pvalues"].items():
                        base_row[f"pvalue_{param}"] = value

                # Add corrected p-values
                if "corrected_pvalues" in metric_results:
                    for param, correction_info in metric_results[
                        "corrected_pvalues"
                    ].items():
                        base_row[f"corrected_pvalue_{param}"] = correction_info[
                            "corrected_pvalue"
                        ]
                        base_row[f"significant_corrected_{param}"] = correction_info[
                            "significant_corrected"
                        ]

                # Add effect sizes
                if "effect_sizes" in metric_results:
                    for effect_name, effect_value in metric_results[
                        "effect_sizes"
                    ].items():
                        base_row[f"effect_{effect_name}"] = effect_value

                rows.append(base_row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)


def main():
    """Command line interface for statistical analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Statistical analysis of connectivity metrics"
    )
    parser.add_argument("input_file", help="CSV file with connectivity metrics")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--config", help="Configuration file (JSON)")
    parser.add_argument("--plots", action="store_true", help="Generate analysis plots")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load configuration
    config = None
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            config = json.load(f).get("analysis", {})

    # Load data
    df = pd.read_csv(args.input_file)
    logger.info(f"Loaded {len(df)} records from {args.input_file}")

    # Initialize analyzer and run analysis
    analyzer = StatisticalAnalysis(config)
    results = analyzer.analyze_metrics(df)

    # Save results
    analyzer.save_results(args.output_dir)

    # Generate plots if requested
    if args.plots:
        plot_files = analyzer.generate_plots(df, args.output_dir)
        print(f"Generated {len(plot_files)} analysis plots")

    # Print summary
    summary = results.get("summary", {})
    overall_stats = summary.get("overall_statistics", {})

    print("\nStatistical Analysis Summary:")
    print(f"Groups analyzed: {results.get('n_groups', 0)}")
    print(f"Metrics analyzed: {results.get('n_metrics', 0)}")
    print(f"Total tests: {overall_stats.get('total_tests', 0)}")
    print(f"Significant tests: {overall_stats.get('significant_tests', 0)}")
    print(f"Large effects found: {overall_stats.get('large_effects_count', 0)}")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
