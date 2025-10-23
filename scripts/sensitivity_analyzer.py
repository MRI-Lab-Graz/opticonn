#!/usr/bin/env python3
"""
Sensitivity Analysis for Tractography Parameters
=================================================

Analyzes how changes in tractography parameters affect QA scores.
Identifies which parameters have the most impact on network quality.

This helps understand:
- Which parameters matter most for your data
- Which parameters can be held constant
- How to focus optimization efforts

Author: Braingraph Pipeline Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import sys

from scripts.utils.runtime import configure_stdio

logger = logging.getLogger(__name__)


class SensitivityAnalyzer:
    """
    Performs sensitivity analysis on tractography parameters.
    
    For each parameter, computes the gradient (∂QA/∂param) to measure
    how much QA score changes when the parameter changes.
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        baseline_config: Dict,
        perturbation_factor: float = 0.1,
        verbose: bool = False
    ):
        """
        Initialize sensitivity analyzer.

        Args:
            data_dir: Path to input data directory
            output_dir: Path to output directory
            baseline_config: Baseline configuration dictionary
            perturbation_factor: Fraction to perturb each parameter (default: 0.1 = 10%)
            verbose: Enable verbose output
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.baseline_config = baseline_config
        self.perturbation_factor = perturbation_factor
        self.verbose = verbose

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.baseline_qa = None
        self.sensitivities = {}
        self.evaluation_results = []

    def _get_parameter_value(self, param_name: str) -> float:
        """Get current value of a parameter from config."""
        if param_name == 'tract_count':
            return self.baseline_config.get('tract_count', 10000)
        
        tracking = self.baseline_config.get('tracking_parameters', {})
        connectivity = self.baseline_config.get('connectivity_options', {})
        
        param_map = {
            'fa_threshold': tracking.get('fa_threshold', 0.1),
            'min_length': tracking.get('min_length', 10),
            'turning_angle': tracking.get('turning_angle', 60.0),
            'step_size': tracking.get('step_size', 1.0),
            'track_voxel_ratio': tracking.get('track_voxel_ratio', 2.0),
            'connectivity_threshold': connectivity.get('connectivity_threshold', 0.001),
        }
        
        return param_map.get(param_name, 0)

    def _set_parameter_value(self, config: Dict, param_name: str, value: float) -> Dict:
        """Set a parameter value in config."""
        config = config.copy()
        
        if param_name == 'tract_count':
            config['tract_count'] = int(value)
        else:
            tracking = config.get('tracking_parameters', {}).copy()
            connectivity = config.get('connectivity_options', {}).copy()
            
            if param_name in ['fa_threshold', 'min_length', 'turning_angle', 'step_size', 'track_voxel_ratio']:
                if param_name == 'min_length':
                    tracking[param_name] = int(value)
                else:
                    tracking[param_name] = float(value)
                config['tracking_parameters'] = tracking
            elif param_name == 'connectivity_threshold':
                connectivity[param_name] = float(value)
                config['connectivity_options'] = connectivity
        
        return config

    def _evaluate_config(self, config: Dict, label: str) -> float:
        """
        Evaluate a configuration and return QA score.

        Args:
            config: Configuration dictionary
            label: Label for this evaluation

        Returns:
            Mean QA score
        """
        logger.info(f" Evaluating: {label}")

        # Save config
        config_path = self.output_dir / f"config_{label}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Create output directory
        eval_output = self.output_dir / label
        eval_output.mkdir(exist_ok=True)

        try:
            # Run pipeline
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "run_pipeline.py"),
                "--data-dir", str(self.data_dir),
                "--output", str(eval_output),
                "--extraction-config", str(config_path),
                "--step", "all",
                "--quiet"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                logger.warning(f"  Pipeline failed for {label}")
                return 0.0

            # Extract QA score
            opt_csv = eval_output / "02_optimization" / "optimized_metrics.csv"
            if not opt_csv.exists():
                logger.warning(f"  No optimization results for {label}")
                return 0.0

            df = pd.read_csv(opt_csv)
            
            if 'pure_qa_score' in df.columns:
                mean_qa = float(df['pure_qa_score'].mean())
            elif 'quality_score' in df.columns:
                mean_qa = float(df['quality_score'].mean())
            else:
                return 0.0

            logger.info(f"  QA Score: {mean_qa:.4f}")
            
            return mean_qa

        except Exception as e:
            logger.error(f" Error evaluating {label}: {e}")
            return 0.0

    def analyze_parameter(self, param_name: str) -> Tuple[float, float, float]:
        """
        Analyze sensitivity of a single parameter.

        Args:
            param_name: Name of parameter to analyze

        Returns:
            Tuple of (sensitivity, baseline_value, perturbed_value)
        """
        logger.info(f"\n{'='*70}")
        logger.info(f" Analyzing parameter: {param_name}")
        logger.info(f"{'='*70}")

        # Get baseline value
        baseline_value = self._get_parameter_value(param_name)
        logger.info(f"Baseline value: {baseline_value}")

        # Calculate perturbation
        if param_name == 'min_length':
            # Integer parameter
            delta = max(1, int(baseline_value * self.perturbation_factor))
        else:
            delta = baseline_value * self.perturbation_factor
        
        perturbed_value = baseline_value + delta
        logger.info(f"Perturbed value: {perturbed_value} (delta: {delta})")

        # Evaluate baseline (if not already done)
        if self.baseline_qa is None:
            logger.info("\n Evaluating baseline configuration...")
            self.baseline_qa = self._evaluate_config(
                self.baseline_config,
                "baseline"
            )
            logger.info(f"Baseline QA Score: {self.baseline_qa:.4f}")

        # Evaluate perturbed configuration
        perturbed_config = self._set_parameter_value(
            self.baseline_config,
            param_name,
            perturbed_value
        )
        
        perturbed_qa = self._evaluate_config(
            perturbed_config,
            f"perturbed_{param_name}"
        )

        # Calculate sensitivity (gradient)
        if delta != 0:
            sensitivity = (perturbed_qa - self.baseline_qa) / delta
        else:
            sensitivity = 0.0

        logger.info(f"\n Sensitivity (∂QA/∂{param_name}): {sensitivity:.6f}")
        
        # Interpret sensitivity
        if abs(sensitivity) > 0.01:
            impact = "HIGH"
        elif abs(sensitivity) > 0.001:
            impact = "MEDIUM"
        else:
            impact = "LOW"
        
        logger.info(f"Impact level: {impact}")

        return sensitivity, baseline_value, perturbed_value

    def analyze_all_parameters(
        self,
        parameters: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Analyze sensitivity of all (or specified) parameters.

        Args:
            parameters: List of parameter names to analyze (None = all)

        Returns:
            Dictionary of sensitivity results
        """
        if parameters is None:
            parameters = [
                'tract_count',
                'fa_threshold',
                'min_length',
                'turning_angle',
                'step_size',
                'track_voxel_ratio',
                'connectivity_threshold'
            ]

        logger.info("\n" + "="*70)
        logger.info(" TRACTOGRAPHY PARAMETER SENSITIVITY ANALYSIS")
        logger.info("="*70)
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Parameters to analyze: {', '.join(parameters)}")
        logger.info(f"Perturbation factor: {self.perturbation_factor:.1%}")
        logger.info("="*70)

        results = {}
        
        for param_name in parameters:
            try:
                sensitivity, baseline, perturbed = self.analyze_parameter(param_name)
                
                results[param_name] = {
                    'sensitivity': float(sensitivity),
                    'baseline_value': float(baseline),
                    'perturbed_value': float(perturbed),
                    'baseline_qa': float(self.baseline_qa),
                    'abs_sensitivity': float(abs(sensitivity))
                }
                
            except Exception as e:
                logger.error(f" Failed to analyze {param_name}: {e}")
                results[param_name] = {
                    'sensitivity': 0.0,
                    'error': str(e)
                }

        # Sort by absolute sensitivity
        sorted_params = sorted(
            results.items(),
            key=lambda x: x[1].get('abs_sensitivity', 0),
            reverse=True
        )

        # Print summary
        logger.info("\n" + "="*70)
        logger.info(" SENSITIVITY ANALYSIS SUMMARY")
        logger.info("="*70)
        logger.info(f"Baseline QA Score: {self.baseline_qa:.4f}\n")
        logger.info("Parameters ranked by impact:\n")
        
        for i, (param, data) in enumerate(sorted_params, 1):
            if 'error' in data:
                continue
            
            sens = data['sensitivity']
            abs_sens = data['abs_sensitivity']
            
            if abs_sens > 0.01:
                impact = " HIGH"
            elif abs_sens > 0.001:
                impact = " MEDIUM"
            else:
                impact = " LOW"
            
            logger.info(f"{i}. {param:25s} {impact}")
            logger.info(f"   Sensitivity: {sens:+.6f} QA per unit change")
            logger.info(f"   Baseline: {data['baseline_value']:.4f}")
            logger.info("")

        # Save results
        results_file = self.output_dir / "sensitivity_analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'baseline_qa': float(self.baseline_qa),
                'perturbation_factor': self.perturbation_factor,
                'parameters': results,
                'ranked_parameters': [p for p, _ in sorted_params]
            }, f, indent=2)

        logger.info(f" Results saved to: {results_file}")
        
        # Create visualization
        self._create_visualization(results, sorted_params)

        logger.info("="*70 + "\n")
        
        return results

    def _create_visualization(
        self,
        results: Dict,
        sorted_params: List[Tuple[str, Dict]]
    ):
        """Create visualization of sensitivity analysis."""
        try:
            # Extract data for plotting
            param_names = [p for p, d in sorted_params if 'error' not in d]
            sensitivities = [results[p]['sensitivity'] for p in param_names]
            abs_sensitivities = [results[p]['abs_sensitivity'] for p in param_names]

            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Plot 1: Sensitivity (with sign)
            colors = ['red' if s < 0 else 'green' for s in sensitivities]
            ax1.barh(param_names, sensitivities, color=colors, alpha=0.7)
            ax1.set_xlabel('Sensitivity (∂QA/∂param)', fontsize=12)
            ax1.set_title('Parameter Sensitivity\n(Positive = Increases QA)', fontsize=14)
            ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
            ax1.grid(axis='x', alpha=0.3)

            # Plot 2: Absolute sensitivity (impact magnitude)
            ax2.barh(param_names, abs_sensitivities, color='steelblue', alpha=0.7)
            ax2.set_xlabel('Absolute Sensitivity (Impact Magnitude)', fontsize=12)
            ax2.set_title('Parameter Impact Magnitude', fontsize=14)
            ax2.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            
            # Save figure
            plot_file = self.output_dir / "sensitivity_analysis_plot.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f" Visualization saved to: {plot_file}")
            
            plt.close()

        except Exception as e:
            logger.warning(f"  Could not create visualization: {e}")


def main():
    """Command line interface for sensitivity analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for tractography parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Analyze all parameters
  python sensitivity_analyzer.py -i data/fib_samples -o results/sensitivity \\
      --config configs/base_config.json

  # Analyze specific parameters only
  python sensitivity_analyzer.py -i data/fib_samples -o results/sensitivity \\
      --config configs/base_config.json \\
      --parameters tract_count fa_threshold min_length

  # Use larger perturbation (20% instead of 10%)
  python sensitivity_analyzer.py -i data/fib_samples -o results/sensitivity \\
      --config configs/base_config.json --perturbation 0.2

This helps identify which parameters have the most impact on QA scores,
allowing you to focus optimization efforts where they matter most.
        """
    )

    parser.add_argument(
        '-i', '--data-dir',
        required=True,
        help='Input data directory containing .fz or .fib.gz files'
    )
    parser.add_argument(
        '-o', '--output-dir',
        required=True,
        help='Output directory for sensitivity analysis results'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Baseline configuration JSON file'
    )
    parser.add_argument(
        '--parameters',
        nargs='+',
        help='Specific parameters to analyze (default: all)'
    )
    parser.add_argument(
        '--perturbation',
        type=float,
        default=0.1,
        help='Perturbation factor as fraction of baseline (default: 0.1 = 10%%)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--no-emoji',
        action='store_true',
        help='Disable emoji in console output'
    )

    args = parser.parse_args()

    configure_stdio(args.no_emoji)

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    # Load baseline configuration
    try:
        with open(args.config, 'r') as f:
            baseline_config = json.load(f)
    except FileNotFoundError:
        logger.error(f" Configuration file not found: {args.config}")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f" Invalid JSON in configuration file: {e}")
        return 1

    # Create analyzer
    analyzer = SensitivityAnalyzer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        baseline_config=baseline_config,
        perturbation_factor=args.perturbation,
        verbose=args.verbose
    )

    # Run analysis
    try:
        results = analyzer.analyze_all_parameters(args.parameters)
        logger.info(" Sensitivity analysis completed successfully!")
        return 0
    except Exception as e:
        logger.error(f" Sensitivity analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
