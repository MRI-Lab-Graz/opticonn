#!/usr/bin/env python3
"""
Cross-Validation Bootstrap Parameter Optimizer

This implements the correct approach:
1. Both waves run the same parameter sweep on different random subject subsets
2. Cross-validation ensures both waves find the same optimal parameters
3. User selects the validated optimal parameters for full analysis

This is much more scientifically sound than comparing different parameter sets.
"""

import json
import logging
import subprocess
import sys
import shutil
import random
import itertools
import time
from pathlib import Path
import pandas as pd
import argparse
from typing import Dict, List, Any, Tuple
import time
import itertools
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CrossValidationBootstrapOptimizer:
    """Cross-validation bootstrap optimizer with parameter sweep validation"""
    
    def __init__(self, config_file: str = "configs/bootstrap_optimization_config.json"):
        """Initialize optimizer with configuration"""
        self.config_file = config_file
        self.config = self.load_config()
        self.wave_results = {}
        self.parameter_combinations = []
        
    def load_config(self) -> Dict[str, Any]:
        """Load bootstrap optimization configuration"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            logging.info(f"✅ Loaded cross-validation config: {self.config_file}")
            return config
        except FileNotFoundError:
            logging.error(f"❌ Configuration file not found: {self.config_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logging.error(f"❌ Invalid JSON in config file: {e}")
            sys.exit(1)
            
    def generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search"""
        
        bootstrap_config = self.config["bootstrap_optimization"]
        param_sweep = bootstrap_config["parameter_sweep"]
        parameters = param_sweep["parameters"]
        
        # Generate all combinations
        param_names = list(parameters.keys())
        param_values = [parameters[name] for name in param_names]
        
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
            
        logging.info(f"📊 Generated {len(combinations)} parameter combinations")
        for i, combo in enumerate(combinations[:3]):  # Show first 3
            logging.info(f"   Example {i+1}: {combo}")
        if len(combinations) > 3:
            logging.info(f"   ... and {len(combinations)-3} more combinations")
            
        return combinations
    
    def generate_parameter_combinations_from_config(self, parameters: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate parameter combinations from given parameter ranges"""
        
        # Generate all combinations
        param_names = list(parameters.keys())
        param_values = [parameters[name] for name in param_names]
        
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
            
        return combinations
        
    def split_subjects_for_cross_validation(self, data_dir: str) -> Tuple[List[str], List[str]]:
        """Split subjects into two random subsets for cross-validation"""
        
        data_path = Path(data_dir)
        all_files = list(data_path.glob("*.fz"))
        
        if not all_files:
            logging.error(f"❌ No .fz files found in: {data_dir}")
            return [], []
            
        # Extract subject identifiers
        subjects = [f.stem for f in all_files]
        
        # Calculate subset sizes
        bootstrap_config = self.config["bootstrap_optimization"]
        sample_percentage = bootstrap_config["sample_percentage"]
        total_subset_size = max(2, int(len(subjects) * sample_percentage / 100))
        subset_size = total_subset_size // 2  # Split into two equal subsets
        
        # Randomly shuffle and split
        random.shuffle(subjects)
        subset_a = subjects[:subset_size]
        subset_b = subjects[subset_size:2*subset_size]
        
        logging.info(f"📈 Cross-validation split:")
        logging.info(f"   Total subjects: {len(subjects)}")
        logging.info(f"   Subset A: {len(subset_a)} subjects")
        logging.info(f"   Subset B: {len(subset_b)} subjects")
        logging.info(f"   Remaining: {len(subjects) - len(subset_a) - len(subset_b)} subjects for full analysis")
        
        return subset_a, subset_b
        
    def create_sweep_configs(self, data_dir: str, subset_a: List[str], subset_b: List[str]) -> List[str]:
        """Create sweep configuration files for both validation waves"""
        
        bootstrap_config = self.config["bootstrap_optimization"]
        output_structure = bootstrap_config.get("output_structure", {})
        bootstrap_base_dir = output_structure.get("bootstrap_base_dir", "results/bootstrap")
        
        # Create bootstrap base directory
        Path(bootstrap_base_dir).mkdir(parents=True, exist_ok=True)
        
        wave_configs = []
        subsets = [subset_a, subset_b]
        
        for i, wave in enumerate(bootstrap_config["waves"], 1):
            wave_name = wave["name"]
            wave_output_dir = Path(bootstrap_base_dir) / wave.get("output_subdir", f"wave_{i}")
            subject_subset = subsets[i-1]
            
            # Create sweep configuration that will test all parameter combinations
            sweep_config = {
                "wave_id": f"cv_wave_{i}",
                "wave_name": wave_name,
                "description": f"{wave['description']} ({len(subject_subset)} subjects)",
                "cross_validation_wave": i,
                "data_directory": data_dir,
                "output_directory": str(wave_output_dir),
                "subject_subset": subject_subset,
                "parameter_sweep": bootstrap_config["parameter_sweep"],
                "evaluation_metrics": bootstrap_config["parameter_sweep"]["evaluation_metrics"]
            }
            
            config_filename = f"cv_wave_{i}_{wave_name}_config.json"
            
            with open(config_filename, 'w') as f:
                json.dump(sweep_config, f, indent=2)
                
            wave_configs.append(config_filename)
            logging.info(f"📝 Created CV wave config: {config_filename}")
            logging.info(f"   └─ Output directory: {wave_output_dir}")
            logging.info(f"   └─ Subjects: {len(subject_subset)}")
            
        return wave_configs
        
    def run_parameter_sweep_wave(self, wave_config: str, wave_num: int) -> Tuple[bool, str, Dict[str, Any]]:
        """Run parameter sweep for a single validation wave"""
        
        logging.info(f"🌊 Starting Cross-Validation Wave {wave_num}...")
        
        # Load wave configuration
        with open(wave_config, 'r') as f:
            config = json.load(f)
            
        wave_name = config["wave_name"]
        output_dir = config["output_directory"]
        subjects = config["subject_subset"]
        param_sweep = config["parameter_sweep"]
        
        logging.info(f"   └─ Wave: {wave_name}")
        logging.info(f"   └─ Subjects: {len(subjects)}")
        logging.info(f"   └─ Parameter combinations: {len(self.parameter_combinations)}")
        
        # For demo purposes, we'll simulate the parameter sweep
        # In practice, this would run the actual connectivity extraction with each parameter set
        
        start_time = time.time()
        
        # Execute real parameter sweep with DSI Studio
        best_params, best_score = self.execute_parameter_sweep(subjects, param_sweep, config["data_directory"])
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Store results
        wave_results = {
            "config": config,
            "output_directory": output_dir,
            "processing_time": processing_time,
            "wave_name": wave_name,
            "best_parameters": best_params,
            "best_score": best_score,
            "n_subjects": len(subjects),
            "subjects": subjects
        }
        
        self.wave_results[wave_num] = wave_results
        
        logging.info(f"✅ CV Wave {wave_num} completed in {processing_time:.1f} seconds")
        logging.info(f"   └─ Best parameters: {best_params}")
        logging.info(f"   └─ Best score: {best_score:.3f}")
        
        return True, output_dir, best_params
        
    def execute_parameter_sweep(self, subjects: List[str], param_sweep: Dict[str, Any], data_dir: str) -> Tuple[Dict[str, Any], float]:
        """Execute real parameter sweep optimization using DSI Studio"""
        
        logger = logging.getLogger(__name__)
        
        # Generate parameter combinations from the current parameter sweep  
        param_combinations = self.generate_parameter_combinations_from_config(param_sweep["parameters"])
        
        logger.info(f"🔄 Testing {len(param_combinations)} parameter combinations on {len(subjects)} subjects")
        
        best_params = None
        best_score = -1.0
        results = []
        
        # Create temporary directory for this sweep
        sweep_temp_dir = Path(f"temp_sweep_{random.randint(1000, 9999)}")
        sweep_temp_dir.mkdir(exist_ok=True)
        
        try:
            for i, params in enumerate(param_combinations, 1):
                logger.info(f"   Testing combination {i}/{len(param_combinations)}: {params}")
                
                # Create extraction config for this parameter combination
                extraction_config = self.create_extraction_config(params, subjects, data_dir)
                config_file = sweep_temp_dir / f"config_{i}.json"
                
                with open(config_file, 'w') as f:
                    json.dump(extraction_config, f, indent=2)
                
                # Create output directory for this combination
                output_dir = sweep_temp_dir / f"matrices_{i}"
                output_dir.mkdir(exist_ok=True)
                
                # Run connectivity extraction
                success = self.run_connectivity_extraction(config_file, output_dir, data_dir)
                
                if success:
                    # Evaluate network quality
                    score = self.evaluate_network_quality(output_dir)
                    results.append({
                        'params': params,
                        'score': score
                    })
                    
                    logger.info(f"     → Score: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        logger.info(f"     ✨ New best score: {score:.4f}")
                else:
                    logger.warning(f"     ❌ Failed to extract connectivity for combination {i}")
                
        finally:
            # Cleanup temporary files
            if sweep_temp_dir.exists():
                shutil.rmtree(sweep_temp_dir)
        
        if best_params is None:
            raise RuntimeError("No parameter combinations succeeded")
        
        logger.info(f"✅ Best parameters: {best_params} (score: {best_score:.4f})")
        return best_params, best_score
    
    def create_extraction_config(self, params: Dict[str, Any], subjects: List[str], data_dir: str) -> Dict[str, Any]:
        """Create DSI Studio extraction configuration for given parameters"""
        
        # Get base directory for subjects
        data_path = Path(data_dir)
        subject_files = []
        
        for subject in subjects:
            # Find .fz files for this subject
            fz_files = list(data_path.glob(f"*{subject}*.fz"))
            if fz_files:
                subject_files.append(str(fz_files[0]))
        
        # Create configuration based on sweep_config.json template
        config = {
            "dsi_studio_cmd": "/Applications/dsi_studio.app/Contents/MacOS/dsi_studio",
            "atlases": ["FreeSurferDKT_Cortical"],  # Use single atlas for speed
            "connectivity_values": ["count", "fa"],  # Reduced for speed
            "tract_count": params["track_count"],
            "thread_count": 8,
            "tracking_parameters": {
                "method": 0,
                "otsu_threshold": 0.6,
                "fa_threshold": params["fa_threshold"],
                "turning_angle": params["angular_threshold"],
                "step_size": params["step_size"],
                "smoothing": 0.0,
                "min_length": 10,
                "max_length": 200,
                "track_voxel_ratio": 2.0,
                "check_ending": 0,
                "random_seed": 0,
                "dt_threshold": 0.2
            },
            "connectivity_options": {
                "connectivity_type": "pass",
                "connectivity_threshold": 0.001,
                "connectivity_output": "matrix"
            },
            "subjects": subject_files
        }
        
        return config
    
    def run_connectivity_extraction(self, config_file: Path, output_dir: Path, data_dir: str) -> bool:
        """Run connectivity extraction using the extraction script"""
        
        logger = logging.getLogger(__name__)
        
        try:
            # Use the existing extraction script
            cmd = [
                "python", "scripts/extract_connectivity_matrices.py",
                str(data_dir),
                str(output_dir),
                "--config", str(config_file),
                "--batch"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout per combination
            )
            
            if result.returncode == 0:
                return True
            else:
                logger.warning(f"Extraction failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning("Extraction timed out")
            return False
        except Exception as e:
            logger.warning(f"Extraction error: {e}")
            return False
    
    def evaluate_network_quality(self, matrices_dir: Path) -> float:
        """Evaluate network quality of extracted connectivity matrices"""
        
        # Look for connectivity matrices
        matrix_files = list(matrices_dir.glob("**/*.csv"))
        
        if not matrix_files:
            return 0.0
        
        try:
            scores = []
            
            for matrix_file in matrix_files:
                # Load connectivity matrix
                df = pd.read_csv(matrix_file, index_col=0)
                matrix = df.values
                
                # Calculate basic network metrics
                # 1. Network density (non-zero connections)
                density = (matrix > 0).sum() / (matrix.size - matrix.shape[0])  # Exclude diagonal
                
                # 2. Mean connectivity strength
                mean_strength = matrix[matrix > 0].mean() if (matrix > 0).any() else 0
                
                # 3. Network efficiency (simple proxy)
                # Higher density + higher strength = better network
                efficiency = density * mean_strength
                
                scores.append(efficiency)
            
            # Return mean score across all matrices
            return sum(scores) / len(scores) if scores else 0.0
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Quality evaluation failed: {e}")
            return 0.0
        
    def validate_cross_validation_results(self) -> Dict[str, Any]:
        """Validate that both waves found consistent optimal parameters"""
        
        if len(self.wave_results) != 2:
            logging.error(f"❌ Expected 2 waves, found {len(self.wave_results)}")
            return {"validation_passed": False, "reason": "Incomplete waves"}
            
        wave1_params = self.wave_results[1]["best_parameters"]
        wave2_params = self.wave_results[2]["best_parameters"]
        wave1_score = self.wave_results[1]["best_score"]
        wave2_score = self.wave_results[2]["best_score"]
        
        # Check parameter consistency
        param_matches = []
        param_differences = []
        
        for param_name in wave1_params:
            if param_name in wave2_params:
                if wave1_params[param_name] == wave2_params[param_name]:
                    param_matches.append(param_name)
                else:
                    param_differences.append({
                        "parameter": param_name,
                        "wave1": wave1_params[param_name],
                        "wave2": wave2_params[param_name]
                    })
                    
        # Calculate consistency score
        consistency_score = len(param_matches) / len(wave1_params)
        
        # Calculate performance correlation
        score_diff = abs(wave1_score - wave2_score)
        score_correlation = 1.0 - score_diff  # Simple correlation measure
        
        # Determine validation result
        validation_threshold = self.config["bootstrap_optimization"]["cross_validation"]["validation_threshold"]
        validation_passed = consistency_score >= validation_threshold
        
        validation_results = {
            "validation_passed": validation_passed,
            "consistency_score": consistency_score,
            "score_correlation": score_correlation,
            "parameter_matches": param_matches,
            "parameter_differences": param_differences,
            "wave1_results": {
                "parameters": wave1_params,
                "score": wave1_score,
                "n_subjects": self.wave_results[1]["n_subjects"]
            },
            "wave2_results": {
                "parameters": wave2_params,
                "score": wave2_score,
                "n_subjects": self.wave_results[2]["n_subjects"]
            },
            "validation_threshold": validation_threshold
        }
        
        return validation_results
        
    def display_cross_validation_results(self, validation_results: Dict[str, Any]) -> None:
        """Display cross-validation results for user review"""
        
        print("\n" + "="*80)
        print("🔬 CROSS-VALIDATION BOOTSTRAP PARAMETER OPTIMIZATION")
        print("="*80)
        
        wave1 = validation_results["wave1_results"]
        wave2 = validation_results["wave2_results"]
        
        print(f"\n📊 Wave 1 Results (Subset A - {wave1['n_subjects']} subjects):")
        print(f"   ├─ Best Parameters: {wave1['parameters']}")
        print(f"   └─ Performance Score: {wave1['score']:.3f}")
        
        print(f"\n📊 Wave 2 Results (Subset B - {wave2['n_subjects']} subjects):")
        print(f"   ├─ Best Parameters: {wave2['parameters']}")
        print(f"   └─ Performance Score: {wave2['score']:.3f}")
        
        print(f"\n🎯 CROSS-VALIDATION ASSESSMENT:")
        print(f"   ├─ Parameter Consistency: {validation_results['consistency_score']:.1%}")
        print(f"   ├─ Score Correlation: {validation_results['score_correlation']:.3f}")
        print(f"   └─ Validation Threshold: {validation_results['validation_threshold']:.1%}")
        
        if validation_results["parameter_matches"]:
            print(f"\n✅ Consistent Parameters: {validation_results['parameter_matches']}")
            
        if validation_results["parameter_differences"]:
            print(f"\n⚠️  Parameter Differences:")
            for diff in validation_results["parameter_differences"]:
                print(f"   └─ {diff['parameter']}: Wave1={diff['wave1']}, Wave2={diff['wave2']}")
                
        validation_status = "✅ PASSED" if validation_results["validation_passed"] else "❌ FAILED"
        print(f"\n🎯 CROSS-VALIDATION: {validation_status}")
        
    def get_validated_parameters(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get the validated optimal parameters for full analysis"""
        
        if not validation_results["validation_passed"]:
            print(f"\n⚠️  Cross-validation failed. Manual review recommended.")
            print(f"   Consider adjusting parameters or increasing sample sizes.")
            return None
            
        # Use parameters from the better-performing wave
        wave1_score = validation_results["wave1_results"]["score"]
        wave2_score = validation_results["wave2_results"]["score"]
        
        if wave1_score >= wave2_score:
            optimal_params = validation_results["wave1_results"]["parameters"]
            best_wave = 1
        else:
            optimal_params = validation_results["wave2_results"]["parameters"]
            best_wave = 2
            
        print(f"\n✅ Validated Optimal Parameters (from Wave {best_wave}):")
        for param, value in optimal_params.items():
            print(f"   └─ {param}: {value}")
            
        return optimal_params

def main():
    """Main cross-validation bootstrap optimization workflow"""
    
    parser = argparse.ArgumentParser(
        description="Cross-Validation Bootstrap Parameter Optimizer"
    )
    parser.add_argument("--data-dir", required=True,
                       help="Directory containing input data")
    parser.add_argument("--output-dir", default="results",
                       help="Base output directory")
    parser.add_argument("--config", default="configs/bootstrap_optimization_config.json",
                       help="Bootstrap optimization configuration file")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for reproducible subject splitting")
    
    args = parser.parse_args()
    
    # Set random seed for reproducible results
    random.seed(args.random_seed)
    
    print("🧠 CROSS-VALIDATION BOOTSTRAP PARAMETER OPTIMIZER")
    print("="*60)
    
    # Initialize optimizer
    optimizer = CrossValidationBootstrapOptimizer(args.config)
    
    # Phase 1: Generate parameter combinations
    logging.info("📝 Phase 1: Generating parameter combinations...")
    optimizer.parameter_combinations = optimizer.generate_parameter_combinations()
    
    # Phase 2: Split subjects for cross-validation
    logging.info("🎲 Phase 2: Splitting subjects for cross-validation...")
    subset_a, subset_b = optimizer.split_subjects_for_cross_validation(args.data_dir)
    
    if not subset_a or not subset_b:
        logging.error("❌ Failed to create subject subsets")
        return 1
    
    # Phase 3: Create sweep configurations
    logging.info("📝 Phase 3: Creating parameter sweep configurations...")
    wave_configs = optimizer.create_sweep_configs(args.data_dir, subset_a, subset_b)
    
    # Phase 4: Run cross-validation waves
    logging.info("🌊 Phase 4: Running cross-validation parameter sweeps...")
    
    for i, wave_config in enumerate(wave_configs, 1):
        success, result_dir, best_params = optimizer.run_parameter_sweep_wave(wave_config, i)
        if not success:
            logging.error(f"❌ Cross-validation wave {i} failed")
            return 1
    
    # Phase 5: Validate cross-validation results
    logging.info("🎯 Phase 5: Validating cross-validation results...")
    validation_results = optimizer.validate_cross_validation_results()
    
    # Phase 6: Display results and get validated parameters
    optimizer.display_cross_validation_results(validation_results)
    optimal_params = optimizer.get_validated_parameters(validation_results)
    
    if optimal_params:
        # Create configuration for full analysis
        full_analysis_config = {
            "cross_validation_optimized": True,
            "validation_results": validation_results,
            "optimal_parameters": optimal_params,
            "data_directory": args.data_dir,
            "output_directory": f"{args.output_dir}/full_analysis",
            "bootstrap_reference": {
                "wave_results": optimizer.wave_results,
                "validation_summary": validation_results
            }
        }
        
        config_filename = "cross_validated_optimal_config.json"
        with open(config_filename, 'w') as f:
            json.dump(full_analysis_config, f, indent=2)
            
        print(f"\n📝 Created validated configuration: {config_filename}")
        print(f"🚀 Ready for full analysis with cross-validated parameters!")
        print(f"\n💡 Next step:")
        print(f"   python run_pipeline.py --config {config_filename}")
        
        return 0
    else:
        print(f"\n❌ Cross-validation failed. Review results and adjust parameters.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
