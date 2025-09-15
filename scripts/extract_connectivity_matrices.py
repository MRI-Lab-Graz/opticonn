#!/usr/bin/env python3
"""
DSI Studio Connectivity Matrix Extraction Script

This script extracts connectivity matrices for multiple atlases from DSI Studio fiber files.
It provides batch processing capabilities and detailed logging.

Author: Generated for connectivity analysis
Usage: python extract_connectivity_matrices.py [options] input_file output_dir
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import random
import glob
from typing import List, Optional, Dict, Any
from typing import List, Dict, Optional

# Add scipy for .mat file reading and numpy for array handling
try:
    import scipy.io
    import numpy as np
    MAT_SUPPORT = True
except ImportError:
    MAT_SUPPORT = False
    print("⚠️ Warning: scipy not available - .mat to CSV conversion disabled")
    print("   Install with: pip install scipy")

# Default configuration based on DSI Studio source code analysis
DEFAULT_CONFIG = {
    # Common atlases - Note: Actual availability depends on your DSI Studio installation
    'atlases': [
        'AAL', 'AAL2', 'AAL3', 'Brodmann', 'HCP-MMP', 'AICHA', 
        'Talairach', 'FreeSurferDKT', 'FreeSurferDKT_Cortical', 'Schaefer100', 
        'Schaefer200', 'Schaefer400', 'Gordon333', 'Power264'
    ],
    # All connectivity values from DSI Studio source code
    'connectivity_values': ['count', 'ncount', 'ncount2', 'mean_length', 'qa', 'fa', 'dti_fa', 
                           'md', 'ad', 'rd', 'iso', 'rdi', 'ndi', 'dti_ad', 'dti_rd', 
                           'dti_md', 'trk'],
    'track_count': 100000,
    'thread_count': 8,
    'dsi_studio_cmd': 'dsi_studio',
    # Tracking parameters from source code analysis
    'tracking_parameters': {
        'method': 0,  # 0=streamline(Euler), 1=RK4, 2=voxel tracking
        'otsu_threshold': 0.6,  # Default Otsu threshold
        'fa_threshold': 0.0,  # FA threshold (0=automatic)
        'turning_angle': 0.0,  # Maximum turning angle (0=random 15-90°)
        'step_size': 0.0,  # Step size in mm (0=random 1-3 voxels)
        'smoothing': 0.0,  # Fraction of previous direction (0-1)
        'min_length': 0,  # Minimum fiber length (0=dataset specific)
        'max_length': 0,  # Maximum fiber length (0=dataset specific)
        'track_voxel_ratio': 2.0,  # Seeds-per-voxel ratio
        'check_ending': 0,  # Drop tracks not terminating in ROI (0=off, 1=on)
        'random_seed': 0,  # Random seed for tracking
        'dt_threshold': 0.2  # Differential tracking threshold
    },
    'connectivity_options': {
        'connectivity_type': 'pass',  # 'pass' or 'end'
        'connectivity_threshold': 0.001,  # Threshold for connectivity matrix
        'connectivity_output': 'matrix,connectogram,measure'  # Output types
    }
}

class ConnectivityExtractor:
    """Main class for extracting connectivity matrices from DSI Studio."""
    
    def __init__(self, config: Dict = None):
        """Initialize the extractor with configuration."""
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        # Verbosity flags (quiet by default unless explicitly disabled)
        self.quiet: bool = bool(self.config.get('quiet', True))
        self.debug_dsi: bool = bool(self.config.get('debug_dsi', False))
        self.setup_logging()
    
    def find_fib_files(self, input_folder: str, pattern: str = "*.fib.gz") -> List[str]:
        """
        Find all fiber files in a folder, supporting both .fib.gz and .fz extensions.
        
        Parameters:
        -----------
        input_folder : str
            Path to folder containing fiber files
        pattern : str
            File pattern to match (default: *.fib.gz)
            
        Returns:
        --------
        List[str]
            List of found fiber files
        """
        # Enhanced patterns to catch both .fz and .fib.gz files
        base_patterns = []
        
        if pattern == "*.fib.gz":
            # If default pattern, search for both extensions
            base_patterns = ["*.fib.gz", "*.fz"]
        else:
            # Use provided pattern, but also try .fz variant
            base_patterns = [pattern]
            if not pattern.endswith(".fz"):
                fz_pattern = pattern.replace(".fib.gz", ".fz")
                base_patterns.append(fz_pattern)
        
        # Create comprehensive search patterns
        search_patterns = []
        for base_pattern in base_patterns:
            # Direct search in folder
            search_patterns.append(os.path.join(input_folder, base_pattern))
            # Recursive search
            search_patterns.append(os.path.join(input_folder, "**", base_pattern))
        
        all_files = []
        for search_pattern in search_patterns:
            files = glob.glob(search_pattern, recursive=True)
            all_files.extend(files)
        
        # Remove duplicates and sort
        unique_files = sorted(list(set(all_files)))
        
        # Categorize files by type
        fz_files = [f for f in unique_files if f.endswith('.fz')]
        fib_gz_files = [f for f in unique_files if f.endswith('.fib.gz')]
        
        self.logger.info(f"Found {len(unique_files)} fiber files in {input_folder}")
        if fz_files:
            self.logger.info(f"  - {len(fz_files)} .fz files")
        if fib_gz_files:
            self.logger.info(f"  - {len(fib_gz_files)} .fib.gz files")
            
        # Show first few files as examples
        for i, file in enumerate(unique_files[:5]):
            self.logger.info(f"    {i+1}. {os.path.basename(file)}")
        if len(unique_files) > 5:
            self.logger.info(f"    ... and {len(unique_files) - 5} more")
            
        return unique_files
    
    def select_pilot_files(self, file_list: List[str], pilot_count: int = 1) -> List[str]:
        """
        Select random files for pilot testing.
        
        Parameters:
        -----------
        file_list : List[str]
            List of all available files
        pilot_count : int
            Number of files to select for pilot (default: 1)
            
        Returns:
        --------
        List[str]
            List of selected pilot files
        """
        if not file_list:
            self.logger.warning("No files available for pilot selection")
            return []
        
        if pilot_count >= len(file_list):
            self.logger.info(f"Pilot count ({pilot_count}) >= available files ({len(file_list)}), using all files")
            return file_list
        
        pilot_files = random.sample(file_list, pilot_count)
        
        self.logger.info(f"Selected {len(pilot_files)} pilot files:")
        for file in pilot_files:
            self.logger.info(f"  - {os.path.basename(file)}")
            
        return pilot_files
        
    def setup_logging(self):
        """Set up console logging immediately; attach file logging once run dir is known.
        Console: no timestamp; File: with timestamps.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # Clear duplicate handlers on re-init
        logger.handlers.clear()
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(ch)
        self.logger = logger
        
        # Log session header (concise in quiet mode)
        if not self.quiet:
            self.logger.info("=" * 60)
            self.logger.info("🧠 DSI STUDIO CONNECTIVITY EXTRACTION SESSION START")
        # Try to get and log DSI Studio version early (skip in quiet unless debug is enabled)
        dsi_check = self.check_dsi_studio()
        if not self.quiet or self.debug_dsi:
            if dsi_check['available'] and dsi_check['version']:
                self.logger.info(f"🔧 DSI Studio Version: {dsi_check['version']}")
            self.logger.info(f"📁 DSI Studio Path: {dsi_check['path']}")
            self.logger.info(f"📅 Session Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if not self.quiet:
                self.logger.info("=" * 60)
        else:
            # Minimal one-liner when quiet
            self.logger.info("Starting extraction...")

    def _attach_file_logger(self, log_dir: Path) -> Path:
        """Attach a file handler that writes into log_dir. Returns the log file path."""
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'connectivity_extraction_{timestamp}.log'
        # Avoid duplicate handlers
        existing_files = {Path(getattr(h, 'baseFilename')).resolve() for h in self.logger.handlers
                          if isinstance(h, logging.FileHandler) and hasattr(h, 'baseFilename')}
        if log_file.resolve() not in existing_files:
            fh = logging.FileHandler(str(log_file))
            # Capture full details in file logs
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(fh)
            self.logger.info(f"📄 Session log file: {log_file}")
        return log_file
    
    def check_dsi_studio(self) -> Dict[str, Any]:
        """Check if DSI Studio is available and working properly."""
        dsi_cmd = self.config['dsi_studio_cmd']
        result = {
            'available': False,
            'path': dsi_cmd,
            'version': None,
            'error': None
        }
        
        # Check if file exists (for absolute paths)
        if os.path.isabs(dsi_cmd):
            if not os.path.exists(dsi_cmd):
                result['error'] = f"DSI Studio executable not found at: {dsi_cmd}"
                return result
            if not os.access(dsi_cmd, os.X_OK):
                result['error'] = f"DSI Studio executable is not executable: {dsi_cmd}"
                return result
        
        # Test execution with --version (avoids GUI launch)
        try:
            version_result = subprocess.run(
                [dsi_cmd, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if version_result.returncode == 0:
                result['available'] = True
                # Extract version info
                if version_result.stdout:
                    result['version'] = version_result.stdout.strip()
                elif version_result.stderr:
                    # Some versions output to stderr
                    result['version'] = version_result.stderr.strip()
                else:
                    result['version'] = "Version detected but no output"
            else:
                # If --version fails, try --help as fallback (but with shorter timeout)
                try:
                    help_result = subprocess.run(
                        [dsi_cmd, '--help'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if help_result.returncode == 0:
                        result['available'] = True
                        result['version'] = "Version unknown (--help works)"
                    else:
                        result['error'] = f"DSI Studio returned error code {version_result.returncode}"
                except subprocess.TimeoutExpired:
                    result['error'] = "DSI Studio --help command timed out (GUI launch issue?)"
                except Exception as e:
                    result['error'] = f"Error testing DSI Studio --help: {str(e)}"
                
        except subprocess.TimeoutExpired:
            result['error'] = "DSI Studio --version command timed out"
        except FileNotFoundError:
            result['error'] = f"DSI Studio command not found: {dsi_cmd}. Check PATH or use absolute path."
        except Exception as e:
            result['error'] = f"Error running DSI Studio --version: {str(e)}"
            
        return result
    
    def validate_input_file(self, filepath: str) -> bool:
        """Validate the input fiber file."""
        path = Path(filepath)
        if not path.exists():
            self.logger.error(f"Input file does not exist: {filepath}")
            return False
        
        if not (filepath.endswith('.fib.gz') or filepath.endswith('.fz')):
            self.logger.warning(f"Input file should be .fib.gz or .fz: {filepath}")
        
        return True
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Comprehensive validation of configuration and environment."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # 1. Check DSI Studio
        self.logger.info("🔍 Checking DSI Studio availability...")
        dsi_check = self.check_dsi_studio()
        if not dsi_check['available']:
            validation_result['errors'].append(f"DSI Studio check failed: {dsi_check['error']}")
            validation_result['valid'] = False
        else:
            msg = f"✅ DSI Studio found at: {dsi_check['path']}"
            if dsi_check['version']:
                msg += f" (Version: {dsi_check['version']})"
            validation_result['info'].append(msg)
            self.logger.info(msg)
        
        # 2. Validate atlases
        atlases = self.config.get('atlases', [])
        if not atlases:
            validation_result['warnings'].append("No atlases specified")
        else:
            self.logger.info(f"📊 Will process {len(atlases)} atlases: {', '.join(atlases)}")
            validation_result['info'].append(f"Configured atlases: {', '.join(atlases)}")
        
        # 3. Validate connectivity values
        conn_values = self.config.get('connectivity_values', [])
        if not conn_values:
            validation_result['warnings'].append("No connectivity values specified")
        else:
            self.logger.info(f"📊 Will extract {len(conn_values)} connectivity metrics: {', '.join(conn_values)}")
            validation_result['info'].append(f"Connectivity metrics: {', '.join(conn_values)}")
        
        # 4. Check tracking parameters for reasonable values
        tracking_params = self.config.get('tracking_parameters', {})
        track_count = self.config.get('track_count', 100000)
        
        if track_count <= 0:
            validation_result['errors'].append(f"Track count must be positive, got: {track_count}")
            validation_result['valid'] = False
        elif track_count < 1000:
            validation_result['warnings'].append(f"Low track count ({track_count}), results may be sparse")
        elif track_count > 1000000:
            validation_result['warnings'].append(f"Very high track count ({track_count}), processing may be slow")
        
        # Check FA threshold
        fa_threshold = tracking_params.get('fa_threshold', 0.0)
        if fa_threshold < 0 or fa_threshold > 1:
            validation_result['warnings'].append(f"FA threshold {fa_threshold} outside normal range [0-1]")
        
        # Check turning angle
        turning_angle = tracking_params.get('turning_angle', 0.0)
        if turning_angle > 180:
            validation_result['warnings'].append(f"Turning angle {turning_angle}° seems too large")
        
        # 5. Check thread count
        thread_count = self.config.get('thread_count', 8)
        if thread_count <= 0:
            validation_result['errors'].append(f"Thread count must be positive, got: {thread_count}")
            validation_result['valid'] = False
        elif thread_count > 32:
            validation_result['warnings'].append(f"Very high thread count ({thread_count}), may exceed system capacity")
        
        # Summary
        if validation_result['valid']:
            self.logger.info("✅ Configuration validation passed")
        else:
            self.logger.error("❌ Configuration validation failed")
        
        if validation_result['warnings']:
            self.logger.warning(f"⚠️  {len(validation_result['warnings'])} warning(s) found")
            for warning in validation_result['warnings']:
                self.logger.warning(f"   - {warning}")
        
        if validation_result['errors']:
            self.logger.error(f"❌ {len(validation_result['errors'])} error(s) found")
            for error in validation_result['errors']:
                self.logger.error(f"   - {error}")
        
        return validation_result
    
    def validate_input_path(self, input_path: str, file_pattern: str = "*.fib.gz") -> Dict[str, Any]:
        """Validate input path and find fiber files (runtime validation)."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': [],
            'files_found': []
        }
        
        if not os.path.exists(input_path):
            validation_result['errors'].append(f"Input path does not exist: {input_path}")
            validation_result['valid'] = False
            return validation_result
        
        if os.path.isfile(input_path):
            # Single file processing
            if not (input_path.endswith('.fib.gz') or input_path.endswith('.fz')):
                validation_result['warnings'].append(f"File extension should be .fib.gz or .fz: {input_path}")
            validation_result['files_found'] = [input_path]
            validation_result['info'].append(f"Single file mode: {os.path.basename(input_path)}")
            
        elif os.path.isdir(input_path):
            # Directory processing
            self.logger.info(f"🔍 Scanning directory: {input_path}")
            files_found = self.find_fib_files(input_path, file_pattern)
            
            if not files_found:
                validation_result['errors'].append(f"No fiber files found in directory: {input_path}")
                validation_result['valid'] = False
            else:
                validation_result['files_found'] = files_found
                # File info is logged by find_fib_files method
                
        else:
            validation_result['errors'].append(f"Input path is neither file nor directory: {input_path}")
            validation_result['valid'] = False
        
        return validation_result
    
    def create_output_structure(self, output_dir: str, base_name: str) -> Path:
        """Create organized output directory structure based on settings."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create parameter-based directory name for better organization
        tracking_params = self.config.get('tracking_parameters', {})
        method_name = {0: 'streamline', 1: 'rk4', 2: 'voxel'}.get(tracking_params.get('method', 0), 'streamline')
        track_count = self.config.get('track_count', 100000)
        
        # Create meaningful directory structure
        param_dir = f"tracks_{track_count//1000}k_{method_name}"
        if tracking_params.get('turning_angle', 0) != 0:
            param_dir += f"_angle{int(tracking_params['turning_angle'])}"
        if tracking_params.get('fa_threshold', 0) != 0:
            param_dir += f"_fa{tracking_params['fa_threshold']:.2f}"
            
        run_dir = Path(output_dir) / f"{base_name}_{timestamp}" / param_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create simplified structure - only combined directory 
        # (by_atlas and by_metric are redundant and cause 3x duplication)
        (run_dir / "results").mkdir(exist_ok=True)  # Main results directory
        logs_dir = (run_dir / "logs")
        logs_dir.mkdir(exist_ok=True)
        # Attach per-run file logger to write logs into the run's logs folder
        try:
            self._attach_file_logger(logs_dir)
        except Exception as e:
            # Do not fail the run if log file attachment fails; console logging remains
            self.logger.warning(f"Could not attach file logger in {logs_dir}: {e}")
        
        return run_dir
    
    def extract_connectivity_matrix(self, input_file: str, output_dir: Path, 
                                  atlas: str, base_name: str) -> Dict:
        """Extract connectivity matrix for a specific atlas."""
        if self.quiet:
            self.logger.info(f"[Atlas] {atlas} → running…")
        else:
            self.logger.info(f"Processing atlas: {atlas}")
        
        # Use new simplified results structure  
        atlas_dir = output_dir / "results" / atlas
        atlas_dir.mkdir(parents=True, exist_ok=True)
        output_prefix = atlas_dir / f"{base_name}_{atlas}"
        
        # Build DSI Studio command with comprehensive parameters
        cmd = [
            self.config['dsi_studio_cmd'],
            '--action=trk',
            f'--source={input_file}',
            f'--tract_count={self.config["track_count"]}',
            f'--connectivity={atlas}',
            f'--connectivity_value={",".join(self.config["connectivity_values"])}',
            f'--connectivity_type={self.config["connectivity_options"]["connectivity_type"]}',
            f'--connectivity_threshold={self.config["connectivity_options"]["connectivity_threshold"]}',
            f'--connectivity_output={self.config["connectivity_options"]["connectivity_output"]}',
            f'--thread_count={self.config["thread_count"]}',
            f'--output={output_prefix}.tt.gz',
            '--export=stat'
        ]
        
        # Add tracking parameters if they differ from defaults
        tracking_params = self.config.get('tracking_parameters', {})
        if tracking_params.get('method', 0) != 0:
            cmd.append(f'--method={tracking_params["method"]}')
        if tracking_params.get('otsu_threshold', 0.6) != 0.6:
            cmd.append(f'--otsu_threshold={tracking_params["otsu_threshold"]}')
        if tracking_params.get('fa_threshold', 0.0) != 0.0:
            cmd.append(f'--fa_threshold={tracking_params["fa_threshold"]}')
        if tracking_params.get('turning_angle', 0.0) != 0.0:
            cmd.append(f'--turning_angle={tracking_params["turning_angle"]}')
        if tracking_params.get('step_size', 0.0) != 0.0:
            cmd.append(f'--step_size={tracking_params["step_size"]}')
        if tracking_params.get('smoothing', 0.0) != 0.0:
            cmd.append(f'--smoothing={tracking_params["smoothing"]}')
        if tracking_params.get('min_length', 0) != 0:
            cmd.append(f'--min_length={tracking_params["min_length"]}')
        if tracking_params.get('max_length', 0) != 0:
            cmd.append(f'--max_length={tracking_params["max_length"]}')
        if tracking_params.get('track_voxel_ratio', 2.0) != 2.0:
            cmd.append(f'--track_voxel_ratio={tracking_params["track_voxel_ratio"]}')
        if tracking_params.get('check_ending', 0) != 0:
            cmd.append(f'--check_ending={tracking_params["check_ending"]}')
        if tracking_params.get('random_seed', 0) != 0:
            cmd.append(f'--random_seed={tracking_params["random_seed"]}')
            
        # DSI Studio command: print only in debug mode to keep console concise
        if self.debug_dsi:
            self.logger.info(f"DSI Studio command: {' '.join(str(c) for c in cmd)}")
        else:
            self.logger.debug(f"DSI Studio command: {' '.join(str(c) for c in cmd)}")
        # Execute command
        start_time = datetime.now()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            success = result.returncode == 0
            if success:
                if self.quiet:
                    self.logger.info(f"[Atlas] {atlas} → done in {duration:.1f}s")
                else:
                    self.logger.info(f"✓ Successfully processed {atlas} in {duration:.1f}s")
                # Organize output files by metric type
                self._organize_output_files(output_dir, atlas, base_name)
            else:
                self.logger.error(f"✗ Failed to process {atlas}")
                self.logger.error(f"Error output: {result.stderr}")
            return {
                'atlas': atlas,
                'success': success,
                'duration': duration,
                'command': ' '.join(cmd),
                'stdout': result.stdout,
                'stderr': result.stderr,
                'output_files': [str(f) for f in atlas_dir.glob(f"{base_name}_{atlas}*")]
            }
        except subprocess.TimeoutExpired:
            self.logger.error(f"✗ Timeout while processing {atlas}")
            return {
                'atlas': atlas,
                'success': False,
                'duration': 3600,
                'error': 'Timeout'
            }
    
    def _organize_output_files(self, output_dir: Path, atlas: str, base_name: str):
        """Check and process files in the new DSI Studio results structure."""
        # Files are already created in results/atlas/ by DSI Studio
        results_dir = output_dir / "results"
        atlas_results_dir = results_dir / atlas
        
        if not atlas_results_dir.exists():
            self.logger.warning(f"📂 Atlas directory not found: {atlas_results_dir}")
            return
        
        # Count existing files in the atlas directory
        atlas_files = list(atlas_results_dir.glob("*"))
        connectogram_files = list(atlas_results_dir.glob("*.connectogram.txt"))
        
        self.logger.info(f"📂 Found {len(atlas_files)} files for atlas '{atlas}' in results directory")
        
        # Process connectogram files for enhanced CSV conversion
        enhanced_count = 0
        for connectogram_file in connectogram_files:
            try:
                csv_file = self._convert_connectogram_to_csv(connectogram_file)
                if csv_file and csv_file.exists():
                    enhanced_count += 1
                    self.logger.debug(f"  ✓ Enhanced conversion: {connectogram_file.name} → CSV with anatomical names")
            except Exception as e:
                self.logger.warning(f"  ⚠️  Failed to convert {connectogram_file.name}: {e}")
        
        self.logger.info(f"✅ Processed {len(atlas_files)} files for atlas '{atlas}' ({enhanced_count} enhanced conversions)")
    
    def _convert_connectogram_to_csv(self, connectogram_file: Path) -> Path:
        """Convert DSI Studio connectogram.txt file to CSV format."""
        try:
            # Read the connectogram file
            with open(connectogram_file, 'r') as f:
                content = f.read().strip()
            
            if not content:
                raise ValueError("Connectogram file is empty")
            
            # DSI Studio connectogram format analysis:
            # The file might be in different formats - let's detect the actual format
            lines = content.split('\n')
            
            # Try to detect the format by checking first few lines
            if len(lines) < 2:
                raise ValueError("Connectogram file has insufficient data")
            
            # Check if it's a connectivity matrix (space-separated numbers)
            try:
                # Try to parse first line as numbers
                first_line_values = lines[0].strip().split()
                # If all values are numeric, it's likely a matrix format
                numeric_values = [float(val) for val in first_line_values]
                
                # It's a matrix format - parse as connectivity matrix
                import pandas as pd
                import numpy as np
                
                matrix_data = []
                for line in lines:
                    if line.strip():
                        row = [float(val) for val in line.strip().split()]
                        matrix_data.append(row)
                
                # Convert to numpy array
                connectivity_matrix = np.array(matrix_data)
                
                # Create DataFrame with generic region names
                n_regions = connectivity_matrix.shape[0]
                region_names = [f"region_{i+1}" for i in range(n_regions)]
                
                df = pd.DataFrame(connectivity_matrix, 
                                index=region_names,
                                columns=region_names[:connectivity_matrix.shape[1]])
                
                # Save CSV
                csv_file = connectogram_file.with_suffix('.csv')
                df.to_csv(csv_file, index=True)
                
                return csv_file
                
            except ValueError:
                # Not a simple matrix format - might be edge list or other format
                # Try to parse as edge list (region1 region2 value)
                import pandas as pd
                
                edges = []
                for line in lines:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            try:
                                # Try to extract: source, target, weight
                                source = parts[0]
                                target = parts[1] 
                                weight = float(parts[2])
                                edges.append({'source_region': source, 'target_region': target, 'connectivity_value': weight})
                            except ValueError:
                                # Skip lines that can't be parsed
                                continue
                
                if edges:
                    # Save as edge list CSV
                    df = pd.DataFrame(edges)
                    csv_file = connectogram_file.with_suffix('.csv')
                    df.to_csv(csv_file, index=False)
                    return csv_file
                else:
                    # Try as simple text file - just save as CSV with single column
                    df = pd.DataFrame({'content': lines})
                    csv_file = connectogram_file.with_suffix('.csv')
                    df.to_csv(csv_file, index=False)
                    return csv_file
            
        except Exception as e:
            self.logger.error(f"Failed to convert connectogram {connectogram_file.name}: {e}")
            return None
    
    def extract_all_matrices(self, input_file: str, output_dir: str, 
                           atlases: List[str] = None) -> Dict:
        """Extract connectivity matrices for all specified atlases."""
        # Run comprehensive validation first
        self.logger.info("🚀 Starting connectivity matrix extraction...")
        self.logger.info("=" * 60)
        
        validation_result = self.validate_configuration()
        if not validation_result['valid']:
            raise RuntimeError(f"Configuration validation failed: {validation_result['errors']}")
        
        # Additional DSI Studio check with detailed info
        dsi_check = self.check_dsi_studio()
        if not dsi_check['available']:
            raise RuntimeError(f"DSI Studio not available: {dsi_check['error']}")
        
        if not self.validate_input_file(input_file):
            raise ValueError(f"Invalid input file: {input_file}")
        
        atlases = atlases or self.config['atlases']
        base_name = Path(input_file).stem.replace('.fib', '').replace('.gz', '')
        
        # Create output directory structure
        run_dir = self.create_output_structure(output_dir, base_name)
        
        total_atlases = len(atlases)
        self.logger.info(f"🎯 Starting connectivity extraction for {total_atlases} atlases")
        self.logger.info(f"📁 Input: {input_file}")
        self.logger.info(f"📁 Output: {run_dir}")
        self.logger.info(f"🧠 DSI Studio: {dsi_check['path']}")
        if dsi_check['version']:
            self.logger.info(f"📊 Version: {dsi_check['version']}")
        self.logger.info("=" * 60)
        
        # Process each atlas
        results = []
        for idx, atlas in enumerate(atlases, start=1):
            if self.quiet:
                self.logger.info(f"[Atlas {idx}/{total_atlases}] {atlas} → queued")
            result = self.extract_connectivity_matrix(input_file, run_dir, atlas, base_name)
            results.append(result)
        
        # Save processing summary in logs directory
        dsi_check = self.check_dsi_studio()
        summary = {
            'input_file': input_file,
            'output_directory': str(run_dir),
            'timestamp': datetime.now().isoformat(),
            'dsi_studio': {
                'path': dsi_check['path'],
                'version': dsi_check.get('version', 'Unknown'),
                'available': dsi_check['available']
            },
            'config': self.config,
            'results': results,
            'summary': {
                'total_atlases': len(atlases),
                'successful': sum(1 for r in results if r.get('success', False)),
                'failed': sum(1 for r in results if not r.get('success', False)),
                'total_duration': sum(r.get('duration', 0) for r in results)
            }
        }
        
        # Save files in logs directory
        logs_dir = run_dir / "logs"
        summary_file = logs_dir / 'extraction_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create results CSV in logs directory
        results_df = pd.DataFrame([
            {
                'atlas': r['atlas'],
                'success': r.get('success', False),
                'duration_seconds': r.get('duration', 0),
                'error': r.get('error', '')
            }
            for r in results
        ])
        results_df.to_csv(logs_dir / 'processing_results.csv', index=False)
        
        # Create analysis-ready summary files
        self._create_analysis_summary(run_dir, base_name, results)
        
        # Convert DSI Studio outputs to CSV format (check config)
        convert_to_csv = True  # Default to True for user convenience
        
        # Check if CSV conversion is configured
        if 'convert_to_csv' in self.config.get('connectivity_options', {}):
            convert_to_csv = self.config['connectivity_options']['convert_to_csv']
        
        if convert_to_csv:
            self.logger.info("🔄 Converting all DSI Studio outputs to CSV format...")
            csv_conversion = self.convert_all_outputs_to_csv(run_dir)
            summary['csv_conversion'] = csv_conversion
        else:
            self.logger.info("⏭️ Skipping CSV conversion (disabled)")
            csv_conversion = {'success': True, 'total_converted': 0, 'skipped': True}
            summary['csv_conversion'] = csv_conversion
        
        self.logger.info(f"Extraction completed: {summary['summary']['successful']}/{summary['summary']['total_atlases']} successful")
        if csv_conversion.get('success') and csv_conversion.get('total_converted', 0) > 0:
            total_converted = csv_conversion['total_converted']
            self.logger.info(f"📊 CSV files generated: {total_converted} files converted (.mat, .connectogram.txt, .network_measures.txt)")
        
        return summary

    def _create_analysis_summary(self, run_dir: Path, base_name: str, results: List[Dict]):
        """Create analysis-ready summary files and directory structure overview."""
        combined_dir = run_dir / "combined"
        
        # Create directory structure README
        readme_content = f"""# Connectivity Analysis Results for {base_name}

## 🎯 Simplified Directory Structure

📁 **results/** - All connectivity outputs organized by brain atlas
   ├── **Cerebellum-SUIT/** - Cerebellar regions (SUIT atlas)
   ├── **FreeSurferDKT_Cortical/** - Cortical regions (Desikan-Killiany-Tourville)
   ├── **FreeSurferDKT_Subcortical/** - Subcortical structures
   ├── **FreeSurferDKT_Tissue/** - Tissue segmentation
   └── **FreeSurferSeg/** - Full FreeSurfer segmentation
   
📁 **logs/** - Processing logs and summaries
   └── extraction_summary.json, processing_results.csv, connectivity_extraction.log

## 📊 Enhanced Output Files 

🔢 **Connectivity Matrices** (.mat → .csv)
   - **Enhanced CSV**: *.connectivity.csv (with anatomical region names as headers/indices)
   - **Simple CSV**: *.connectivity.simple.csv (numbers only for computational analysis)
   - **MATLAB**: *.connectivity.mat (original DSI Studio format)

🌐 **Enhanced Connectograms** (.connectogram.txt → .csv)
   - **Full Matrix**: *.connectogram.csv (connectivity matrix with anatomical names)
   - **Region Info**: *.connectogram.region_info.csv (streamline counts + anatomical names)
   - **Original**: *.connectogram.txt (original DSI Studio format)

📊 **Network Measures** (.network_measures.txt → .csv)
   - Graph-theoretic measures (clustering, path length, efficiency, etc.)
   - Pre-calculated network statistics for each atlas
   - Ready for statistical analysis

## Quick Analysis Commands

### Load connectivity matrices:
```python
import pandas as pd
import numpy as np

# Load CSV matrix (easier)
matrix_df = pd.read_csv('by_atlas/AAL3/subject_AAL3.connectivity.csv', index_col=0)

# Load simple CSV as numpy array  
matrix = np.loadtxt('by_atlas/AAL3/subject_AAL3.connectivity.simple.csv', delimiter=',')
```

### Load connectogram (edge list):
```python
# Load connectogram for network analysis
edges_df = pd.read_csv('by_atlas/AAL3/subject_AAL3.connectogram.csv')
print(f"Found {{len(edges_df)}} connections")

# Convert to NetworkX graph
import networkx as nx
G = nx.from_pandas_edgelist(edges_df, 
                           source='source_region', 
                           target='target_region', 
                           edge_attr='connectivity_value')
```

### Load network measures:
```python
# Load pre-calculated network statistics
measures_df = pd.read_csv('by_atlas/AAL3/subject_AAL3.network_measures.csv')
print("Available measures:", measures_df.columns.tolist())
```

### Compare atlases for same metric:
```python
# Compare AAL2 vs HCP-MMP for FA metric
aal2_fa = scipy.io.loadmat('by_atlas/AAL2/*fa*.connectivity.mat')
hcp_fa = scipy.io.loadmat('by_atlas/HCP-MMP/*fa*.connectivity.mat')
```

### Batch load all results:
```python
combined_files = glob.glob('combined/*.connectivity.mat')
all_matrices = {{f.split('/')[-1]: scipy.io.loadmat(f) for f in combined_files}}
```

## Processing Summary

"""
        
        # Add results summary
        successful_atlases = [r['atlas'] for r in results if r.get('success', False)]
        failed_atlases = [r['atlas'] for r in results if not r.get('success', False)]
        
        readme_content += f"✅ **Successfully processed**: {', '.join(successful_atlases)}\n"
        if failed_atlases:
            readme_content += f"❌ **Failed**: {', '.join(failed_atlases)}\n"
        
        readme_content += f"\n📊 **Total matrices generated**: ~{len(successful_atlases) * len(self.config['connectivity_values'])}\n"
        
        # Write README
        with open(run_dir / "README.md", 'w') as f:
            f.write(readme_content)
            
        # Create a quick analysis starter script
        analysis_script = f'''#!/usr/bin/env python3
"""
Quick analysis starter script for connectivity matrices
Generated for: {base_name}
"""

import glob
import numpy as np
import pandas as pd
import scipy.io
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent
ATLASES = {repr(self.config['atlases'])}
METRICS = {repr(self.config['connectivity_values'])}

def load_connectivity_matrix(atlas, metric):
    """Load connectivity matrix for specific atlas and metric."""
    pattern = f"by_atlas/{{atlas}}/*{{metric}}*.connectivity.mat"
    files = list(BASE_DIR.glob(pattern))
    if files:
        return scipy.io.loadmat(files[0])
    return None

def load_all_matrices():
    """Load all connectivity matrices into a nested dictionary."""
    matrices = {{}}
    for atlas in ATLASES:
        matrices[atlas] = {{}}
        for metric in METRICS:
            mat = load_connectivity_matrix(atlas, metric)
            if mat:
                matrices[atlas][metric] = mat
    return matrices

def get_matrix_summary():
    """Get summary statistics for all matrices."""
    summary = []
    for atlas in ATLASES:
        for metric in METRICS:
            mat = load_connectivity_matrix(atlas, metric)
            if mat and 'connectivity' in mat:
                conn = mat['connectivity']
                summary.append({{
                    'atlas': atlas,
                    'metric': metric,
                    'shape': conn.shape,
                    'nonzero_connections': np.count_nonzero(conn),
                    'mean_strength': np.mean(conn[conn > 0]),
                    'density': np.count_nonzero(conn) / (conn.shape[0] * conn.shape[1])
                }})
    return pd.DataFrame(summary)

if __name__ == "__main__":
    print("Loading connectivity matrices...")
    matrices = load_all_matrices()
    
    print("\\nGenerating summary...")
    summary_df = get_matrix_summary()
    print(summary_df)
    
    print("\\nSaving summary to CSV...")
    summary_df.to_csv("analysis_summary.csv", index=False)
    
    print(f"\\nAnalysis complete! Found matrices for {{len(summary_df)}} atlas-metric combinations.")
'''
        
        with open(run_dir / "quick_analysis.py", 'w') as f:
            f.write(analysis_script)
            
        # Make analysis script executable
        import stat
        analysis_script_path = run_dir / "quick_analysis.py"
        analysis_script_path.chmod(analysis_script_path.stat().st_mode | stat.S_IEXEC)
    
    def convert_mat_to_csv(self, mat_file_path: Path, atlas: str) -> Dict[str, str]:
        """Convert .mat connectivity matrix to CSV format.
        
        Parameters:
        -----------
        mat_file_path : Path
            Path to the .mat file
        atlas : str
            Atlas name for informative naming
            
        Returns:
        --------
        Dict[str, str]
            Dictionary with conversion results and output paths
        """
        if not MAT_SUPPORT:
            return {'success': False, 'error': 'scipy not available for .mat conversion'}
            
        try:
            # Load .mat file
            mat_data = scipy.io.loadmat(str(mat_file_path))
            
            # Find the connectivity matrix (common keys: 'connectivity', 'matrix', 'data')
            connectivity_key = None
            for key in ['connectivity', 'matrix', 'data']:
                if key in mat_data:
                    connectivity_key = key
                    break
            
            if connectivity_key is None:
                # List available keys for debugging
                available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                self.logger.warning(f"No standard connectivity key found in {mat_file_path.name}")
                self.logger.warning(f"Available keys: {available_keys}")
                # Use the first non-metadata key
                if available_keys:
                    connectivity_key = available_keys[0]
                else:
                    return {'success': False, 'error': 'No data found in .mat file'}
            
            connectivity_matrix = mat_data[connectivity_key]
            
            # Convert to DataFrame for better CSV output
            if connectivity_matrix.ndim == 2:
                # Create meaningful row/column names if available
                if 'labels' in mat_data:
                    labels = [str(label[0]) if hasattr(label, '__getitem__') else str(label) 
                             for label in mat_data['labels'].flatten()]
                    if len(labels) == connectivity_matrix.shape[0]:
                        df = pd.DataFrame(connectivity_matrix, index=labels, columns=labels)
                    else:
                        df = pd.DataFrame(connectivity_matrix)
                else:
                    # Use generic row/column names
                    n_regions = connectivity_matrix.shape[0]
                    region_names = [f'{atlas}_region_{i+1:03d}' for i in range(n_regions)]
                    df = pd.DataFrame(connectivity_matrix, index=region_names, columns=region_names)
                
                # Save as CSV
                csv_path = mat_file_path.with_suffix('.csv')
                df.to_csv(csv_path, index=True)
                
                # Also save a simplified version without row names for easy loading
                simple_csv_path = mat_file_path.with_suffix('.simple.csv')
                np.savetxt(simple_csv_path, connectivity_matrix, delimiter=',', fmt='%.6f')
                
                return {
                    'success': True,
                    'csv_path': str(csv_path),
                    'simple_csv_path': str(simple_csv_path),
                    'matrix_shape': connectivity_matrix.shape,
                    'connectivity_key': connectivity_key
                }
            else:
                return {'success': False, 'error': f'Unexpected matrix dimensions: {connectivity_matrix.shape}'}
                
        except Exception as e:
            self.logger.error(f"Failed to convert {mat_file_path.name} to CSV: {str(e)}")
            return {'success': False, 'error': str(e)}

    def convert_all_mats_to_csv(self, output_dir: Path) -> Dict[str, Any]:
        """Convert all .mat files in output directory to CSV format.
        
        Parameters:
        -----------
        output_dir : Path
            Output directory containing .mat files
            
        Returns:
        --------
        Dict[str, Any]
            Summary of .mat conversion results
        """
        # Find all .mat files
        mat_files = list(output_dir.rglob('*.mat'))
        
        if not mat_files:
            self.logger.info("No .mat files found for conversion")
            return {'success': True, 'converted': 0, 'files': []}
        
        self.logger.info(f"Converting {len(mat_files)} .mat files...")
        
        conversion_results = []
        successful_conversions = 0
        
        for mat_file in mat_files:
            # Extract atlas name from file path for better naming
            atlas = 'unknown'
            if 'by_atlas' in str(mat_file):
                parts = str(mat_file).split('by_atlas')
                if len(parts) > 1:
                    atlas_part = parts[1].strip('/').split('/')[0]
                    if atlas_part:
                        atlas = atlas_part
            
            result = self.convert_mat_to_csv(mat_file, atlas)
            
            if result.get('success'):
                successful_conversions += 1
                conversion_results.append(result)
            else:
                self.logger.warning(f"Failed to convert {mat_file.name}: {result.get('error', 'Unknown error')}")
                conversion_results.append(result)
        
        return {
            'success': True,
            'total_files': len(mat_files),
            'converted': successful_conversions,
            'failed': len(mat_files) - successful_conversions,
            'files': conversion_results
        }

    def convert_all_outputs_to_csv(self, output_dir: Path) -> Dict[str, Any]:
        """Convert all DSI Studio outputs (.mat, .connectogram.txt, .network_measures.txt) to CSV format.
        
        Parameters:
        -----------
        output_dir : Path
            Output directory containing DSI Studio output files
            
        Returns:
        --------
        Dict[str, Any]
            Summary of conversion results for all file types
        """
        conversion_summary = {
            'success': True,
            'mat_conversion': {},
            'connectogram_conversion': {},
            'measures_conversion': {},
            'total_converted': 0
        }
        
        # Convert .mat files (existing functionality)
        if MAT_SUPPORT:
            self.logger.info("🔄 Converting .mat files to CSV...")
            mat_conversion = self.convert_all_mats_to_csv(output_dir)
            conversion_summary['mat_conversion'] = mat_conversion
            conversion_summary['total_converted'] += mat_conversion.get('converted', 0)
        else:
            self.logger.warning("scipy not available - skipping .mat to CSV conversion")
            conversion_summary['mat_conversion'] = {'success': False, 'error': 'scipy not available'}
        
        # Convert .connectogram.txt files
        self.logger.info("🔄 Converting .connectogram.txt files...")
        connectogram_conversion = self.convert_connectogram_files(output_dir)
        conversion_summary['connectogram_conversion'] = connectogram_conversion
        conversion_summary['total_converted'] += connectogram_conversion.get('converted', 0)
        
        # Convert .network_measures.txt files  
        self.logger.info("🔄 Converting .network_measures.txt files...")
        measures_conversion = self.convert_measures_files(output_dir)
        conversion_summary['measures_conversion'] = measures_conversion
        conversion_summary['total_converted'] += measures_conversion.get('converted', 0)
        
        # Save comprehensive conversion log
        conversion_log = output_dir / "logs" / "all_outputs_conversion_summary.json"
        if conversion_log.parent.exists():
            with open(conversion_log, 'w') as f:
                json.dump(conversion_summary, f, indent=2)
        
        total_converted = conversion_summary['total_converted']
        self.logger.info(f"📊 All outputs conversion complete: {total_converted} files converted to CSV")
        
        return conversion_summary

    def convert_connectogram_files(self, output_dir: Path) -> Dict[str, Any]:
        """Convert .connectogram.txt files to CSV format.
        
        Parameters:
        -----------
        output_dir : Path
            Output directory containing .connectogram.txt files
            
        Returns:
        --------
        Dict[str, Any]
            Summary of connectogram conversion results
        """
        # Find all .connectogram.txt files
        connectogram_files = list(output_dir.rglob('*.connectogram.txt'))
        
        if not connectogram_files:
            self.logger.info("No .connectogram.txt files found for conversion")
            return {'success': True, 'converted': 0, 'files': []}
        
        self.logger.info(f"Converting {len(connectogram_files)} .connectogram.txt files...")
        
        conversion_results = []
        successful_conversions = 0
        
        for connectogram_file in connectogram_files:
            try:
                # Read DSI Studio connectogram format properly
                # Format: Line 1: streamline counts, Line 2: region names, Line 3+: connectivity matrix
                with open(connectogram_file, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) < 3:
                    raise ValueError("Invalid connectogram format: needs at least 3 lines")
                
                # Parse line 1: streamline counts (skip first two "data" entries)
                streamline_counts = lines[0].strip().split('\t')[2:]
                
                # Parse line 2: region names (skip first two "data" entries) 
                region_names = lines[1].strip().split('\t')[2:]
                
                # Ensure we have matching counts and names
                if len(streamline_counts) != len(region_names):
                    self.logger.warning(f"Mismatch in {connectogram_file.name}: {len(streamline_counts)} counts vs {len(region_names)} names")
                    # Use the shorter length to avoid index errors
                    min_length = min(len(streamline_counts), len(region_names))
                    streamline_counts = streamline_counts[:min_length]
                    region_names = region_names[:min_length]
                
                # Parse connectivity matrix (line 3 onwards)
                matrix_data = []
                row_info = []
                
                for i, line in enumerate(lines[2:]):
                    parts = line.strip().split('\t')
                    if len(parts) < 3:
                        continue
                    
                    # Extract row info: streamline_count, region_name, connectivity_values
                    row_streamlines = parts[0]
                    row_region = parts[1] 
                    connectivity_values = parts[2:]
                    
                    matrix_data.append(connectivity_values)
                    row_info.append({
                        'streamline_count': row_streamlines,
                        'region_name': row_region,
                        'row_index': i
                    })
                
                # Create enhanced connectivity matrix CSV with anatomical names
                if matrix_data and len(matrix_data[0]) == len(region_names):
                    # Create DataFrame with anatomical region names as headers and indices
                    connectivity_df = pd.DataFrame(
                        matrix_data,
                        index=[info['region_name'] for info in row_info],
                        columns=region_names,
                        dtype=float
                    )
                    
                    # Save enhanced connectivity matrix
                    csv_path = connectogram_file.with_suffix('.csv')
                    connectivity_df.to_csv(csv_path, index=True)
                    
                    # Create additional metadata file with streamline counts
                    metadata_path = connectogram_file.with_name(connectogram_file.stem + '.region_info.csv')
                    region_metadata = pd.DataFrame({
                        'region_name': region_names,
                        'streamline_count': streamline_counts,
                        'region_index': range(len(region_names))
                    })
                    region_metadata.to_csv(metadata_path, index=False)
                    
                    self.logger.info(f"✓ Enhanced conversion {connectogram_file.name} → CSV with anatomical names ({len(matrix_data)} x {len(region_names)} matrix)")
                    
                else:
                    # Fallback to original format if parsing fails
                    df = pd.read_csv(connectogram_file, sep='\t', header=None)
                    csv_path = connectogram_file.with_suffix('.csv')
                    df.to_csv(csv_path, index=False)
                    self.logger.warning(f"⚠️  Fallback conversion for {connectogram_file.name} (matrix dimension mismatch)")
                

                
                successful_conversions += 1
                
                conversion_results.append({
                    'success': True,
                    'original_file': str(connectogram_file),
                    'csv_path': str(csv_path),
                    'connections_count': len(matrix_data) if matrix_data else 0,
                    'matrix_size': f"{len(matrix_data)}x{len(region_names)}" if matrix_data else "unknown"
                })
                
            except Exception as e:
                self.logger.warning(f"✗ Failed to convert {connectogram_file.name}: {str(e)}")
                conversion_results.append({
                    'success': False,
                    'original_file': str(connectogram_file),
                    'error': str(e)
                })
        
        return {
            'success': True,
            'total_files': len(connectogram_files),
            'converted': successful_conversions,
            'failed': len(connectogram_files) - successful_conversions,
            'files': conversion_results
        }

    def convert_measures_files(self, output_dir: Path) -> Dict[str, Any]:
        """Convert .network_measures.txt files to CSV format.
        
        Parameters:
        -----------
        output_dir : Path
            Output directory containing .network_measures.txt files
            
        Returns:
        --------
        Dict[str, Any]
            Summary of network measures conversion results
        """
        # Find all .network_measures.txt files
        measures_files = list(output_dir.rglob('*.network_measures.txt'))
        
        if not measures_files:
            self.logger.info("No .network_measures.txt files found for conversion")
            return {'success': True, 'converted': 0, 'files': []}
        
        self.logger.info(f"Converting {len(measures_files)} .network_measures.txt files...")
        
        conversion_results = []
        successful_conversions = 0
        
        for measures_file in measures_files:
            try:
                # Read network measures file (typically has headers and is tab/space separated)
                # Try tab separation first
                try:
                    df = pd.read_csv(measures_file, sep='\t')
                except:
                    # If tab doesn't work, try space separation
                    df = pd.read_csv(measures_file, sep=' ')
                
                # Clean up column names (remove extra spaces)
                df.columns = df.columns.str.strip()
                
                # Save as CSV
                csv_path = measures_file.with_suffix('.csv')
                df.to_csv(csv_path, index=False)
                
                successful_conversions += 1
                self.logger.info(f"✓ Converted {measures_file.name} → CSV ({df.shape[0]} measures)")
                
                conversion_results.append({
                    'success': True,
                    'original_file': str(measures_file),
                    'csv_path': str(csv_path),
                    'measures_count': df.shape[0],
                    'metrics': list(df.columns) if df.shape[1] > 0 else []
                })
                
            except Exception as e:
                self.logger.warning(f"✗ Failed to convert {measures_file.name}: {str(e)}")
                conversion_results.append({
                    'success': False,
                    'original_file': str(measures_file),
                    'error': str(e)
                })
        
        return {
            'success': True,
            'total_files': len(measures_files),
            'converted': successful_conversions,
            'failed': len(measures_files) - successful_conversions,
            'files': conversion_results
        }
        """Convert all .mat files in output directory to CSV format.
        
        Parameters:
        -----------
        output_dir : Path
            Output directory containing .mat files
            
        Returns:
        --------
        Dict[str, Any]
            Summary of conversion results
        """
        if not MAT_SUPPORT:
            self.logger.warning("scipy not available - skipping .mat to CSV conversion")
            return {'success': False, 'error': 'scipy not available'}
        
        # Find all .mat files in the output directory
        mat_files = list(output_dir.rglob('*.mat'))
        
        if not mat_files:
            self.logger.info("No .mat files found for conversion")
            return {'success': True, 'converted': 0, 'files': []}
        
        self.logger.info(f"Converting {len(mat_files)} .mat files to CSV...")
        
        conversion_results = []
        successful_conversions = 0
        
        for mat_file in mat_files:
            # Extract atlas name from file path or filename
            atlas = "unknown"
            if "by_atlas" in mat_file.parts:
                atlas_idx = mat_file.parts.index("by_atlas")
                if atlas_idx + 1 < len(mat_file.parts):
                    atlas = mat_file.parts[atlas_idx + 1]
            
            result = self.convert_mat_to_csv(mat_file, atlas)
            result['mat_file'] = str(mat_file)
            result['atlas'] = atlas
            
            if result['success']:
                successful_conversions += 1
                self.logger.info(f"✓ Converted {mat_file.name} → CSV")
            else:
                self.logger.warning(f"✗ Failed to convert {mat_file.name}: {result.get('error', 'Unknown error')}")
            
            conversion_results.append(result)
        
        # Create conversion summary
        summary = {
            'success': True,
            'total_files': len(mat_files),
            'converted': successful_conversions,
            'failed': len(mat_files) - successful_conversions,
            'files': conversion_results
        }
        
        # Save conversion log
        conversion_log = output_dir / "logs" / "csv_conversion_summary.json"
        if conversion_log.parent.exists():
            with open(conversion_log, 'w') as f:
                json.dump(summary, f, indent=2)
        
        self.logger.info(f"CSV conversion complete: {successful_conversions}/{len(mat_files)} files converted")
        
        return summary
def create_batch_processor(input_dir: str, output_dir: str, pattern: str = "*.fib.gz") -> List[Dict]:
    """Process multiple fiber files in batch."""
    extractor = ConnectivityExtractor()
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Find all matching files
    fiber_files = list(input_path.glob(pattern))
    if not fiber_files:
        raise ValueError(f"No files found matching pattern: {pattern}")
    
    extractor.logger.info(f"Found {len(fiber_files)} files to process")
    
    batch_results = []
    for fiber_file in fiber_files:
        try:
            result = extractor.extract_all_matrices(str(fiber_file), output_dir)
            batch_results.append(result)
        except Exception as e:
            extractor.logger.error(f"Failed to process {fiber_file}: {e}")
            batch_results.append({
                'input_file': str(fiber_file),
                'error': str(e),
                'success': False
            })
    
    return batch_results

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="🧠 DSI Studio Connectivity Matrix Extraction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 QUICK START EXAMPLES:
  
  # 1. Validate setup first (recommended)
  python scripts/validate_setup.py --config my_config.json
  
  # 2. Single file processing
  python extract_connectivity_matrices.py --config my_config.json subject.fz output/
  
  # 3. Pilot test (test 1-2 files first)
  python extract_connectivity_matrices.py --config my_config.json --pilot --batch data_dir/ output/
  
  # 4. Full batch processing
  python extract_connectivity_matrices.py --config my_config.json --batch data_dir/ output/

📋 DETAILED EXAMPLES:

  # Basic single file with custom atlases
  python extract_connectivity_matrices.py --config my_config.json \\
      --atlases "AAL3,Brainnetome" subject.fz output/
  
  # Batch with specific settings
  python extract_connectivity_matrices.py --config my_config.json \\
      --batch --pattern "*.fz" --tracks 50000 --threads 16 data_dir/ output/
  
  # High-resolution tracking
  python extract_connectivity_matrices.py --config my_config.json \\
      --method 1 --fa_threshold 0.15 --turning_angle 35 subject.fz output/

📁 SUPPORTED FILE FORMATS: .fib.gz, .fz (auto-detected)

🔧 CONFIGURATION: Use --config to specify JSON configuration file
   (see example_config.json for template)

For more help: see README.md
        """)
    
    # Required arguments (made optional to show help when missing)
    # Positional aliases (kept for backward compatibility)
    parser.add_argument('input', nargs='?', 
                       help='📁 Input: .fib.gz/.fz file OR directory (for --batch mode)')
    parser.add_argument('output', nargs='?', 
                       help='📂 Output: Directory where results will be saved')
    # Optional aliases for consistency across tools
    parser.add_argument('-i', '--input', dest='input_opt', 
                       help='Alias for input (file or directory)')
    parser.add_argument('-o', '--output', dest='output_opt', 
                       help='Alias for output directory')
    
    # Configuration
    parser.add_argument('--config', type=str,
                       help='📄 JSON configuration file (recommended - see example_config.json)')
    
    # Processing mode
    parser.add_argument('--batch', action='store_true',
                       help='🔄 Batch mode: Process all files in input directory')
    
    parser.add_argument('--pilot', action='store_true',
                       help='🧪 Pilot mode: Test on subset of files first (use with --batch)')
    
    parser.add_argument('--pilot-count', type=int, default=1,
                       help='🔢 Number of files for pilot test (default: 1)')
    
    parser.add_argument('--pattern', default='*.fib.gz',
                       help='🔍 File pattern for batch mode (default: *.fib.gz, also searches .fz)')
    
    # Override configuration settings
    parser.add_argument('-a', '--atlases', 
                       help='🧠 Override config: Comma-separated atlases (e.g., "AAL3,Brainnetome")')
    
    parser.add_argument('-v', '--values',
                       help='📊 Override config: Comma-separated connectivity metrics')
    
    parser.add_argument('-t', '--tracks', type=int,
                       help='🛤️  Override config: Number of tracks to generate (e.g., 100000)')
    
    parser.add_argument('-j', '--threads', type=int,
                       help='⚡ Override config: Number of processing threads')
    
    # Advanced tracking parameters (override config)
    parser.add_argument('--method', type=int, choices=[0, 1, 2],
                       help='🎯 Tracking method: 0=Streamline(Euler), 1=RK4, 2=Voxel')
    
    parser.add_argument('--fa_threshold', type=float,
                       help='📉 FA threshold for termination (0=automatic, 0.1-0.3 typical)')
    
    parser.add_argument('--turning_angle', type=float,
                       help='🔄 Max turning angle in degrees (0=auto 15-90°, 35-60° typical)')
    
    parser.add_argument('--step_size', type=float,
                       help='📏 Step size in mm (0=auto 1-3 voxels)')
    
    parser.add_argument('--smoothing', type=float,
                       help='🌊 Smoothing fraction (0-1, higher=smoother tracks)')
    
    parser.add_argument('--track_voxel_ratio', type=float,
                       help='🎲 Seeds per voxel ratio (higher=more tracks per region)')
    
    parser.add_argument('--connectivity_type', choices=['pass', 'end'],
                       help='🔗 Connectivity type: pass=whole tract, end=endpoints only')
    
    parser.add_argument('--connectivity_threshold', type=float,
                       help='🎚️  Connectivity threshold for matrix filtering')
    
    parser.add_argument('--csv', action='store_true',
                       help='📊 Convert .mat files to CSV format (requires scipy)')
    
    parser.add_argument('--no-csv', action='store_true', 
                       help='🚫 Skip automatic .mat to CSV conversion')

    # Verbosity controls
    # Quiet by default; use --no-quiet to disable minimal console output
    parser.add_argument('--quiet', dest='quiet', action='store_true', default=True,
                       help='🔕 Minimal console output (default: ON)')
    parser.add_argument('--no-quiet', dest='quiet', action='store_false',
                       help='🔊 Full console output (show detailed steps)')
    parser.add_argument('--debug-dsi', action='store_true',
                       help='🐞 Print full DSI Studio command to console')
    
    args = parser.parse_args()

    # Reconcile optional -i/-o with positional args
    # If both provided and differ, raise a clear error
    if args.input_opt and args.input and args.input_opt != args.input:
        print("❌ Conflicting input provided via positional and -i/--input")
        sys.exit(2)
    if args.output_opt and args.output and args.output_opt != args.output:
        print("❌ Conflicting output provided via positional and -o/--output")
        sys.exit(2)
    # Prefer explicit optional flags when given
    if args.input_opt and not args.input:
        args.input = args.input_opt
    if args.output_opt and not args.output:
        args.output = args.output_opt
    
    # Show help if no arguments provided
    if len(sys.argv) == 1 or (not args.input and not args.output and not args.config):
        parser.print_help()
        print("\n💡 TIP: Start with validation:")
        print("   python scripts/validate_setup.py --config example_config.json")
        print("\n💡 Or see the README.md for detailed examples!")
        sys.exit(0)
    
    # Load configuration from file if provided
    config = DEFAULT_CONFIG.copy()
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config.update(json.load(f))
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {args.config}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in configuration file: {e}")
            sys.exit(1)
    
    # Override with command line arguments (only if provided)
    if args.atlases:
        config['atlases'] = args.atlases.split(',')
    if args.values:
        config['connectivity_values'] = args.values.split(',')
    if args.tracks:
        config['track_count'] = args.tracks
    if args.threads:
        config['thread_count'] = args.threads
    
    # Update tracking parameters if provided
    tracking_params = config.get('tracking_parameters', {})
    if args.method is not None:
        tracking_params['method'] = args.method
    if args.fa_threshold is not None:
        tracking_params['fa_threshold'] = args.fa_threshold
    if args.turning_angle is not None:
        tracking_params['turning_angle'] = args.turning_angle
    if args.step_size is not None:
        tracking_params['step_size'] = args.step_size
    if args.smoothing is not None:
        tracking_params['smoothing'] = args.smoothing
    if args.track_voxel_ratio is not None:
        tracking_params['track_voxel_ratio'] = args.track_voxel_ratio
    
    config['tracking_parameters'] = tracking_params
    
    # Update connectivity options if provided
    connectivity_options = config.get('connectivity_options', {})
    if args.connectivity_type is not None:
        connectivity_options['connectivity_type'] = args.connectivity_type
    if args.connectivity_threshold is not None:
        connectivity_options['connectivity_threshold'] = args.connectivity_threshold
    
    config['connectivity_options'] = connectivity_options
    
    # Handle CSV conversion options from command line
    if args.no_csv:
        connectivity_options['convert_to_csv'] = False
    elif args.csv:
        connectivity_options['convert_to_csv'] = True
    # If neither --csv nor --no-csv specified, use config file setting (or default to True)
    
    config['connectivity_options'] = connectivity_options
    
    # Apply verbosity flags into config so ConnectivityExtractor can read them
    config['quiet'] = bool(args.quiet)
    config['debug_dsi'] = bool(args.debug_dsi)

    # Check for required arguments
    if not args.input or not args.output:
        print("❌ Error: Both input and output arguments are required!\n")
        parser.print_help()
        print("\n💡 QUICK START:")
        print("   python scripts/validate_setup.py --config example_config.json")
        print("   python extract_connectivity_matrices.py --config example_config.json input.fz output/")
        sys.exit(1)
    
    try:
        extractor = ConnectivityExtractor(config)
        
        # Run validation first
        print("🔍 Validating configuration...")
        validation_result = extractor.validate_configuration()
        
        if not validation_result['valid']:
            print("❌ Configuration validation failed!")
            for error in validation_result['errors']:
                print(f"   ❌ {error}")
            sys.exit(1)
        
        if validation_result['warnings']:
            print(f"⚠️  {len(validation_result['warnings'])} warning(s):")
            for warning in validation_result['warnings']:
                print(f"   ⚠️  {warning}")
            print()
        
        if args.batch or os.path.isdir(args.input):
            # Batch processing mode
            print(f"🔍 Batch processing mode activated")
            print(f"📁 Input directory: {args.input}")
            print(f"🔍 File pattern: {args.pattern}")
            
            # Validate input path and find files
            input_validation = extractor.validate_input_path(args.input, args.pattern)
            if not input_validation['valid']:
                print("❌ Input validation failed!")
                for error in input_validation['errors']:
                    print(f"   ❌ {error}")
                sys.exit(1)
            
            fiber_files = input_validation['files_found']
            if not fiber_files:
                print("❌ No fiber files found!")
                print("💡 Supported formats: .fib.gz and .fz files")
                sys.exit(1)
            
            # Handle pilot mode
            if args.pilot:
                print(f"🧪 Pilot mode: selecting {args.pilot_count} random file(s)")
                fiber_files = extractor.select_pilot_files(fiber_files, args.pilot_count)
            
            # Process files
            print(f"📊 Processing {len(fiber_files)} file(s)...")
            batch_results = []
            
            for i, fiber_file in enumerate(fiber_files, 1):
                print(f"\n{'='*60}")
                print(f"Processing file {i}/{len(fiber_files)}: {os.path.basename(fiber_file)}")
                print(f"{'='*60}")
                
                try:
                    result = extractor.extract_all_matrices(str(fiber_file), args.output)
                    batch_results.append({
                        'file': fiber_file,
                        'success': True,
                        'output_dir': result.get('output_folder', 'unknown'),
                        'matrices_extracted': result.get('matrices_extracted', 0)
                    })
                    print(f"✅ Successfully processed {os.path.basename(fiber_file)}")
                    
                except Exception as e:
                    print(f"❌ Failed to process {os.path.basename(fiber_file)}: {e}")
                    batch_results.append({
                        'file': fiber_file,
                        'success': False,
                        'error': str(e)
                    })
                    continue
            
            # Summary
            successful = sum(1 for r in batch_results if r.get('success', False))
            failed = len(batch_results) - successful
            
            print(f"\n{'='*60}")
            print(f"BATCH PROCESSING SUMMARY")
            print(f"{'='*60}")
            print(f"📁 Total files processed: {len(batch_results)}")
            print(f"✅ Successful: {successful}")
            print(f"❌ Failed: {failed}")
            
            if args.pilot:
                print(f"🧪 Pilot mode: {args.pilot_count} file(s) tested")
                print(f"   Ready for full batch processing: {'YES' if successful > 0 else 'NO'}")
            
            # Save batch summary
            dsi_check = extractor.check_dsi_studio()
            summary_file = os.path.join(args.output, 'batch_processing_summary.json')
            with open(summary_file, 'w') as f:
                json.dump({
                    'processed_files': batch_results,
                    'dsi_studio': {
                        'path': dsi_check['path'],
                        'version': dsi_check.get('version', 'Unknown'),
                        'available': dsi_check['available']
                    },
                    'summary': {
                        'total': len(batch_results),
                        'successful': successful,
                        'failed': failed,
                        'pilot_mode': args.pilot,
                        'pilot_count': args.pilot_count if args.pilot else None
                    },
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            print(f"📄 Batch summary saved: {summary_file}")
            
        else:
            # Single file processing mode
            print(f"📊 Processing single file: {args.input}")
            
            # Log DSI Studio version at start of single file processing
            dsi_check = extractor.check_dsi_studio()
            logging.info(f"DSI Studio: {dsi_check['path']} - Version: {dsi_check.get('version', 'Unknown')}")
            
            # Validate single input file
            input_validation = extractor.validate_input_path(args.input)
            if not input_validation['valid']:
                print("❌ Input file validation failed!")
                for error in input_validation['errors']:
                    print(f"   ❌ {error}")
                sys.exit(1)

            result = extractor.extract_all_matrices(args.input, args.output)
            print("✅ Processing completed successfully!")
            
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
