import itertools
import json
import os
import random
import subprocess
import numpy as np
def get_subjects(data_dir):
    return [f for f in os.listdir(data_dir) if f.startswith('sub-')]
def count_zeros_in_matrix(matrix_path):
    # DSI Studio matrices haben 2 Header-Zeilen
    try:
        mat = np.loadtxt(matrix_path, skiprows=2)
        # Nur die numerischen Spalten verwenden (ab Spalte 2)
        if mat.ndim == 2 and mat.shape[1] > 2:
            numeric_data = mat[:, 2:]  # Erste 2 Spalten sind Labels
        else:
            numeric_data = mat
        return int((numeric_data == 0).sum()), int(numeric_data.size)
    except Exception as e:
        # Fallback: Versuche anders zu parsen
        with open(matrix_path, 'r') as f:
            lines = f.readlines()
        # Finde die erste numerische Zeile
        numeric_lines = []
        for i, line in enumerate(lines):
            if i < 2:  # Überspringe Header
                continue
            parts = line.strip().split()
            if len(parts) > 2:
                try:
                    # Versuche die numerischen Teile zu konvertieren (ab Index 2)
                    numeric_parts = [float(x) for x in parts[2:]]
                    numeric_lines.append(numeric_parts)
                except ValueError:
                    continue
        if numeric_lines:
            mat = np.array(numeric_lines)
            return int((mat == 0).sum()), int(mat.size)
        else:
            raise Exception(f"Could not parse matrix: {e}")
def run_sweep(config_path, data_dir, results_dir, sweep_log='sweep_results.csv', quick=False):
    # Lade Sweep-Parameter aus der Konfigurationsdatei
    with open(config_path) as f:
        base_config = json.load(f)
    
    # Hole Sweep-Parameter aus der Config oder verwende Defaults
    sweep_config = base_config.get('sweep_parameters', {})
    
    if quick and 'quick_sweep' in sweep_config:
        # Verwende quick_sweep Parameter
        sweep_params = sweep_config['quick_sweep']
        print(f"[SWEEP] Using QUICK sweep parameters")
    else:
        # Verwende normale Sweep Parameter
        sweep_params = sweep_config
        print(f"[SWEEP] Using FULL sweep parameters")
    
    # Parameter-Ranges aus Config laden
    otsu_range = sweep_params.get('otsu_range', [0.3, 0.4])
    min_length_range = sweep_params.get('min_length_range', [10, 20])
    track_voxel_ratio_range = sweep_params.get('track_voxel_ratio_range', [2.0])
    fa_threshold_range = sweep_params.get('fa_threshold_range', [0.1])
    sweep_tract_count = sweep_params.get('sweep_tract_count', 50000)
    
    print(f"[SWEEP] Parameter ranges from config:")
    print(f"  otsu_range: {otsu_range}")
    print(f"  min_length_range: {min_length_range}")
    print(f"  track_voxel_ratio_range: {track_voxel_ratio_range}")
    print(f"  fa_threshold_range: {fa_threshold_range}")
    print(f"  sweep_tract_count: {sweep_tract_count}")
    
    subjects = get_subjects(data_dir)
    pilot = random.choice(subjects)
    pilot_file = os.path.join(data_dir, pilot)
    print(f"Pilot subject: {pilot}")
    print(f"Pilot file: {pilot_file}")
    
    with open(config_path) as f:
        base_config = json.load(f)
    
    with open(sweep_log, 'w') as logf:
        logf.write('otsu,min_length,track_voxel_ratio,fa_threshold,zeros,total,percent_zeros,combination,status\n')
    
    combination_num = 0
    total_combinations = len(list(itertools.product(otsu_range, min_length_range, track_voxel_ratio_range, fa_threshold_range)))
    
    for otsu, min_len, tvr, fa in itertools.product(otsu_range, min_length_range, track_voxel_ratio_range, fa_threshold_range):
        combination_num += 1
        print(f"\\n[SWEEP] Testing combination {combination_num}/{total_combinations}: otsu={otsu}, min_len={min_len}, tvr={tvr}, fa={fa}")
        
        # Cleanup previous temp files
        for tmp_file in ['tmp_sweep_config*.json']:
            import glob
            for f in glob.glob(tmp_file):
                try:
                    os.remove(f)
                except:
                    pass
        
        config = json.loads(json.dumps(base_config))
        config['tracking_parameters']['otsu_threshold'] = otsu
        config['tracking_parameters']['min_length'] = min_len
        config['tracking_parameters']['track_voxel_ratio'] = tvr
        config['tracking_parameters']['fa_threshold'] = fa
        
        # Reduzierte Anzahl Trakte um Crashes zu vermeiden
        config['tract_count'] = sweep_tract_count  # Verwende die einstellbare Variable
        
        tmp_config = f'tmp_sweep_config_{combination_num}.json'
        with open(tmp_config, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Führe direkt die Python-Extraktion aus mit Error Handling
        sweep_output_dir = os.path.join(results_dir, f'sweep_combo_{combination_num}')
        os.makedirs(sweep_output_dir, exist_ok=True)
        
        cmd = [
            'python3', '/Volumes/Work/github/braingraph-pipeline/extract_connectivity_matrices.py',
            '--config', tmp_config,
            pilot_file,
            os.path.join(sweep_output_dir, 'connectivity_matrices')
        ]
        
        print(f"[SWEEP] Running: {' '.join(cmd)}")
        status = "SUCCESS"
        
        try:
            # Längeres Timeout und besseres Error Handling
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"[SWEEP] Error in extraction (return code {result.returncode}): {result.stderr}")
                status = "EXTRACTION_ERROR"
                zeros, total, percent_zeros = 'EXTRACTION_ERROR', 'EXTRACTION_ERROR', 'EXTRACTION_ERROR'
            else:
                print(f"[SWEEP] Extraction successful")
                
                # Suche nach Matrizen in der rohen Output-Struktur (ohne Organisation)
                raw_dir = os.path.join(sweep_output_dir, 'connectivity_matrices')
                
                # Suche nach connectogram-Dateien
                import glob
                pattern = os.path.join(raw_dir, '**', '*count.pass.connectogram.txt')
                files = glob.glob(pattern, recursive=True)
                
                if files:
                    matrix_path = files[0]
                    try:
                        zeros, total = count_zeros_in_matrix(matrix_path)
                        percent_zeros = zeros / total * 100
                        print(f"[SWEEP] Matrix analysis: {zeros}/{total} zeros ({percent_zeros:.1f}%)")
                        status = "SUCCESS"
                    except Exception as e:
                        print(f"[SWEEP] Error analyzing matrix: {e}")
                        zeros, total, percent_zeros = 'ANALYSIS_ERROR', 'ANALYSIS_ERROR', 'ANALYSIS_ERROR'
                        status = "ANALYSIS_ERROR"
                else:
                    print(f"[SWEEP] No matrix found at {pattern}")
                    # Versuche andere Dateitypen zu finden
                    all_files = glob.glob(os.path.join(raw_dir, '**', '*.txt'), recursive=True)
                    print(f"[SWEEP] Found {len(all_files)} files in output directory")
                    if all_files:
                        print(f"[SWEEP] Example files: {all_files[:3]}")
                    zeros, total, percent_zeros = 'NO_MATRIX', 'NO_MATRIX', 'NO_MATRIX'
                    status = "NO_MATRIX"
                    
        except subprocess.TimeoutExpired:
            print(f"[SWEEP] Timeout for combination {combination_num}")
            zeros, total, percent_zeros = 'TIMEOUT', 'TIMEOUT', 'TIMEOUT'
            status = "TIMEOUT"
        except Exception as e:
            print(f"[SWEEP] Exception: {e}")
            zeros, total, percent_zeros = 'EXCEPTION', 'EXCEPTION', 'EXCEPTION'
            status = "EXCEPTION"
        
        # Log das Ergebnis
        with open(sweep_log, 'a') as logf:
            logf.write(f"{otsu},{min_len},{tvr},{fa},{zeros},{total},{percent_zeros},{combination_num},{status}\\n")
        
        # Cleanup
        if os.path.exists(tmp_config):
            os.remove(tmp_config)
            
        # Kurze Pause zwischen den Kombinationen um DSI Studio zu entlasten
        import time
        time.sleep(2)
        
    print(f"\\n[SWEEP] Completed {combination_num} combinations. Results in {sweep_log}")
if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='01_working_config.json')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--quick', action='store_true', help='Use quick sweep parameters')
    args = parser.parse_args()
    run_sweep(args.config, args.data_dir, args.results_dir, quick=args.quick)
