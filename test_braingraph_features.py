#!/usr/bin/env python3
"""
Test die neuen Braingraph-Features auf bestehenden Daten
"""
import sys
import os
sys.path.append('/Volumes/Work/github/braingraph-pipeline/scripts')

from extract_connectivity_matrices import ConnectivityExtractor
import pandas as pd
from pathlib import Path
import logging

def test_new_features():
    """Test all three improvements on existing data"""
    
    print("🧪 TESTING BRAINGRAPH VERBESSERUNGEN")
    print("=" * 50)
    
    # Test directory with existing connectogram files  
    test_dir = Path("/Volumes/Work/github/braingraph-pipeline/studies/soccer_122_final/results/01_connectivity/organized_matrices/sub-122BPAF171001.odf.qsdr_20250910_083444/tracks_100k_streamline_fa0.10/by_atlas/FreeSurferDKT_Cortical")
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    print(f"📁 Test directory: {test_dir}")
    
    # Initialize extractor
    extractor = ConnectivityExtractor()
    
    # 1. TEST ENHANCED CONNECTOGRAM CONVERSION
    print("\n🔧 TEST 1: Enhanced Connectogram Conversion")
    print("-" * 45)
    
    # Find a connectogram file
    connectogram_files = list(test_dir.glob("*.connectogram.txt"))
    if connectogram_files:
        test_file = connectogram_files[0]
        print(f"📄 Testing file: {test_file.name}")
        
        # Test conversion
        result = extractor.convert_connectogram_files(test_dir)
        
        # Check enhanced CSV
        enhanced_csv = test_file.with_suffix('.csv')
        region_info_csv = test_file.with_name(test_file.stem + '.region_info.csv')
        
        if enhanced_csv.exists():
            df = pd.read_csv(enhanced_csv, index_col=0)
            print(f"  ✅ Enhanced CSV: {df.shape} matrix")
            print(f"  🏷️  Sample regions: {list(df.columns[:3])}")
            
        if region_info_csv.exists():
            region_df = pd.read_csv(region_info_csv)
            print(f"  ✅ Region info: {len(region_df)} regions")
            print(f"  🔢 Sample streamlines: {list(region_df['streamline_count'].head(3))}")
    else:
        print("❌ No connectogram files found")
    
    # 2. TEST SIMPLIFIED ORGANIZATION 
    print(f"\n🔧 TEST 2: Directory Organization")
    print("-" * 45)
    
    # Check current structure
    parent_dir = test_dir.parents[1]  # tracks_100k_streamline_fa0.10 level
    
    print(f"📁 Current structure in: {parent_dir.name}")
    subdirs = [d.name for d in parent_dir.iterdir() if d.is_dir()]
    print(f"  📂 Directories: {subdirs}")
    
    if 'by_atlas' in subdirs and 'by_metric' in subdirs and 'combined' in subdirs:
        print("  ⚠️  Old structure (3x duplication detected)")
        
        # Count files in each
        by_atlas_files = len(list((parent_dir / "by_atlas").rglob("*"))) if (parent_dir / "by_atlas").exists() else 0
        by_metric_files = len(list((parent_dir / "by_metric").rglob("*"))) if (parent_dir / "by_metric").exists() else 0
        combined_files = len(list((parent_dir / "combined").rglob("*"))) if (parent_dir / "combined").exists() else 0
        
        print(f"  📊 by_atlas: {by_atlas_files} files")
        print(f"  📊 by_metric: {by_metric_files} files")
        print(f"  📊 combined: {combined_files} files")
        print(f"  💾 Total: {by_atlas_files + by_metric_files + combined_files} files")
        print("  💡 New structure would use only 'results/' → ~67% reduction")
        
    elif 'results' in subdirs:
        print("  ✅ New simplified structure detected!")
        results_files = len(list((parent_dir / "results").rglob("*")))
        print(f"  📊 results: {results_files} files")
    
    # 3. TEST INFORMATION PRESERVATION
    print(f"\n🔧 TEST 3: Information Preservation")
    print("-" * 45)
    
    if connectogram_files:
        test_file = connectogram_files[0]
        
        # Compare TXT vs old CSV vs new CSV
        print(f"📄 Comparing formats for: {test_file.name}")
        
        # Original TXT format
        with open(test_file, 'r') as f:
            txt_lines = f.readlines()
        
        print(f"  📊 Original TXT:")
        if len(txt_lines) >= 2:
            streamlines = txt_lines[0].strip().split('\t')[2:5]  # First few streamline counts
            regions = txt_lines[1].strip().split('\t')[2:5]      # First few region names
            print(f"    🔢 Streamlines: {streamlines}")
            print(f"    🏷️  Regions: {regions}")
        
        # Old CSV format (if exists)
        old_csv = test_dir / test_file.name.replace('.connectogram.txt', '.connectivity.csv')
        if old_csv.exists():
            old_df = pd.read_csv(old_csv, index_col=0)
            print(f"  📊 Old CSV: {old_df.shape}")
            print(f"    🏷️  Headers: {list(old_df.columns[:3])}")
        
        # New enhanced CSV
        new_csv = test_file.with_suffix('.csv')
        if new_csv.exists():
            new_df = pd.read_csv(new_csv, index_col=0)
            print(f"  📊 Enhanced CSV: {new_df.shape}")
            print(f"    🏷️  Headers: {list(new_df.columns[:3])}")
            
            # Check if anatomical names preserved
            has_anatomical_names = any('Left_' in col or 'Right_' in col or 'Vermis_' in col for col in new_df.columns[:5])
            print(f"    ✅ Anatomical names: {'Preserved' if has_anatomical_names else 'Lost'}")
    
    print(f"\n🎉 TESTING COMPLETE")
    print("=" * 50)
    return True

if __name__ == "__main__":
    test_new_features()
