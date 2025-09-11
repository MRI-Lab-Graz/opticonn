#!/usr/bin/env python3
"""
Test script for enhanced connectogram conversion
"""
import sys
import os
sys.path.append('/Volumes/Work/github/braingraph-pipeline/scripts')

from extract_connectivity_matrices import ConnectivityExtractor
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_enhanced_conversion():
    """Test the enhanced connectogram conversion on existing files"""
    
    # Path to existing connectogram
    test_file = Path("/Volumes/Work/github/braingraph-pipeline/studies/soccer/bootstrap_qa_wave_1/organized_matrices/sub-122BPAF171001.odf.qsdr_20250911_141656/tracks_100k_streamline_fa0.10/by_atlas/Cerebellum-SUIT/sub-122BPAF171001.odf.qsdr_Cerebellum-SUIT.tt.gz.Cerebellum-SUIT.count..pass.connectogram.txt")
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    print(f"🧪 Testing enhanced conversion on: {test_file.name}")
    
    # Initialize extractor
    extractor = ConnectivityExtractor()
    
    # Test the conversion
    result = extractor.convert_connectogram_files(test_file.parent)
    
    # Check results
    csv_file = test_file.with_suffix('.csv')
    region_info_file = test_file.with_name(test_file.stem + '.region_info.csv')
    
    print(f"\n📊 Conversion Results:")
    print(f"  Success: {result.get('success', False)}")
    print(f"  Converted files: {result.get('converted', 0)}")
    
    if csv_file.exists():
        print(f"\n✅ Enhanced CSV created: {csv_file.name}")
        # Load and show sample
        df = pd.read_csv(csv_file, index_col=0)
        print(f"  📏 Matrix dimensions: {df.shape}")
        print(f"  🏷️  Column names: {list(df.columns[:5])}...")
        print(f"  🏷️  Row names: {list(df.index[:5])}...")
    
    if region_info_file.exists():
        print(f"\n✅ Region info created: {region_info_file.name}")
        region_df = pd.read_csv(region_info_file)
        print(f"  📏 Regions: {len(region_df)}")
        print(f"  🏷️  Sample regions: {list(region_df['region_name'].head(3))}")
        print(f"  🔢 Sample streamline counts: {list(region_df['streamline_count'].head(3))}")

if __name__ == "__main__":
    test_enhanced_conversion()
