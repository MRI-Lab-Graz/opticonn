#!/usr/bin/env python3
"""
Simple Parameter Uniqueness Checker
===================================

A lightweight check to ensure different parameter combinations produce 
different results by analyzing network measures files only.

Author: Braingraph Pipeline Team
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
import argparse
import logging
from collections import defaultdict

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def quick_uniqueness_check(matrices_dir):
    """Quick check using only network measures files."""
    logging.info("🔍 Quick parameter uniqueness check...")
    
    # Find aggregated network measures CSV
    agg_file = os.path.join(matrices_dir, 'aggregated_network_measures.csv')
    
    if not os.path.exists(agg_file):
        logging.error(f"❌ Aggregated file not found: {agg_file}")
        return False
    
    # Load aggregated data
    df = pd.read_csv(agg_file)
    logging.info(f"📊 Loaded {len(df)} network measures records")
    
    # Check parameter diversity
    subjects = df['subject_id'].nunique()
    atlases = df['atlas'].nunique()
    metrics = df['connectivity_metric'].nunique()
    
    logging.info(f"📈 Data diversity:")
    logging.info(f"  - Subjects: {subjects}")
    logging.info(f"  - Atlases: {atlases}")
    logging.info(f"  - Connectivity metrics: {metrics}")
    
    # Expected combinations per subject
    expected_combinations = atlases * metrics
    actual_combinations = df.groupby('subject_id').size()
    
    logging.info(f"\n🎯 Parameter combination check:")
    logging.info(f"  - Expected combinations per subject: {expected_combinations}")
    logging.info(f"  - Actual range: {actual_combinations.min()} - {actual_combinations.max()}")
    
    # Check for subjects with missing combinations
    incomplete_subjects = actual_combinations[actual_combinations < expected_combinations]
    if len(incomplete_subjects) > 0:
        logging.warning(f"⚠️  {len(incomplete_subjects)} subjects have incomplete parameter combinations")
        for subject, count in incomplete_subjects.head().items():
            logging.warning(f"    {subject}: {count}/{expected_combinations} combinations")
    else:
        logging.info("✅ All subjects have complete parameter combinations")
    
    # Check value diversity within subjects
    diversity_results = []
    key_measures = ['density', 'global_efficiency(binary)', 'clustering_coeff_average(binary)']
    
    for subject in df['subject_id'].unique()[:5]:  # Check first 5 subjects
        subject_data = df[df['subject_id'] == subject]
        
        for measure in key_measures:
            if measure in df.columns:
                measure_values = subject_data[measure]
                if len(measure_values) > 1:
                    values = measure_values.values
                    diversity_score = np.std(values) / (np.mean(values) + 1e-10)  # Coefficient of variation
                    diversity_results.append({
                        'subject': subject,
                        'measure': measure,
                        'n_values': len(values),
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'diversity_score': diversity_score
                    })
    
    if diversity_results:
        diversity_df = pd.DataFrame(diversity_results)
        logging.info(f"\n📊 Parameter diversity analysis (sample of {len(diversity_df)} measure/subject combinations):")
        
        avg_diversity = diversity_df.groupby('measure')['diversity_score'].mean()
        for measure, score in avg_diversity.items():
            logging.info(f"  - {measure}: diversity = {score:.4f}")
            
        # Flag low diversity
        low_diversity = diversity_df[diversity_df['diversity_score'] < 0.01]
        if len(low_diversity) > 0:
            logging.warning(f"⚠️  {len(low_diversity)} measure/subject combinations show low parameter diversity (<0.01)")
        else:
            logging.info("✅ Good parameter diversity detected across all measures")
    
    return True

def quality_outlier_analysis(matrices_dir):
    """Analyze quality metrics for outlier detection."""
    logging.info("\n🔍 Quality outlier analysis...")
    
    agg_file = os.path.join(matrices_dir, 'aggregated_network_measures.csv')
    if not os.path.exists(agg_file):
        logging.error(f"❌ Aggregated file not found: {agg_file}")
        return False
    
    df = pd.read_csv(agg_file)
    
    # Focus on key quality measures that exist as columns
    quality_measures = ['density', 'global_efficiency(binary)', 'clustering_coeff_average(binary)', 'small-worldness(binary)']
    
    outlier_summary = []
    
    for measure in quality_measures:
        if measure not in df.columns:
            continue
            
        values = df[measure].dropna().values
        if len(values) == 0:
            continue
        
        # Calculate outlier thresholds (using IQR method)
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find outliers based on values
        outlier_mask = (values < lower_bound) | (values > upper_bound)
        outlier_indices = df[df[measure].notna()].index[outlier_mask]
        outliers = df.loc[outlier_indices]
        
        outlier_subjects = outliers['subject_id'].unique()
        
        outlier_summary.append({
            'measure': measure,
            'n_total': len(values),
            'n_outliers': len(outliers),
            'outlier_rate': len(outliers) / len(values) if len(values) > 0 else 0,
            'outlier_subjects': len(outlier_subjects),
            'mean': np.mean(values),
            'std': np.std(values),
            'Q1': Q1,
            'Q3': Q3
        })
        
        if len(outliers) > 0:
            logging.warning(f"⚠️  {measure}: {len(outliers)} outliers ({len(outlier_subjects)} subjects)")
            worst_outliers = outliers.nlargest(3, measure)['subject_id'].tolist()
            logging.warning(f"    Subjects with extreme values: {worst_outliers[:3]}")
        else:
            logging.info(f"✅ {measure}: No outliers detected")
    
    # Overall quality assessment
    outlier_df = pd.DataFrame(outlier_summary)
    if not outlier_df.empty:
        avg_outlier_rate = outlier_df['outlier_rate'].mean()
        total_outlier_subjects = set()
        
        for measure in quality_measures:
            if measure not in df.columns:
                continue
                
            values = df[measure].dropna().values
            if len(values) == 0:
                continue
                
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[measure] < lower_bound) | (df[measure] > upper_bound)
            outlier_subjects = df[outlier_mask]['subject_id'].unique()
            
            total_outlier_subjects.update(outlier_subjects)
        
        logging.info(f"\n📊 Quality outlier summary:")
        logging.info(f"  - Average outlier rate: {avg_outlier_rate:.1%}")
        logging.info(f"  - Subjects with quality issues: {len(total_outlier_subjects)}")
        
        if len(total_outlier_subjects) > 0:
            logging.info(f"  - Potentially problematic subjects: {list(total_outlier_subjects)[:5]}...")
            
            if avg_outlier_rate > 0.10:  # More than 10% outliers
                logging.warning("⚠️  High outlier rate detected - consider reviewing data quality!")
            else:
                logging.info("✅ Outlier rate within acceptable range")
        else:
            logging.info("✅ No quality outliers detected across all measures")
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Quick parameter uniqueness and quality check')
    parser.add_argument('matrices_dir', help='Directory containing organized matrices')
    parser.add_argument('--quality-only', action='store_true', help='Run only quality analysis')
    
    args = parser.parse_args()
    setup_logging()
    
    logging.info("🚀 Starting parameter uniqueness and quality analysis...")
    
    success = True
    
    if not args.quality_only:
        success &= quick_uniqueness_check(args.matrices_dir)
    
    success &= quality_outlier_analysis(args.matrices_dir)
    
    if success:
        logging.info("\n✅ Analysis completed successfully!")
        return 0
    else:
        logging.error("\n❌ Analysis completed with issues!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
