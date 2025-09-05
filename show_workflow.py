#!/usr/bin/env python3
"""
Test the simplified JSON config workflow
"""

import json
from pathlib import Path

def show_workflow_examples():
    """Show recommended workflow examples."""
    
    print("🧠 BRAINGRAPH PIPELINE - SIMPLIFIED JSON CONFIG WORKFLOW")
    print("=" * 60)
    
    print("\n📋 Available Configuration Files:")
    
    # Check available configs
    configs = {
        "01_working_config.json": "Routine connectivity extraction",
        "sweep_config.json": "Parameter optimization/research", 
        "production_config.json": "Analysis pipeline steps 02-04"
    }
    
    for config_file, description in configs.items():
        if Path(config_file).exists():
            print(f"  ✅ {config_file:<25} - {description}")
        else:
            print(f"  ❌ {config_file:<25} - {description} (missing)")
    
    print("\n🎯 RECOMMENDED WORKFLOW:")
    print("\n1️⃣  Connectivity Extraction (Step 01):")
    print("   python extract_connectivity_matrices.py --config 01_working_config.json --batch raw_data/ output/")
    
    print("\n2️⃣  Analysis Pipeline (Steps 02-04):")
    print("   python run_pipeline.py --step analysis --input output/organized_matrices/ --config production_config.json")
    
    print("\n🧪 PILOT TESTING:")
    print("   python extract_connectivity_matrices.py --config 01_working_config.json --batch --pilot raw_data/ output/")
    
    print("\n🔄 SINGLE COMMAND (ALL STEPS):")
    print("   python run_pipeline.py --step all --data-dir raw_data/ \\")
    print("                          --extraction-config 01_working_config.json \\")
    print("                          --config production_config.json")
    
    print("\n📊 EXAMPLE CONFIGS:")
    
    # Show config structure examples
    if Path("01_working_config.json").exists():
        with open("01_working_config.json") as f:
            config = json.load(f)
        
        print(f"\n📄 01_working_config.json (Extraction):")
        print(f"   - Atlases: {', '.join(config.get('atlases', [])[:3])}...")
        print(f"   - Metrics: {', '.join(config.get('connectivity_values', []))}")
        print(f"   - Tracks: {config.get('tract_count', 'N/A'):,}")
        print(f"   - Method: {config.get('tracking_parameters', {}).get('method', 'N/A')}")
    
    print("\n✨ BENEFITS:")
    print("   ✅ Reproducible: Exact parameters saved")
    print("   ✅ Shareable: Easy team collaboration")
    print("   ✅ Versioned: Track changes in git")
    print("   ✅ Clean: No complex command-line parsing")
    print("   ✅ Flexible: Easy study-specific configs")
    
    print("\n📖 For detailed documentation, see:")
    print("   - SIMPLIFIED_WORKFLOW.md")
    print("   - CONFIG_GUIDE.md")
    print("   - README.md")

if __name__ == "__main__":
    show_workflow_examples()
