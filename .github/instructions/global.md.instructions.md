# Global Development Instructions

All scrpt should have this in the header
MRI - Lab Graz
Karl Koschutnig
karl.koschutnig@uni-graz.at

## Package Management
- **All packages MUST be installed via `install.sh`** - never install packages manually
- **All installations MUST occur within the virtual environment** - no global installations
- The virtual environment is located at `braingraph_pipeline/` and should be activated before any operations

## Script Standards
- **Every script MUST include a README** with:
  - Purpose and functionality description
  - Usage examples
  - Parameter explanations
- **Every script MUST support `--dry-run` flag** for safe testing
- **Scripts executed without flags MUST display help/usage information**

## Virtual Environment Usage
```bash
# Always activate the virtual environment first
source braingraph_pipeline/bin/activate

# Install new packages via install.sh
./install.sh

# Run scripts with dry-run for testing
python scripts/your_script.py --dry-run
```

## Development Workflow
1. Activate virtual environment
2. Install/update dependencies via `install.sh`
3. Test scripts with `--dry-run` flag
4. Read script READMEs for proper usage
5. Execute with appropriate parameters

## Compliance Requirements
- No direct pip/conda installations outside of `install.sh`
- No script execution without proper documentation
- All scripts must be testable via `--dry-run`

## scripts
- check for orphan functions in scripts and remove them
- check for functions that may do the same and merge them
- try to reduce friction between input and output of scripts