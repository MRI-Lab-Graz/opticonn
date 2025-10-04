Run Pipeline (scripts/run_pipeline.py)
=====================================

This is a lightweight orchestrator for the pipeline steps. Usage examples:

- python scripts/run_pipeline.py --step all --data-dir /path/to/data --output results/
- python scripts/run_pipeline.py --step analysis --input organized_matrices/ --output results/

The script supports a --dry-run flag to preview actions without executing external commands.

If run without arguments the script prints usage/help.
