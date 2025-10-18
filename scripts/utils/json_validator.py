#!/usr/bin/env python3
"""
JSON Configuration Validator
============================

Validates JSON configuration files against schemas to ensure required fields
are present and values are valid.

Author: Braingraph Pipeline Team
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    import jsonschema
except ImportError:
    jsonschema = None


class JSONValidator:
    """
    Comprehensive JSON validation class for DSI Studio configurations.
    """

    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize validator with optional schema path.

        Args:
            schema_path: Path to JSON schema file
        """
        self.schema_path = schema_path
        self.schema = None
        self.errors = []

        if schema_path and Path(schema_path).exists():
            self.load_schema(schema_path)

    def load_schema(self, schema_path: str) -> bool:
        """
        Load JSON schema from file.

        Args:
            schema_path: Path to schema file

        Returns:
            True if schema loaded successfully
        """
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
            self.schema_path = schema_path
            return True
        except Exception as e:
            self.errors.append(f"Failed to load schema from {schema_path}: {e}")
            return False

    @staticmethod
    def is_valid_json(filepath: str) -> bool:
        """
        Check if file contains valid JSON.

        Args:
            filepath: Path to JSON file

        Returns:
            True if file contains valid JSON
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                json.load(f)
            return True
        except (json.JSONDecodeError, FileNotFoundError, OSError):
            return False

    def validate_config(self, config_path: str) -> Tuple[bool, List[str]]:
        """
        Validate configuration file against schema.

        Args:
            config_path: Path to configuration file

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        self.errors = []

        # Check if config file exists
        if not Path(config_path).exists():
            self.errors.append(f"Configuration file not found: {config_path}")
            return False, self.errors

        # Check if config is valid JSON
        if not self.is_valid_json(config_path):
            self.errors.append(f"Invalid JSON in configuration file: {config_path}")
            return False, self.errors

        # Load configuration
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            self.errors.append(f"Failed to load configuration: {e}")
            return False, self.errors

        # Determine config type and validate accordingly
        if self._is_pipeline_test_config(config):
            validation_errors = self._validate_pipeline_test_config(config)
        else:
            # Traditional DSI Studio config validation
            validation_errors = self._validate_dsi_studio_config(config)

        if validation_errors:
            self.errors.extend(validation_errors)
            return False, self.errors

        return True, []

    def _is_pipeline_test_config(self, config: Dict[str, Any]) -> bool:
        """Check if this is a pipeline test configuration."""
        return "test_config" in config or "data_selection" in config or "pipeline_config" in config

    def _validate_pipeline_test_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate pipeline test configuration."""
        errors = []

        # Check test_config section
        if "test_config" in config:
            test_config = config["test_config"]
            if "name" not in test_config:
                errors.append("test_config.name is required")
            if "enabled" in test_config and not isinstance(test_config["enabled"], bool):
                errors.append("test_config.enabled must be boolean")

        # Check data_selection section
        if "data_selection" in config:
            data_sel = config["data_selection"]
            if "source_dir" not in data_sel:
                errors.append("data_selection.source_dir is required")
            elif not Path(data_sel["source_dir"]).exists():
                errors.append(f"data_selection.source_dir does not exist: {data_sel['source_dir']}")

            if "selection_method" in data_sel:
                valid_methods = ["random", "first", "specific"]
                if data_sel["selection_method"] not in valid_methods:
                    errors.append(f"data_selection.selection_method must be one of: {valid_methods}")

            if "n_subjects" in data_sel:
                n_subjects = data_sel["n_subjects"]
                if n_subjects != "all" and not isinstance(n_subjects, int):
                    errors.append("data_selection.n_subjects must be integer or 'all'")
                elif isinstance(n_subjects, int) and n_subjects <= 0:
                    errors.append("data_selection.n_subjects must be positive")

        # Check pipeline_config section
        if "pipeline_config" in config:
            pipeline_config = config["pipeline_config"]
            if "extraction_config" in pipeline_config:
                config_file = pipeline_config["extraction_config"]
                if not Path(config_file).exists():
                    errors.append(f"pipeline_config.extraction_config file not found: {config_file}")

            if "steps_to_run" in pipeline_config:
                valid_steps = ["01", "02", "03", "04"]
                for step in pipeline_config["steps_to_run"]:
                    if step not in valid_steps:
                        errors.append(f"Invalid step in pipeline_config.steps_to_run: {step}")

        return errors

    def _validate_dsi_studio_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Additional validation specific to DSI Studio configurations.

        Args:
            config: Configuration dictionary

        Returns:
            List of validation errors
        """
        errors = []

        # Check DSI Studio executable path
        if "dsi_studio_cmd" in config:
            dsi_path = config["dsi_studio_cmd"]
            
            # If dsi_studio_cmd is the generic "dsi_studio" command, try to resolve it using DSI_STUDIO_PATH
            if dsi_path == "dsi_studio" and "DSI_STUDIO_PATH" in os.environ:
                dsi_path = os.environ["DSI_STUDIO_PATH"]
            
            dsi_path_obj = Path(dsi_path)
            if not dsi_path_obj.exists():
                errors.append(f"DSI Studio executable not found: {dsi_path}")
            elif not dsi_path_obj.is_file():
                errors.append(f"DSI Studio path is not a file: {dsi_path}")

        # Validate atlas names
        valid_atlases = {
            "AAL3",
            "ATAG_basal_ganglia",
            "BrainSeg",
            "Brainnectome",
            "Brodmann",
            "Campbell",
            "Cerebellum-SUIT",
            "CerebrA",
            "FreeSurferDKT_Cortical",
            "FreeSurferDKT_Subcortical",
            "FreeSurferDKT_Tissue",
            "FreeSurferSeg",
            "HCP-MMP",
            "HCP842_tractography",
            "HCPex",
            "JulichBrain",
            "Kleist",
        }

        if "atlases" in config:
            for atlas in config["atlases"]:
                if atlas not in valid_atlases:
                    errors.append(f"Unknown atlas: {atlas}. Valid atlases: {sorted(valid_atlases)}")

        # Validate connectivity values
        valid_metrics = {"count", "fa", "qa", "ncount2", "md", "ad", "rd", "iso"}

        if "connectivity_values" in config:
            for metric in config["connectivity_values"]:
                if metric not in valid_metrics:
                    errors.append(f"Unknown connectivity metric: {metric}. Valid metrics: {sorted(valid_metrics)}")

        # Check parameter ranges
        if "tracking_parameters" in config:
            params = config["tracking_parameters"]

            # Check specific parameter constraints
            if "otsu_threshold" in params:
                if not 0.0 <= params["otsu_threshold"] <= 1.0:
                    errors.append("otsu_threshold must be between 0.0 and 1.0")

            if "fa_threshold" in params:
                if not 0.0 <= params["fa_threshold"] <= 1.0:
                    errors.append("fa_threshold must be between 0.0 and 1.0")

            if "turning_angle" in params:
                if not 0.0 <= params["turning_angle"] <= 90.0:
                    errors.append("turning_angle must be between 0.0 and 90.0 degrees")

            if "min_length" in params and "max_length" in params:
                if params["min_length"] >= params["max_length"]:
                    errors.append("min_length must be less than max_length")

        # Check thread count is reasonable
        if "thread_count" in config:
            if config["thread_count"] < 1 or config["thread_count"] > 32:
                errors.append("thread_count should be between 1 and 32")

        # Check tract count is reasonable
        if "tract_count" in config:
            if config["tract_count"] < 1000:
                errors.append("tract_count should be at least 1000 for meaningful results")
            elif config["tract_count"] > 10000000:
                errors.append("tract_count over 10 million may cause memory issues")

        return errors

    def get_missing_required_fields(self, config_path: str) -> List[str]:
        """
        Get list of missing required fields.

        Args:
            config_path: Path to configuration file

        Returns:
            List of missing required field names
        """
        if not self.schema or not Path(config_path).exists():
            return []

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            return []

        required_fields = self.schema.get("required", [])
        missing_fields = []

        for field in required_fields:
            if field not in config:
                missing_fields.append(field)

        return missing_fields

    def suggest_fixes(self, config_path: str) -> List[str]:
        """
        Suggest fixes for common configuration issues.

        Args:
            config_path: Path to configuration file

        Returns:
            List of suggested fixes
        """
        suggestions = []

        # Check if config exists
        if not Path(config_path).exists():
            suggestions.append(f"Create configuration file: {config_path}")
            return suggestions

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except json.JSONDecodeError:
            suggestions.append("Fix JSON syntax errors in configuration file")
            return suggestions
        except (FileNotFoundError, PermissionError, IOError):
            suggestions.append("Configuration file cannot be read")
            return suggestions

        # Check missing required fields
        missing_fields = self.get_missing_required_fields(config_path)
        if missing_fields:
            suggestions.append(f"Add missing required fields: {', '.join(missing_fields)}")

        # Check DSI Studio path
        if "dsi_studio_cmd" in config:
            dsi_path = Path(config["dsi_studio_cmd"])
            if not dsi_path.exists():
                suggestions.append("Update dsi_studio_cmd path to correct DSI Studio executable location")

        # Suggest common fixes
        if "track_count" in config and "tract_count" not in config:
            suggestions.append("Rename 'track_count' to 'tract_count' (DSI Studio expects 'tract_count')")

        if "atlases" not in config or not config.get("atlases"):
            suggestions.append("Add at least one atlas to the 'atlases' array")

        if "connectivity_values" not in config or not config.get("connectivity_values"):
            suggestions.append("Add at least one connectivity metric to 'connectivity_values' array")

        return suggestions


def validate_config_file(config_path: str, schema_path: Optional[str] = None) -> bool:
    """
    Standalone function to validate a configuration file.

    Args:
        config_path: Path to configuration file
        schema_path: Optional path to schema file

    Returns:
        True if valid, False otherwise
    """
    validator = JSONValidator(schema_path)
    is_valid, errors = validator.validate_config(config_path)

    if not is_valid:
        print(f"❌ Configuration validation failed for {config_path}:")
        for error in errors:
            print(f"   • {error}")

        # Print suggestions
        suggestions = validator.suggest_fixes(config_path)
        if suggestions:
            print("\n💡 Suggested fixes:")
            for suggestion in suggestions:
                print(f"   • {suggestion}")

        return False
    else:
        print(f"✅ Configuration {config_path} is valid!")
        return True


def main():
    """Command line interface for JSON validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate JSON configuration files")
    parser.add_argument("config_file", help="Path to configuration file")
    parser.add_argument("--schema", help="Path to JSON schema file")
    parser.add_argument(
        "--suggest-fixes",
        action="store_true",
        help="Show suggested fixes for validation errors",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform a dry-run: validate JSON syntax and show potential issues without exiting non-zero",
    )
    # Print help when called with no args
    import sys

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    # Set default schema if not provided
    if not args.schema:
        script_dir = Path(__file__).parent
        default_schema = script_dir / "dsi_studio_config_schema.json"
        if default_schema.exists():
            args.schema = str(default_schema)

    is_valid = validate_config_file(args.config_file, args.schema)

    if not is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
