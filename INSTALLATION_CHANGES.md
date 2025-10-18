# Installation Changes Summary

## Overview

This document summarizes the changes made to implement mandatory `--dsi-path` command-line argument for the `install.sh` script, completing the DSI Studio path configuration system.

---

## Changes Made

### 1. **install.sh** - Added Command-Line Argument Parsing

#### What Changed
- Added argument parsing to handle `--dsi-path` as a mandatory command-line flag
- Added `--help` option to display usage information
- Removed interactive prompt for DSI Studio path (`read -p`)
- Now requires explicit path specification at installation time

#### Key Features
- **Mandatory `--dsi-path`** - Installation fails with helpful error if not provided
- **Usage help** - `bash install.sh --help` shows complete usage information
- **Error handling** - Invalid options are clearly reported with guidance
- **Examples** - Help text includes Linux and macOS examples

#### Code Changes
```bash
# Parse command-line arguments
DSI_STUDIO_PATH=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dsi-path)
            DSI_STUDIO_PATH="$2"
            shift 2
            ;;
        --help|-h)
            # Display help information and exit
            ;;
        *)
            # Error on unknown options
            ;;
    esac
done

# Validate that --dsi-path was provided
if [ -z "$DSI_STUDIO_PATH" ]; then
    echo "Error: --dsi-path is required"
    exit 1
fi
```

### 2. **README.md** - Updated Installation Instructions

#### What Changed
- Updated "Quick install (macOS & Linux)" section with `--dsi-path` flag examples
- Provided separate examples for Linux and macOS paths
- Added note about mandatory `--dsi-path` argument
- Updated "Verify the setup" section to explain DSI_STUDIO_PATH environment variable

#### New Examples
```bash
# Linux
bash install.sh --dsi-path /usr/local/bin/dsi_studio

# macOS
bash install.sh --dsi-path /Applications/dsi_studio.app/Contents/MacOS/dsi_studio
```

### 3. **INSTALLATION_GUIDE.md** - Comprehensive Installation Documentation

#### What Included
- Complete guide on finding DSI Studio executable for each OS
- Detailed installation examples for Linux, macOS, and Windows
- Command-line options reference
- Step-by-step installation process explanation
- Troubleshooting section with common errors and solutions
- Environment variable reference
- Re-installation instructions
- System administrator guidance

---

## Installation Examples

### Before (Interactive)
```bash
cd braingraph-pipeline
bash install.sh
# Script would prompt: "DSI Studio path: "
```

### After (Command-Line Argument)
```bash
cd braingraph-pipeline
bash install.sh --dsi-path /usr/local/bin/dsi_studio
source braingraph_pipeline/bin/activate
```

---

## Backwards Compatibility

**Breaking Change**: The installation script no longer accepts interactive input.

**Impact**: 
- Automated deployments (Docker, CI/CD, scripts) now have a proper way to specify the path
- Manual installations now require the `--dsi-path` flag
- Headless environments can use the flag without TTY requirements

---

## Help Information

Users can now access installation help:

```bash
bash install.sh --help
```

**Output:**
```
Usage: ./install.sh [OPTIONS]

OPTIONS:
  --dsi-path PATH      Path to DSI Studio executable (REQUIRED)
                       Example: /usr/local/bin/dsi_studio
                       Or: /Applications/dsi_studio.app/Contents/MacOS/dsi_studio
  --help               Show this help message

EXAMPLE:
  ./install.sh --dsi-path /usr/local/bin/dsi_studio
```

---

## Error Handling

### Missing DSI Studio Path
```bash
$ bash install.sh

❌ Error: --dsi-path is required

Usage: ./install.sh --dsi-path /path/to/dsi_studio

Examples:
  Linux:   ./install.sh --dsi-path /usr/local/bin/dsi_studio
  macOS:   ./install.sh --dsi-path /Applications/dsi_studio.app/Contents/MacOS/dsi_studio

Use --help for more information
```

### Invalid Options
```bash
$ bash install.sh --invalid-option

❌ Unknown option: --invalid-option
Use --help for usage information
```

### Path Not Found
```bash
$ bash install.sh --dsi-path /nonexistent/path

❌ DSI Studio executable not found at: /nonexistent/path
Installation canceled. Please verify the --dsi-path is correct.
```

---

## Testing Validation

All changes have been validated:

✅ **Syntax**: `bash -n install.sh` passes validation
✅ **Help flag**: `bash install.sh --help` displays correctly
✅ **Missing argument**: Script requires `--dsi-path` flag
✅ **Invalid options**: Unknown options are properly rejected
✅ **Error messages**: Clear, helpful error messaging with examples
✅ **Argument parsing**: Correctly extracts path value from `--dsi-path VALUE`

---

## Integration with Existing Systems

### Virtual Environment Activation
The DSI_STUDIO_PATH is automatically set when activating the environment:

```bash
source braingraph_pipeline/bin/activate
echo $DSI_STUDIO_PATH  # Shows the path you provided during installation
```

### Subprocess Propagation
All subprocess calls receive DSI_STUDIO_PATH via `scripts/utils/runtime.py`:

```python
def propagate_no_emoji():
    """Propagate environment variables to subprocess calls"""
    env = os.environ.copy()
    env['DSI_STUDIO_PATH'] = os.environ.get('DSI_STUDIO_PATH', '')
    env['QT_QPA_PLATFORM'] = 'offscreen'
    return env
```

### JSON Configuration Files
All configuration files use the generic `"dsi_studio"` command:

```json
{
  "dsi_studio_cmd": "dsi_studio"
}
```

The environment variable ensures the correct executable is found.

---

## Migration Guide for Existing Installations

If you have an existing installation:

```bash
# Remove old venv
rm -rf braingraph_pipeline

# Run new installation with --dsi-path flag
bash install.sh --dsi-path /path/to/dsi_studio

# Activate
source braingraph_pipeline/bin/activate
```

---

## Related Documents

- **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** - Complete user guide for installation
- **[README.md](README.md)** - Main pipeline documentation
- **[DSI_STUDIO_PATH_CONFIG.md](DSI_STUDIO_PATH_CONFIG.md)** - Technical path configuration details
- **[HEADLESS_SERVER_FIX.md](HEADLESS_SERVER_FIX.md)** - Headless/remote server setup

---

## Summary

The installation system is now:
- ✅ **Explicit**: Requires DSI Studio path as command-line argument
- ✅ **Non-interactive**: Works in automated environments (Docker, CI/CD, remote servers)
- ✅ **User-friendly**: Provides clear help and error messages
- ✅ **Documented**: Comprehensive guides for all user types
- ✅ **Portable**: Works across Linux, macOS, and Windows
- ✅ **Maintainable**: Single source of truth for DSI Studio path configuration
