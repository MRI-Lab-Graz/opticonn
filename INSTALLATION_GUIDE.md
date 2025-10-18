# Installation Guide: DSI Studio Path Configuration

## Overview

The braingraph pipeline requires [DSI Studio](https://dsi-studio.labsolver.org/) for tractography processing. Starting with this version, the DSI Studio path is configured at **installation time** via a mandatory command-line flag, ensuring proper environment setup from the start.

---

## Installation Instructions

### Quick Start

```bash
# Navigate to the pipeline directory
cd braingraph-pipeline

# Run installation with DSI Studio path
bash install.sh --dsi-path /path/to/dsi_studio

# Activate the virtual environment
source braingraph_pipeline/bin/activate
```

### Finding Your DSI Studio Executable

#### Linux
- **System-wide installation**: `/usr/local/bin/dsi_studio` or `/usr/bin/dsi_studio`
- **User installation**: `~/Applications/dsi_studio` or `~/.local/bin/dsi_studio`
- **Custom installation**: Check the installation directory you specified

To find it:
```bash
which dsi_studio
# or
find /usr -name dsi_studio -type f 2>/dev/null
```

#### macOS
- **Standard installation**: `/Applications/dsi_studio.app/Contents/MacOS/dsi_studio`
- **Alternative**: Check the Applications folder for the DSI Studio app bundle

To find it:
```bash
ls -la /Applications/dsi_studio.app/Contents/MacOS/dsi_studio
```

#### Windows
- **Standard installation**: `C:\Program Files\dsi_studio\dsi_studio.exe` or `C:\Program Files (x86)\dsi_studio\dsi_studio.exe`
- Use `install_windows.bat` instead of the bash script

---

## Installation Examples

### Linux System-Wide Installation
```bash
bash install.sh --dsi-path /usr/local/bin/dsi_studio
```

### macOS Default Installation
```bash
bash install.sh --dsi-path /Applications/dsi_studio.app/Contents/MacOS/dsi_studio
```

### Linux with Custom Installation Path
```bash
bash install.sh --dsi-path /home/username/software/dsi_studio/dsi_studio
```

### Testing the Installation
```bash
# Check if DSI Studio works with the configured path
bash install.sh --dsi-path /path/to/dsi_studio
source braingraph_pipeline/bin/activate

# Verify DSI Studio is accessible
dsi_studio --version

# Run the setup validator
python scripts/validate_setup.py --config configs/braingraph_default_config.json
```

---

## Command-Line Options

### `--dsi-path PATH` (Required)
Specifies the full path to the DSI Studio executable.

**Example:**
```bash
bash install.sh --dsi-path /usr/local/bin/dsi_studio
```

### `--help` or `-h`
Displays usage information and examples.

**Example:**
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

## What Happens During Installation

When you run `bash install.sh --dsi-path /path/to/dsi_studio`, the script:

1. **Validates the DSI Studio path** - Confirms the executable exists and is accessible
2. **Tests DSI Studio functionality** - Runs `dsi_studio --version` to ensure it works properly
3. **Creates the virtual environment** - Sets up Python 3.10 venv with all required packages
4. **Configures environment variables** - Sets:
   - `PYTHONPATH` - Points to the pipeline code
   - `TMPDIR`, `TEMP`, `TMP` - Points to `/data/local/tmp_big` for large temporary files
   - **`DSI_STUDIO_PATH`** - Sets to your provided DSI Studio executable path

5. **Stores configuration persistently** - Adds the DSI Studio path to the virtual environment activation script (`braingraph_pipeline/bin/activate`)

---

## Environment Variable: DSI_STUDIO_PATH

### How It Works

After installation, whenever you activate the virtual environment:
```bash
source braingraph_pipeline/bin/activate
```

The following is automatically set:
```bash
export DSI_STUDIO_PATH=/path/to/dsi_studio  # Your provided path
```

This environment variable is used by:
- `scripts/utils/runtime.py` - Propagates to all subprocess calls
- `scripts/bayesian_optimizer.py` - Uses for parallel optimization workers
- `scripts/extract_connectivity_matrices.py` - Uses for tractography extraction
- All pipeline scripts that need to call DSI Studio

### Verification

Check that the environment variable is set correctly:
```bash
source braingraph_pipeline/bin/activate
echo $DSI_STUDIO_PATH
# Should output: /path/to/dsi_studio

# Verify it works
$DSI_STUDIO_PATH --version
```

---

## Troubleshooting

### Error: "DSI Studio executable not found"

**Cause**: The provided path doesn't exist or isn't an executable file.

**Solutions**:
1. Verify the path is correct:
   ```bash
   ls -la /path/to/dsi_studio
   ```

2. Locate the correct DSI Studio executable:
   ```bash
   # Linux
   which dsi_studio
   
   # macOS
   find /Applications -name dsi_studio -type f
   ```

3. Re-run installation with the correct path:
   ```bash
   bash install.sh --dsi-path /correct/path/to/dsi_studio
   ```

### Error: "DSI Studio failed to run or does not support --version"

**Cause**: DSI Studio can't be executed or doesn't support the `--version` flag.

**Solutions**:
1. Test DSI Studio manually:
   ```bash
   /path/to/dsi_studio --version
   ```

2. If on a headless server (no display), DSI Studio requires:
   ```bash
   export QT_QPA_PLATFORM=offscreen
   /path/to/dsi_studio --version
   ```
   (This is automatically set by the pipeline)

3. Check DSI Studio installation:
   - Ensure DSI Studio is properly installed
   - Verify you have permission to execute it
   - Check system dependencies (on Linux: Qt libraries)

### Error: "--dsi-path is required"

**Cause**: You forgot to provide the `--dsi-path` argument.

**Solution**:
```bash
bash install.sh --dsi-path /path/to/dsi_studio
```

Or for help:
```bash
bash install.sh --help
```

### Error: "Unknown option"

**Cause**: Invalid command-line argument provided.

**Solution**:
```bash
bash install.sh --help
# Shows valid options
```

---

## Re-running Installation

To reinstall or change the DSI Studio path:

```bash
# Remove the existing virtual environment
rm -rf braingraph_pipeline

# Run installation with new path
bash install.sh --dsi-path /new/path/to/dsi_studio

# Activate the environment
source braingraph_pipeline/bin/activate
```

---

## Next Steps

After successful installation:

1. **Verify the setup**:
   ```bash
   python scripts/validate_setup.py --config configs/braingraph_default_config.json
   ```

2. **Check DSI Studio connectivity**:
   ```bash
   dsi_studio --version
   ```

3. **Try a quick test run**:
   ```bash
   python opticonn.py sweep --quick -o tests/demo_sweep
   ```

---

## Configuration Files

All pipeline configuration files now use a generic `"dsi_studio"` command entry instead of hardcoded paths. This works because the `DSI_STUDIO_PATH` environment variable (set during installation) ensures the correct executable is found:

**In JSON config files:**
```json
{
  "dsi_studio_cmd": "dsi_studio",
  ...
}
```

**How it works**:
1. Scripts read `dsi_studio_cmd` from the config file
2. `scripts/utils/runtime.py` receives the command string
3. It expands `dsi_studio` to the full path via the `DSI_STUDIO_PATH` environment variable
4. The subprocess receives the full executable path

---

## For System Administrators

### System-Wide Installation

To set up DSI Studio as a system-wide command, install it to `/usr/local/bin`:

```bash
# Download and install DSI Studio
wget https://dsi-studio.labsolver.org/download/dsi_studio
sudo cp dsi_studio /usr/local/bin/
sudo chmod +x /usr/local/bin/dsi_studio

# Then install braingraph pipeline
bash install.sh --dsi-path /usr/local/bin/dsi_studio
```

### Docker Deployment

For Docker deployments, ensure DSI Studio is installed in the container and pass the path:

```dockerfile
FROM python:3.10

# Install DSI Studio
RUN apt-get install -y dsi-studio
# or download from https://dsi-studio.labsolver.org/

# Install braingraph pipeline
WORKDIR /app
COPY . .
RUN bash install.sh --dsi-path /path/to/dsi_studio/in/container
```

---

## Related Documentation

- [DSI Studio Path Configuration](DSI_STUDIO_PATH_CONFIG.md) - Technical details on path handling
- [Headless Server Setup](HEADLESS_SERVER_FIX.md) - Configuration for headless/remote servers
- [README.md](README.md) - Main pipeline documentation
