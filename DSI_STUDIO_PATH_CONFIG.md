# DSI Studio Path Configuration

## Overview

The DSI Studio executable path is now **environment-based** rather than hardcoded in JSON configs. This ensures the pipeline works across different systems (Linux, macOS, Windows) without requiring config file modifications.

## How It Works

### 1. **Installation Phase** (`install.sh`)
When you run `./install.sh`, it:
- ✅ Asks for your DSI Studio path
- ✅ Validates that DSI Studio is installed and working
- ✅ Stores the path in the virtual environment: `braingraph_pipeline/bin/activate`

### 2. **Runtime Phase**
When you activate the environment and run the pipeline:
```bash
source braingraph_pipeline/bin/activate
```

The environment automatically has `DSI_STUDIO_PATH` set, which is used by the pipeline.

### 3. **JSON Configs**
All `configs/*.json` files now use:
```json
{
  "dsi_studio_cmd": "dsi_studio"
}
```

This means:
- ✅ **Platform-independent** - works on Linux, macOS, Windows
- ✅ **Environment-aware** - uses the path from `$PATH` or `$DSI_STUDIO_PATH`
- ✅ **No modifications needed** - one config file works everywhere

## Setting Up on a New System

### Step 1: Install DSI Studio
Follow the official DSI Studio installation guide for your platform:
- [DSI Studio Documentation](http://dsi-studio.labsolver.org/)

### Step 2: Run Installation Script
```bash
cd /data/local/software/braingraph-pipeline
./install.sh
```

When prompted, provide the **full path** to the DSI Studio executable:
```
DSI Studio path: /usr/local/bin/dsi_studio
```

Or on macOS:
```
DSI Studio path: /Applications/dsi_studio.app/Contents/MacOS/dsi_studio
```

### Step 3: Verify Setup
```bash
source braingraph_pipeline/bin/activate
echo $DSI_STUDIO_PATH  # Should print the path
dsi_studio --version   # Should show version info
```

## Troubleshooting

### Error: "DSI Studio executable not found"
- **Cause**: DSI Studio not installed or wrong path provided
- **Solution**: Install DSI Studio, then re-run `./install.sh`

### Error: "dsi_studio: command not found"
- **Cause**: DSI Studio path not in `$PATH` and environment not activated
- **Solution**: 
  ```bash
  source braingraph_pipeline/bin/activate
  ```

### DSI Studio works manually but not in pipeline
- **Cause**: Virtual environment not activated when running pipeline
- **Solution**:
  ```bash
  source braingraph_pipeline/bin/activate
  python scripts/run_pipeline.py ...  # Now use correct DSI Studio path
  ```

## Files Changed

### Modified Files
- ✅ `install.sh` - Now stores `DSI_STUDIO_PATH` in virtual environment
- ✅ `configs/braingraph_default_config.json` - Uses generic `"dsi_studio"`
- ✅ `configs/production_config.json` - Uses generic `"dsi_studio"`
- ✅ `configs/quick_test_sweep.json` - Uses generic `"dsi_studio"`
- ✅ `configs/sweep_micro.json` - Uses generic `"dsi_studio"`
- ✅ `configs/sweep_nano.json` - Uses generic `"dsi_studio"`
- ✅ `configs/sweep_probe.json` - Uses generic `"dsi_studio"`
- ✅ `configs/sweep_production_full.json` - Uses generic `"dsi_studio"`
- ✅ `configs/test_quick.json` - Uses generic `"dsi_studio"`
- ✅ `configs/user_friendly_sweep.json` - Uses generic `"dsi_studio"`

### Why This Approach

| Aspect | Before | After |
|--------|--------|-------|
| **Portability** | ❌ Hardcoded paths fail on other systems | ✅ Works anywhere |
| **Configuration** | ❌ Must edit JSON per system | ✅ One-time setup in install.sh |
| **Maintenance** | ❌ Update every config file for new path | ✅ Update install.sh only |
| **Clarity** | ❌ Hard-coded paths may be outdated | ✅ Clear what to do during install |

## Example Workflow

```bash
# First time setup
cd /data/local/software/braingraph-pipeline
./install.sh
# → Asks for DSI Studio path once
# → Stores it in virtual environment

# Later, anywhere in the system
cd /some/other/directory
source /data/local/software/braingraph-pipeline/braingraph_pipeline/bin/activate
# → DSI_STUDIO_PATH is automatically set
python /data/local/software/braingraph-pipeline/scripts/run_pipeline.py \
  --data-dir /path/to/data \
  --config /data/local/software/braingraph-pipeline/configs/production_config.json
# → Uses the DSI Studio path from environment
```

## Best Practices

1. **Always activate the environment** before running the pipeline
2. **Use the virtual environment's Python** (not system Python)
3. **Keep DSI Studio in a standard location** (e.g., `/usr/local/bin/`, `~/.local/bin/`)
4. **Don't manually edit JSON DSI Studio paths** - let `install.sh` handle it

---

**Summary**: DSI Studio path is now managed centrally during installation, making the pipeline portable and maintainable across different systems. 🚀
