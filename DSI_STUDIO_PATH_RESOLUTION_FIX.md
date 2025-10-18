# DSI Studio Path Resolution Fix

## Problem

When running Bayesian optimization, the pipeline was failing with error:
```
DSI Studio command not found: dsi_studio. Check PATH or use absolute path.
```

The issue was that while `DSI_STUDIO_PATH` environment variable was correctly set during installation, it wasn't being used to resolve the generic `"dsi_studio"` command in the JSON configuration files.

---

## Root Cause Analysis

1. **Installation Setup**: During `install.sh`, the full path to DSI Studio is stored in the `DSI_STUDIO_PATH` environment variable and added to the virtual environment activation script.

2. **Configuration Storage**: To make configs portable, all JSON files use the generic `"dsi_studio_cmd": "dsi_studio"` instead of hardcoded absolute paths.

3. **Command Resolution Gap**: When subprocesses were spawned, they received:
   - The `DSI_STUDIO_PATH` environment variable ✅
   - The generic `"dsi_studio"` command from the config ❌
   
   But there was no logic to connect them - the code tried to execute `dsi_studio` directly without resolving it via `DSI_STUDIO_PATH`.

---

## Solution Implemented

### 1. **scripts/extract_connectivity_matrices.py**

#### In `check_dsi_studio()` method (lines 292-312)
Added logic to resolve the generic `"dsi_studio"` command:

```python
# If dsi_cmd is generic "dsi_studio" command, try to resolve it using DSI_STUDIO_PATH
if dsi_cmd == "dsi_studio" and "DSI_STUDIO_PATH" in os.environ:
    resolved_path = os.environ["DSI_STUDIO_PATH"]
    if os.path.exists(resolved_path) and os.access(resolved_path, os.X_OK):
        dsi_cmd = resolved_path
        result["path"] = dsi_cmd
```

#### In `extract_connectivity_matrix()` method (lines 596-607)
Added the same resolution logic before executing DSI Studio:

```python
# If dsi_cmd is generic "dsi_studio" command, try to resolve it using DSI_STUDIO_PATH
if dsi_cmd == "dsi_studio" and "DSI_STUDIO_PATH" in os.environ:
    resolved_path = os.environ["DSI_STUDIO_PATH"]
    if os.path.exists(resolved_path) and os.access(resolved_path, os.X_OK):
        dsi_cmd = resolved_path
```

### 2. **scripts/json_validator.py**

#### Added `import os` at top of file
#### Updated DSI Studio path validation (lines 200-211)

```python
# If dsi_studio_cmd is the generic "dsi_studio" command, try to resolve it using DSI_STUDIO_PATH
if dsi_path == "dsi_studio" and "DSI_STUDIO_PATH" in os.environ:
    dsi_path = os.environ["DSI_STUDIO_PATH"]

dsi_path_obj = Path(dsi_path)
if not dsi_path_obj.exists():
    errors.append(f"DSI Studio executable not found: {dsi_path}")
elif not dsi_path_obj.is_file():
    errors.append(f"DSI Studio path is not a file: {dsi_path}")
```

### 3. **scripts/utils/json_validator.py**

#### Added `import os` at top of file
#### Updated DSI Studio path validation (same as json_validator.py)

### 4. **scripts/validate_setup.py**

#### Updated `check_dsi_studio_installation()` function (lines 22-65)
Added check for `DSI_STUDIO_PATH` environment variable as first check:

```python
# First check if DSI_STUDIO_PATH environment variable is set
if "DSI_STUDIO_PATH" in os.environ:
    dsi_path = os.environ["DSI_STUDIO_PATH"]
    if os.path.exists(dsi_path) and os.access(dsi_path, os.X_OK):
        print(f"✅ DSI Studio found via DSI_STUDIO_PATH: {dsi_path}")
        print("✅ DSI Studio is marked as executable (not launched)")
        return True, dsi_path
```

### 5. **scripts/bayesian_optimizer.py**

#### Fixed parameter display in results (lines 651-668)
Changed from direct dictionary access to using `.get()` with fallback to `param_space` attributes for fixed parameters:

```python
# Use .get() with fallback to param_space attributes for fixed parameters
tract_count = p.get('tract_count', self.param_space.tract_count[0])
fa_threshold = p.get('fa_threshold', self.param_space.fa_threshold[0])
min_length = p.get('min_length', self.param_space.min_length[0])
# etc...
```

#### Fixed iteration results display (line 688)
Same fix to handle missing keys for fixed parameters.

---

## How It Works Now

### Execution Flow

1. **Installation** (`install.sh --dsi-path /path/to/dsi_studio`):
   ```bash
   export DSI_STUDIO_PATH=/path/to/dsi_studio
   ```
   Stored in `braingraph_pipeline/bin/activate`

2. **Virtual Environment Activation**:
   ```bash
   source braingraph_pipeline/bin/activate
   echo $DSI_STUDIO_PATH  # Output: /path/to/dsi_studio
   ```

3. **Subprocess Execution** (when running pipeline):
   - `scripts/utils/runtime.py` propagates `DSI_STUDIO_PATH` to subprocess environment ✅
   - JSON config has generic `"dsi_studio_cmd": "dsi_studio"` ✅
   - When executed, code resolves:
     ```
     "dsi_studio" → DSI_STUDIO_PATH env var → "/path/to/dsi_studio"
     ```

4. **Validation** (before execution):
   - `validate_setup.py` checks `DSI_STUDIO_PATH` first
   - `json_validator.py` resolves generic command via `DSI_STUDIO_PATH`
   - All checks pass ✅

---

## Test Results

### Before Fix
```
❌ DSI Studio command not found: dsi_studio. Check PATH or use absolute path.
```

### After Fix
```
✅ DSI Studio found via DSI_STUDIO_PATH: /data/local/software/dsi-studio/dsi_studio
✅ DSI Studio is marked as executable (not launched)
✅ Configuration configs/sweep_nano.json is valid!

[Bayesian optimization completed successfully with 3 iterations]
```

### Full Optimization Run
```
INFO - ✅ BAYESIAN OPTIMIZATION COMPLETE
INFO - Best QA Score: 0.2420
INFO - Total time: 217.7 seconds (3.6 minutes)
INFO - Subjects used: 1 (P100_1002.fz)
INFO - Atlases used: FreeSurferDKT_Cortical, Brainnectome
✅ Optimization completed successfully!
```

---

## Key Benefits

✅ **Portable**: JSON configs work everywhere (no hardcoded paths)
✅ **Reliable**: DSI Studio path properly resolved at runtime
✅ **Maintainable**: Single environment variable (`DSI_STUDIO_PATH`) controls the path
✅ **Validated**: Setup validator checks path availability before execution
✅ **Fallback-safe**: Code gracefully handles both generic and absolute paths

---

## Files Modified

| File | Changes | Lines |
|------|---------|-------|
| `scripts/extract_connectivity_matrices.py` | Resolve generic command via DSI_STUDIO_PATH | 292-312, 596-607 |
| `scripts/json_validator.py` | Added os import, resolve generic command | 12, 200-211 |
| `scripts/utils/json_validator.py` | Added os import, resolve generic command | 12, 186-197 |
| `scripts/validate_setup.py` | Check DSI_STUDIO_PATH first | 22-65 |
| `scripts/bayesian_optimizer.py` | Fixed parameter handling in results | 651-668, 688 |

---

## Related Documentation

- [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) - How to install with --dsi-path flag
- [DSI_STUDIO_PATH_CONFIG.md](DSI_STUDIO_PATH_CONFIG.md) - Technical path configuration details
- [README.md](README.md) - Main pipeline documentation
