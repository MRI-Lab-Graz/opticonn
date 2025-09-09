# Testing & Development Checklist

## 🔧 **Pre-Test Setup (ALWAYS DO FIRST!)**

### 1. Activate Python Environment
```bash
# Navigate to project directory
cd /Volumes/Work/github/braingraph-pipeline

# Activate the virtual environment
source braingraph_pipeline/bin/activate

# Verify activation (should show env name in prompt)
which python
python --version
```

### 2. Verify DSI Studio

```bash
# Check DSI Studio is accessible (don't run --help as it launches GUI!)
ls -la /Applications/dsi_studio.app/Contents/MacOS/dsi_studio

# Verify it's executable
file /Applications/dsi_studio.app/Contents/MacOS/dsi_studio
```

### 3. Check Git Status
```bash
# See current changes
git status

# Check current branch
git branch
```

## 🧪 **Testing Workflow**

### Basic CLI Tests
```bash
# 1. Test main CLI help
python braingraph.py --help

# 2. Test subcommands
python braingraph.py optimize --help
python braingraph.py analyze --help
python braingraph.py pipeline --help

# 3. Test run_pipeline.py
python run_pipeline.py --help
```

### Script Path Validation
```bash
# 4. Test script imports work
python -c "import scripts.cross_validation_bootstrap_optimizer"
python -c "import scripts.validate_setup"

# 5. Verify all scripts are found
ls -la scripts/*.py
```

### Configuration Tests
```bash
# 6. Validate main config
python scripts/json_validator.py configs/01_working_config.json

# 7. Test setup validation
python scripts/validate_setup.py --config configs/01_working_config.json
```

## 🎯 **Full Workflow Test**

### Dry Run Test (Safe)
```bash
# Test optimize command structure (will fail gracefully without data)
python braingraph.py optimize --data-dir /tmp/test --output-dir /tmp/output
```

## 📝 **Common Gotchas Checklist**

- [ ] Environment activated? (`source braingraph_pipeline/bin/activate`)
- [ ] In correct directory? (`cd /Volumes/Work/github/braingraph-pipeline`)
- [ ] All script paths updated after moves?
- [ ] Git status clean before major changes?
- [ ] DSI Studio accessible?
- [ ] Configuration files valid JSON?

## 🚨 **Before Any Major Changes**

1. [ ] Environment activated
2. [ ] Git status checked
3. [ ] Basic tests passing
4. [ ] Backup important files if needed

## 📋 **After File Reorganization**

1. [ ] Update all import statements
2. [ ] Update subprocess calls with new paths
3. [ ] Update help messages and examples
4. [ ] Test all entry points
5. [ ] Validate configuration paths
6. [ ] Run basic smoke tests

---
**💡 TIP**: Always run this checklist before testing anything!
