# Quick Test Commands - Ultra-Minimal Sweep

## Option 1: Ultra-Minimal (Fastest - 2-3 minutes)
```bash
cd /data/local/software/braingraph-pipeline
source braingraph_pipeline/bin/activate

python opticonn.py sweep \
  -i /data/local/Poly/derivatives/meta/fz/ \
  -o /tmp/demo_sweep_test \
  --extraction-config configs/sweep_ultra_minimal.json \
  --subjects 1 \
  --no-emoji \
  --verbose
```

**Config details:**
- Tract count: 1000-5000 (very small)
- Atlases: 1 only (FreeSurferDKT_Cortical)
- Connectivity values: 1 only (count)
- Combinations: 2 (1000 tracts, 5000 tracts)
- Expected time: 2-3 minutes per combo

---

## Option 2: Quick Test (Slightly Larger - 5-10 minutes)
```bash
python opticonn.py sweep \
  -i /data/local/Poly/derivatives/meta/fz/ \
  -o /tmp/demo_sweep_quick \
  --extraction-config configs/sweep_nano.json \
  --subjects 1 \
  --no-emoji
```

**Config details:**
- Uses existing `sweep_nano.json`
- Tract count: 10,000-50,000
- Expected time: 5-10 minutes per combo

---

## Option 3: Modify Existing Config
If you want to use `--quick` flag but with lower tracts, edit `configs/sweep_micro.json`:

```json
{
  "tract_count": 5000,
  "tract_count_range": [1000, 5000]
}
```

---

## After the Sweep Completes

### View results:
```bash
cd /tmp/demo_sweep_test/sweep-*/optimize/bootstrap_qa_wave_1/
head -20 combo_diagnostics.csv
```

### Auto-select best combo:
```bash
python opticonn.py review \
  -i /tmp/demo_sweep_test/sweep-*/optimize \
  --no-emoji
```

### Review with interactive GUI:
```bash
python opticonn.py review \
  -i /tmp/demo_sweep_test/sweep-*/optimize \
  --interactive \
  --no-emoji
```

---

## Configuration Recommendations

### For Different Test Sizes:

**Ultra-fast testing (1-2 min):**
- `tract_count`: 500-1000
- Atlases: 1
- Connectivity values: 1

**Quick testing (5 min):**
- `tract_count`: 5000-10000
- Atlases: 1-2
- Connectivity values: 1-2

**Standard testing (10-20 min):**
- `tract_count`: 25000-50000
- Atlases: 2-3
- Connectivity values: 2-3

**Production (1-2 hours):**
- `tract_count`: 500000+
- Atlases: 3-4
- Connectivity values: 3-4

---

## Troubleshooting

### "Still too slow"
→ Reduce `tract_count` further (try 500)
→ Set `thread_count: 2` (uses fewer CPU cores)
→ Use `--subjects 1` (only 1 subject for testing)

### "DSI Studio not found"
→ Check: `which dsi_studio` or `echo $DSI_STUDIO_PATH`
→ Update `dsi_studio_cmd` in config if needed

### "No .fz files found"
→ Verify path: `ls /data/local/Poly/derivatives/meta/fz/*.fz`
→ Use different data source if needed

