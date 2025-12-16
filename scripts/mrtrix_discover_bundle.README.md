# `mrtrix_discover_bundle.py`

Discover a tracking-ready MRtrix bundle from QSIRecon outputs and generate a `scripts/mrtrix_tune.py` config JSON.

This is designed for two common setups:

1) Explicit QSI folders
- You provide `--qsirecon-dir` (and optionally `--qsiprep-dir`).

2) Single derivatives root (lab-style)
- You provide `--derivatives-dir` and the script auto-finds `qsirecon/` and `qsiprep/` under it.

## What it discovers

- `wm_fod`: WM FOD `.mif/.mif.gz`
- `act_5tt_or_hsvs`: HSVS probseg or 5TT (optional; required only if you run tuning with `--enable-act`)
- `dseg` + `labels_tsv`: atlas parcellation and label lookup

## Examples

### PK01 layout (derivatives root)

```bash
python scripts/mrtrix_discover_bundle.py \
  --derivatives-dir /data/local/129_PK01/derivatives \
  --subject sub-1293171 \
  --session ses-3 \
  --atlas Brainnetome246Ext
```

By default this writes the config under:
- `/data/local/129_PK01/derivatives/opticonn/mrtrix_tune_configs/<subject>_<session>_<atlas>/mrtrix_tune_config.json`

### Explicit QSI folders

```bash
python scripts/mrtrix_discover_bundle.py \
  --qsirecon-dir /path/to/derivatives/qsirecon \
  --qsiprep-dir /path/to/derivatives/qsiprep \
  --subject sub-1293171 \
  --session ses-3 \
  --atlas Brainnetome246Ext
```

### Dry-run

```bash
python scripts/mrtrix_discover_bundle.py \
  --derivatives-dir /data/local/129_PK01/derivatives \
  --subject sub-1293171 \
  --session ses-3 \
  --atlas Brainnetome246Ext \
  --dry-run
```

## Next step: run a small sweep

You can either use the generated config JSON:

```bash
python scripts/mrtrix_tune.py sweep \
  --config /data/local/129_PK01/derivatives/opticonn/mrtrix_tune_configs/sub-1293171_ses-3_Brainnetome246Ext/mrtrix_tune_config.json \
  --output-dir /data/local/129_PK01/derivatives/opticonn/mrtrix_runs \
  --subject sub-1293171 \
  --atlas Brainnetome246Ext \
  --n-samples 3 \
  --max-evals 3

Or run one-step discovery mode directly (no config file needed):

```bash
python scripts/mrtrix_tune.py sweep \
  --derivatives-dir /data/local/129_PK01/derivatives \
  --subject sub-1293171 \
  --session ses-3 \
  --atlas Brainnetome246Ext \
  --output-dir /data/local/129_PK01/derivatives/opticonn/mrtrix_runs \
  --run-name smoke_sub-1293171_ses-3 \
  --n-samples 3 \
  --max-evals 3
```
```
