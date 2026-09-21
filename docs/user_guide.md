# OptiConn User Guide

OptiConn helps you choose tractography parameters for structural connectomes. There is no ground truth for "correct" parameters, so it does not claim to find them. It screens candidate parameter sets by how reproducibly they recover each person's connectome, tells you how much the choice matters compared with the effects you study, and reports what it measured so you can decide.

This guide walks from a set of reconstructed scans to a chosen parameter set. Method details are in [Methods](methods.md); every flag is in `opticonn <command> --help`.

## 1. What you need

- Reconstructed DSI Studio scans: `.fz` or `.fib.gz` files (QSDR/GQI), one per scan. Names should carry BIDS identifiers (`sub-001_ses-1...`), because OptiConn groups scans by subject and session from the file name.
- DSI Studio and Python 3.10+. See [Installation](installation.md); `install.sh` builds a dedicated virtual environment.
- A base configuration (start from `configs/braingraph_default_config.json`) naming the atlases, connectivity metrics and the parameter ranges to sweep.
- Time. Every candidate is tracked `repeats` times on every sampled scan (default 2 repeats). Cost grows with candidates x scans x repeats x `tract_count`.

DataLad/git-annex datasets work, including datasets where `fib/` holds friendly symlinks: OptiConn ignores `.git` internals and de-duplicates by resolved path. Fetch the content first (`datalad get`).

## 2. How OptiConn decides

Each candidate parameter set is run several times on the same scans with different random seeds. Its connectomes are then compared:

| Quantity | Question it answers |
| --- | --- |
| Gates (density, isolated nodes) | Is this an implausible graph? Rejected candidates are excluded, not penalised. |
| Discriminability | Is a repeat of a subject closer to that subject than to other subjects? 1.0 means every scan is identifiable; 0.5 is chance. |
| Discriminability margin | How much closer? Unlike discriminability it does not cap at 1.0, so it separates candidates that all score 1.0. |
| Repeatability | Mean correlation between repeat runs of the same scan (1.0 means no tracking noise). |

Candidates are ranked by discriminability, then margin, then repeatability, then fewer tracts.

**Read the result honestly.** Discriminability is a rejection filter: reasonable candidates typically all reach 1.0, because tracking noise (r about 0.997 between seeds) is far below between-subject differences. Ties are expected and do not mean the candidates are equivalent. The variance decomposition (section 5) shows how much the parameters actually move the connectome compared with biology.

## 3. A first run

`tune-grid` sweeps a grid of parameters with two-wave cross-validation (wave 1 and wave 2 use different subjects, and a winner has to hold up in both).

```console
source braingraph_pipeline/bin/activate

# A quick check that everything is wired up
opticonn tune-grid -i /data/study/derivatives/dsistudio -o runs/quick --quick

# A real run
opticonn tune-grid \
  -i /data/study/derivatives/dsistudio \
  -o runs/study1 \
  --extraction-config configs/study_config.json \
  --subjects 5 --sessions-per-subject 2 \
  --max-parallel 2
```

- `--subjects` counts subjects, not scans. With `--sessions-per-subject 2` (the default), each sampled subject contributes two sessions, so five subjects stage up to ten scans per wave. Use `--sessions-per-subject 1` (or `0`) for legacy scan-level sampling. If no subject has enough sessions, OptiConn falls back to scan-level sampling.
- The `--sessions-per-subject` flag only applies to auto-generated waves. With `--wave1-config`/`--wave2-config`, or a master config that embeds waves, set `data_selection.sessions_per_subject` inside those files. OptiConn warns if you pass the flag and it is ignored.
- Use `--dry-run` (top-level option) to print the commands without running them.

Choose the ranges to sweep in the `sweep_parameters` of your config. Vary the parameters you actually care about, not everything at once: each added value multiplies the run time.

### Bayesian search (optional)

`tune-bayes` proposes candidates with a Gaussian-process search and is cheaper than a full grid over many parameters:

```console
opticonn tune-bayes -i /data/study/derivatives/dsistudio -o runs/bayes \
  --config configs/study_config.json --n-iterations 30 --sample-subjects
```

Its acquisition function optimises the composite quality score, which is descriptive context, not the selection criterion. Treat its output as a source of candidates and confirm them with `tune-grid` (the `--candidates-from-bayes` option in `python scripts/cross_validation_bootstrap_optimizer.py --help`).

## 4. Screen your data first

A scan whose preprocessed DWI degraded during preprocessing distorts the comparison. In study 129, two scans with implausible between-session changes turned out to share one signature in the qsiprep QC files: the final preprocessed DWI contrast (`t1post_dwi_contrast`) was far below the cohort, although the raw data were normal.

```console
python -m scripts.qc_gate /data/study/derivatives/qsiprep -o qc_flags.csv
```

It reads every `*image_qc.tsv`, flags scans whose contrast is a robust outlier (median/MAD z below -3.5) and refuses to flag anything with fewer than 10 scans. Flags are advisory: inspect them, then exclude the scans you decide against in the wave configs:

```json
"data_selection": {
  "source_dir": "/data/study/derivatives/dsistudio",
  "exclude_scans": ["sub-043_ses-1", "sub-096_ses-2"]
}
```

The QC files are small annexed text files; fetch them with `git annex get` if they are missing.

## 5. Reading the results

After a two-wave sweep, look in the output directory for:

- `combo_diagnostics.csv` and the per-combination JSON: discriminability, `discriminability_margin`, repeatability, density, tract count and rejection reasons.
- `variance_decomposition.csv` and `variance_decomposition_summary.txt`: how far the connectome moves under each source of variation, as distributions of dissimilarity (1 - r):

| Stratum | Compares |
| --- | --- |
| `tracking_noise` | the same scan, same parameters, different seed |
| `parameter` | the same scan, different parameter sets |
| `between_session` | the same subject, different sessions, same parameters (multi-session cohorts only) |
| `between_subject` | different subjects, same parameters |

The headline ratio is `parameter / between_session`. Values near or above 1 mean the parameter choice moves the connectome as much as a real change over time does, so the choice deserves care in a longitudinal study. Strata with fewer than 10 pairs are flagged as low confidence. The decomposition is diagnostic only; it never influences ranking, because minimising parameter sensitivity would reward settings that flatten real differences.

If every candidate ties on discriminability, that is the saturation described above. Consult the margin and repeatability columns, and widen the parameter range if you need the screen to discriminate.

## 6. Choosing and applying

```console
opticonn select -i runs/study1/optimize          # confirm the winner, write selected_candidate.json
opticonn apply -i /data/study/derivatives/dsistudio \
  --optimal-config runs/study1/optimize/selected_candidate.json -o runs/study1/final
```

`apply` runs the full connectivity extraction and network analysis on the whole dataset with the selected parameters. Report the candidates you screened, the reproducibility measures, and the variance decomposition alongside the chosen setting.

## 7. Other commands

- `opticonn sensitivity` perturbs each parameter of a baseline configuration and reports which ones move the quality score most.
- `--backend mrtrix` runs the same reliability screen on an MRtrix3/QSIRecon pipeline (see [Workflows](workflows.md)).

## 8. Troubleshooting

See [Troubleshooting](troubleshooting.md). Common cases:

- **Every candidate scores 1.0.** Expected; see section 2.
- **`between_session` is "not available".** The cohort staged only one session per subject; use multi-session data and `--sessions-per-subject 2`.
- **The same scan appears twice / subject names look like `MD5E-...`.** Old versions listed both the annex symlink and the object. Update; discovery now excludes `.git`.
- **The run is too slow.** Lower `tract_count` or `--subjects`, reduce the grid, or use `--max-parallel`.
- **DSI Studio not found.** Set the path during installation, or export `DSI_STUDIO_PATH`.
