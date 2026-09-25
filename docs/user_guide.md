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

**Read the result honestly.** Discriminability is a rejection filter, not a ranking: in two independent cohorts it was exactly 1.000 for every candidate, at every streamline count and tracking method tested. That is structural — it asks only whether a repeat is closer than another subject, never by how much — and it is not fixed by adding subjects. Ties are the expected outcome.

**Before quoting a winner, check two things.** Does the winner agree between wave 1 and wave 2? And does its lead over the runner-up exceed the wave-to-wave variation? In our cohorts the top two candidates differed by 0.0013 with a wave-to-wave shift of 0.0090 — they were tied, and differed in exactly one parameter. The honest report was "FA and angle are determined by the data, track/voxel ratio is not".

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
  --subjects 10 \
  --max-parallel 2
```

- `--subjects` counts subjects (default 10; 3 with `--quick`). OptiConn uses one scan per subject, the first session, because later sessions usually carry the effect you are studying. Fewer than 10 subjects gives noisy reliability estimates, and the reports flag it.
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

A scan whose preprocessed DWI degraded during preprocessing distorts the comparison. In study 129, two scans with implausible tractography turned out to share one signature in the qsiprep QC files: the final preprocessed DWI contrast (`t1post_dwi_contrast`) was far below the cohort, although the raw data were normal.

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

If you exclude a subject's first-session scan, that subject is dropped; OptiConn does not fall back to a later session.

The QC files are small annexed text files; fetch them with `git annex get` if they are missing.

## 5. Reading the results

After a two-wave sweep, look in the output directory for:

- `combo_diagnostics.csv` and the per-combination JSON: discriminability, `discriminability_margin`, repeatability, density, tract count and rejection reasons.
- `variance_decomposition.csv` and `variance_decomposition_summary.txt`: how far the connectome moves under each source of variation, as distributions of dissimilarity (1 - r):

| Stratum | Compares |
| --- | --- |
| `tracking_noise` | the same scan, same parameters, different seed |
| `parameter` | the same scan, different parameter sets |
| `between_subject` | different subjects, same parameters |

The headline ratio is `parameter / between_subject`: how far the parameter choice moves a connectome, relative to the difference between two people. A value near 0.5 means switching between reasonable settings moves a connectome half as far as swapping in a different person, which is large for any group analysis. Strata with fewer than 10 pairs are flagged as low confidence. The decomposition is diagnostic only; it never influences ranking, because minimising parameter sensitivity would reward settings that flatten real differences.

- `graph_icc.csv` and `graph_icc_summary.txt`: for each candidate and each graph measure, how reliably that measure ranks subjects despite tracking noise (ICC with a 95% confidence interval). Look at the measures you plan to analyse: candidates that tie on discriminability often differ here, and different measures can favour different candidates. ICC is reported, not ranked; see [Methods](methods.md) for why a high ICC is necessary but not sufficient.

If every candidate ties on discriminability, that is the saturation described above. Consult the margin and repeatability columns, and widen the parameter range if you need the screen to discriminate.

### How much does the choice actually matter?

This is the question the reports exist to answer, and it is cohort-specific — do not take numbers from a paper, measure them on your data.

- **`parameter / between_subject`** — expect roughly 0.2 to 0.45. At 0.44, switching between two defensible settings moves a connectome nearly half as far as swapping in a different person.
- **`tracking_noise / parameter`** — if this exceeds about 0.5, your streamline count is too low for the screen to see parameter differences at all. Raise `tract_count` before adding subjects. At 5k it was 65-76%; at 50k, 22-48%.
- **Subject ordering on graph measures** — re-running identical settings preserves it at about 0.93; changing settings drops it to 0.66-0.75. A quarter to a third of the ordering your group analysis rests on is contingent on the parameter choice.
- **Effective dimensionality** — the seven global graph measures carry only about two independent dimensions. Do not report them as seven independent findings.

### Already finished an analysis?

Do not redo it because a screen preferred different parameters; leading candidates typically differ by less than the noise between them, and the criterion measures identifiability rather than correctness. Instead re-run a subset under an alternative setting and check whether your *conclusions* hold. A reported robustness check is worth more than a claim of optimal parameters.

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
- **"data_selection.sessions_per_subject was removed".** Delete that key from your wave configs; OptiConn now always uses one first-session scan per subject.
- **The same scan appears twice / subject names look like `MD5E-...`.** Old versions listed both the annex symlink and the object. Update; discovery now excludes `.git`.
- **The run is too slow.** Lower `tract_count` or `--subjects`, reduce the grid, or use `--max-parallel`.
- **DSI Studio not found.** Set the path during installation, or export `DSI_STUDIO_PATH`.
