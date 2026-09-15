# OptiConn

Data-driven screening of tractography parameters.

There is no gold standard for "correct" tractography settings. OptiConn does
not claim to find an optimal parameter set. Instead it sweeps tracking
parameters on a few subjects, tracks every subject several times with
different random seeds, and ranks each setting by explicit, testable
criteria: how well its connectomes tell subjects apart compared with
tracking noise (**discriminability**), and repeat-run reliability. Settings
that produce implausible graphs are rejected first. The top-ranked, defensible
setting is then applied to all subjects — the goal is a transparent,
reproducible choice, not a proof of optimality.

## Install

Python ≥ 3.12 and one tractography backend:

- **MRtrix3** (recommended, open source): `tckgen` and `tck2connectome` on
  `PATH`, e.g. via `conda install -c mrtrix3 mrtrix3`.
- **DSI Studio** (optional): path to the `dsi_studio` executable.

```bash
./install.sh                                   # MRtrix3 backend
./install.sh --dsi-studio /path/to/dsi_studio  # also set up DSI Studio

source .venv/bin/activate
python opticonn.py validate --backend mrtrix3  # or --backend dsi_studio
```

Both backends run the same workflow (sweep, reliability scoring, top
candidates, apply). The backend is chosen by `"backend"` in the sweep config
(`"mrtrix3"` or `"dsi_studio"`, the default).

DSI Studio path precedence: `--dsi-studio` flag, then `DSI_STUDIO_CMD` (or
`.opticonn_config` written by the installer), then `"dsi_studio_cmd"` in the
config file.

## Inputs

**MRtrix3:** OptiConn does not preprocess diffusion data. Provide one folder
per subject (folder names without dots):

```
subjects/
  sub-01/
    wmfod.mif      white-matter FOD
    desikan.mif    parcellation, integer labels, same space as wmfod.mif (one file per atlas in "atlases")
    5tt.mif        optional, enables anatomically constrained tractography
```

A typical route is QSIPrep for preprocessing, then FODs and a parcellation in
diffusion space from QSIRecon's MRtrix3 workflows or directly with MRtrix3,
linked into the layout above. A QSIPrep/QSIRecon input adapter is planned.
With MRtrix3 alone (adapt to your acquisition; register T1-derived images to
diffusion space first):

```bash
dwi2response dhollander dwi.mif wm.txt gm.txt csf.txt
dwi2fod msmt_csd dwi.mif -mask mask.mif wm.txt wm_fod.mif gm.txt gm.mif csf.txt csf.mif
mtnormalise wm_fod.mif wmfod.mif gm.mif gm_norm.mif csf.mif csf_norm.mif -mask mask.mif
labelconvert aparc+aseg.mgz FreeSurferColorLUT.txt fs_default.txt desikan.mif   # fs_default.txt ships with MRtrix3
5ttgen fsl T1_in_dwi.mif 5tt.mif                                               # optional
```

Swept options map to `tckgen`: `cutoff`, `angle`, `step`, `min_length`
(`-minlength`), `max_length` (`-maxlength`), `algorithm`, and `tract_count`
(`-select`); 0 keeps the MRtrix3 default. Seeding is `-seed_dynamic
wmfod.mif`, connectomes are streamline counts (`tck2connectome -symmetric
-zero_diagonal`), and repeats vary `MRTRIX_RNG_SEED`. The MRtrix3 backend
writes matrices only (no network-measures table).

**DSI Studio:** a folder of `.fz` / `.fib.gz` files.

## Phase 1: screen parameters

Run from the repository root:

```bash
# MRtrix3
python opticonn.py sweep --config configs/mrtrix_quick_sweep.json --data subjects --subjects 3
# DSI Studio example data
python opticonn.py sweep --data examples/data/fib_samples --quick --subjects 3
```

Outputs in `results/sweep_run_<timestamp>_<id>/optimize/`:

| File | Content |
|------|---------|
| `optimization_results/top3_candidates.json` | Top three recommended candidates (parameters, atlas, metric) with scores; input to Phase 2 |
| `optimization_results/ranked_candidates.json` | Every candidate that passed the gates, best first |
| `optimization_results/selected_parameters.json` | Full config of the top-ranked candidate |
| `comprehensive_optimization/combo_diagnostics.csv` | Every candidate, including rejected ones and why |

## Phase 2: apply to all subjects

```bash
python opticonn.py apply \
  --config results/sweep_run_<timestamp>_<id>/optimize/optimization_results/top3_candidates.json \
  --data /path/to/subjects            # --candidate 2 for the runner-up
```

Outputs in `analysis/apply_run_<timestamp>_<id>/selected/01_connectivity/`:
one folder per subject with `*.connectivity.mat` matrices (current DSI
Studio builds write one combined matrix per atlas per subject). If the
installed DSI Studio build also writes network-measures files, an
`aggregated_network_measures.csv` with graph measures for all subjects
appears there too; otherwise Phase 2 logs a warning and skips it.

`python opticonn.py auto --data …` runs both phases.

## How candidates are scored

| Field | Meaning |
|-------|---------|
| `discriminability` (`average_score`) | Probability that a repeat of a subject is closer (1 − Pearson r of log edge weights) than a repeat of another subject. 0.5 = chance, 1.0 = perfect. Ranking key. |
| `repeatability` | Mean correlation between repeats of the same subject. First tie-break; fewer tracts is the second. |
| `density`, `isolated_fraction` | Graph plausibility; a candidate is rejected outside `reliability.density_range` or above `reliability.max_isolated_fraction`. |
| `loo_top1_frequency` | Share of leave-one-subject-out rankings in which the candidate is still first (top 10 only, needs ≥ 3 subjects). |

With DSI Studio, Phase 1 scoring reads the `count`, `fa`, and `qa`
connectivity metrics from each subject's `*.connectivity.mat` matrix (this
mapping matches current DSI Studio builds' combined `.mat` output); the
MRtrix3 backend produces `count` matrices only. Rankings across different edge weights
(count vs. fa vs. qa) are not like-for-like: count connectomes are
structurally more repeatable, so comparing a top `count` candidate against a
top `fa` or `qa` candidate is not an apples-to-apples comparison.

## What OptiConn does not do

There is no gold standard, so OptiConn cannot verify a setting is "correct" —
only that it is reliable and plausible by the criteria above. Seed repeats
measure tracking noise only, not accuracy against ground truth. With few
subjects, discriminability often reaches 1.0 for many settings and
repeatability decides the ranking instead. Held-out (scan–rescan) validation,
QA confound checks (e.g. motion), and a parameter-sensitivity report are
planned but not yet implemented — see `docs/ROADMAP.md` §3.3–3.4.

## Configuration

`configs/quick_sweep.json` (minutes), `default_sweep.json`, `comprehensive_sweep.json`.

```json
"sweep_parameters": {
  "fa_threshold_range": [0.05, 0.10, 0.15, 0.20],
  "min_length_range": [10, 20, 30],
  "tract_count_range": [250000, 500000, 1000000],
  "sampling": {"method": "grid"}
},
"reliability": {
  "repeats": 2,
  "density_range": [0.02, 0.6],
  "max_isolated_fraction": 0.1
}
```

Any `<name>_range` sweeps `<name>` (top-level key or inside
`tracking_parameters`). Sampling: `grid`, or `random` with `n_samples`.

Note: DSI Studio ignores `--connectivity_threshold`, so it is omitted from
the example above.

## Development

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests -q
```

See `docs/ROADMAP.md` for status and plans.
