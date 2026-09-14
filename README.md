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

Requires DSI Studio (path to the `dsi_studio` executable) and Python ≥ 3.12.

```bash
./install.sh --dsi-studio /Applications/dsi_studio.app/Contents/MacOS/dsi_studio
source .venv/bin/activate
python opticonn.py validate
```

DSI Studio path precedence: `--dsi-studio` flag, then `DSI_STUDIO_CMD` (or
`.opticonn_config` written by the installer), then `"dsi_studio_cmd"` in the
config file.

## Phase 1: screen parameters

Run from the repository root:

```bash
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

Phase 1 scoring reads the `count`, `fa`, and `qa` connectivity metrics from
each subject's `*.connectivity.mat` matrix — this mapping matches current DSI
Studio builds' combined `.mat` output. Rankings across different edge weights
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
