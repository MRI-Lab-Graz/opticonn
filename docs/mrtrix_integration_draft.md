# MRtrix tractography-optimization backend (draft)

## Goal
Enable OptiConn to **optimize tractography parameters** for structural connectomics using an MRtrix3 workflow.

The intended execution path is:

1. **QSIPrep**: robust diffusion preprocessing and spatial references.
2. **QSIRecon** (MRtrix workflow, e.g. `mrtrix_multishell_msmt_ACT-hsvs`): produces a *tracking-ready bundle* (FODs + ACT tissues + parcellations).
3. **OptiConn (new MRtrix backend)**: repeatedly runs MRtrix tractography + connectome construction while tuning parameters (Bayesian optimization / sweeps), then scores/selects the best parameterizations.

This replaces the DSI Studio `.fz` / `.fib.gz` role with an MRtrix **bundle** (multiple files) that is stable across iterations.

## Current status (as of 2025-12-16)

What we have working end-to-end for a single subject/session (pilot):

- **Bundle manifest created** for `sub-1293171/ses-3` (Brainnetome246Ext)
   - File: `studies/mrtrix_bundle_manifests/sub-1293171_ses-3.json`
   - Confirms presence of WM FOD, hsvs probseg, Brainnetome parcellation dseg, and labels lookup.
- **Single-iteration MRtrix pilot runner implemented**
   - File: `scripts/mrtrix_pilot_connectome.py`
   - Generates a new tractogram (`tckgen`) and a connectome (`tck2connectome`) and writes an OptiConn-compatible `*.connectivity.csv`.
- **Graph/network metrics computation implemented (external to MRtrix)**
   - File: `scripts/compute_network_measures_from_connectivity.py`
   - Uses **Python + NetworkX** to compute summary graph measures from the exported `*.connectivity.csv` (see “How graph metrics are computed”).
- **OptiConn QA scoring runs on MRtrix outputs**
   - Uses existing OptiConn components unchanged:
      - `scripts/aggregate_network_measures.py`
      - `scripts/metric_optimizer.py`
- **Parameter tuning runner implemented (sweep + Bayesian optimization)**
   - File: `scripts/mrtrix_tune.py`
   - Supports sampling parameter sets (“thetas”), running MRtrix, computing network measures, aggregating, and scoring by `quality_score_raw`.
   - Example config for the pilot subject:
      - `studies/pilot_mrtrix/sub-1293171/ses-3/mrtrix_tune_config.json`

## Non-goals
- Re-implement QSIPrep/QSIRecon.
- Re-run preprocessing or model fitting inside OptiConn for each iteration.
- Provide a “one true pipeline” for every acquisition scheme; this draft focuses on MSMT + ACT (hsvs).

## Why this approach
- Shifts heavy preprocessing/model-fitting to QSIPrep/QSIRecon.
- Lets OptiConn focus on what it’s good at: comparing many candidate tractography settings and selecting the best connectome outputs.
- Improves container reproducibility vs relying on DSI Studio execution within Docker.

## Tracking-ready input bundle (MRtrix analogue of `.fz`)
The MRtrix backend should treat the following as the minimum stable inputs per subject/session:

Required:
- **WM FOD** (tracking substrate): `*_label-WM*_dwimap.mif.gz`
- **ACT 5TT / hsvs tissue segmentation** aligned to the tracking space (DWI/ACPC): hsvs probability/segmentation NIfTI or an MRtrix 5TT `.mif` if available
- **Parcellation label image** aligned to the same space as tractography endpoints: `*_space-ACPC_seg-<atlas>_dseg.nii.gz` (or `.mif.gz`)
- **Parcellation labels lookup**: TSV or whitespace-delimited text mapping integer label → region name

Recommended:
- **Brain / DWI mask** (if used for seeding or pruning): `*_desc-brain_mask.nii.gz` or MRtrix mask
- **Reference image in tracking space** (for sanity checks): QSIRecon `*_dwiref.nii.gz` or equivalent

Notes:
- MRtrix “single file” equivalence does not exist; reproducibility comes from tracking these specific inputs and their provenance (QSIPrep/QSIRecon version + recon-spec).
- Space consistency is critical: FOD, 5TT, and parcellation must share the same voxel grid and orientation.

## Iteration loop (per candidate parameter set)
For each parameter set $\theta$:

1. `tckgen` (tractography)
   - inputs: WM FOD (+ optional ACT tissues)
   - output: `tractogram.tck` (optionally compressed)
2. `tcksift2` (weights)
   - inputs: tractogram + FOD (+ ACT tissues)
   - outputs: `weights.csv`, `mu.txt`
3. `tck2connectome` (connectome)
   - inputs: tractogram + parcellation
   - outputs: one or more connectome matrices (count, mean length, and/or others)
4. Convert output to the OptiConn Step01 layout (`results/<atlas>/*.connectivity.csv`) so existing aggregation/scoring works.

### IMPORTANT: do not use `count` as the only connectome measure
We should **always emit and score multiple connectome “edge weights”**, because different metrics probe different failure modes.

At minimum, emit:
- `count` (streamline count; baseline)
- `meanlength` (mean streamline length per edge; sanity check on geometry)

When available, also emit:
- `sift2_count` (count/connectivity weighted by SIFT2 streamline weights)

Potential extensions (requires additional inputs / transforms):
- `fa` / `qa` / `rd` / other microstructure-weighted connectomes (requires scalar images aligned to tractography space and `tcksample`/`tck2connectome` workflows)

## Candidate search space (initial)
Start with a conservative parameter set that is widely used in MRtrix ACT workflows, then tune.

Suggested tunables (examples):
- `-select` (number of streamlines)
- `-seed_dynamic` vs GM/WM interface seeding (if available)
- `-cutoff` (FOD amplitude cutoff)
- `-angle` (max curvature)
- `-step` (step size)
- `-minlength`, `-maxlength`
- `-power` (FOD power)
- `-backtrack` (bool)
- `-crop_at_gmwmi` (bool)

Keep the initial Bayesian space small to avoid expensive exploration.

## Output contract (what OptiConn should consume)
OptiConn’s current analysis steps can already consume `.connectivity.csv` outputs.

Proposed contract for this backend:
- For each candidate $\theta$, emit:
  - `.../01_connectivity/<run>/<theta_id>/results/<atlas>/<subject>_<atlas>.<metric>.connectivity.csv`
  - optional per-run metadata: command lines, MRtrix versions, timing, and parameter JSON.

## How graph metrics are computed
Graph metrics are currently computed **outside MRtrix** (MRtrix produces connectomes; it does not compute the OptiConn QA graph measures directly).

Implementation:
- `scripts/compute_network_measures_from_connectivity.py`
- Reads `*.connectivity.csv` and computes summary measures using **NetworkX**:
   - `density`
   - `global_efficiency(binary)` and `clustering_coeff_average(binary)` on the binarized adjacency ($w>0$)
   - Optional `small_worldness(binary)` via NetworkX `sigma()` (slower)
   - `clustering_coeff_average(weighted)` using edge weights
   - `global_efficiency(weighted)` using shortest-path distances defined as $d=1/w$ for $w>0$

Metric interpretation note:
- For strength-like matrices (e.g., `count`, SIFT2-weighted count): weighted shortest-path distances use $d=1/w$.
- For distance-like matrices (e.g., `meanlength`): weighted shortest-path distances use $d=w$ (and weights are inverted for clustering calculations).

Rationale:
- This keeps the MRtrix backend focused on tractography/connectome construction.
- It reuses OptiConn’s existing aggregation + scoring logic unchanged.

## Milestones
M0 — Draft/spec
- Write this doc + a config template + schema.

M1 — Bundle discovery + validation
- Given a QSIRecon derivatives directory, discover required files.
- Validate space compatibility (voxel sizes, affines, dimensions) conservatively.

Status: DONE for one concrete bundle via a manifest (manual discovery); automated discovery/validation is still pending.

M2 — Single-iteration runner
- Run one tractography iteration (tckgen → tcksift2 → tck2connectome) for one subject and one atlas.
- Export `.connectivity.csv` compatible with current aggregation.

Status: DONE for `tckgen → tck2connectome`; `tcksift2` is supported optionally.

M3 — Batch + caching
- Run multiple parameter sets per subject with caching.
- Avoid re-running expensive steps when outputs exist.

Status: PARTIAL (multiple-theta runs are implemented; caching policy and resume logic should be tightened).

M4 — Bayesian optimization integration
- Plug the MRtrix backend into OptiConn’s `tune-bayes` flow.
- Define the parameter search space + objective function using existing QA metrics.

Status: DONE as a standalone MRtrix tuner (`scripts/mrtrix_tune.py`). Wiring into `opticonn_hub.py` is still pending.

M5 — Reproducibility and testing
- Smoke tests that do not require MRtrix (schema + path validation).
- Optional integration tests when MRtrix is available.

M6 — Documentation + demo
- Document how to run QSIPrep/QSIRecon to produce the input bundle.
- Provide an OptiConn demo command that operates on a small dataset.

## Open questions
- Do we treat the **atlas choice** as part of the optimization, or fix it and only tune tractography?
- Do we always run `tcksift2`, or allow a “no-SIFT2” mode for speed?
- Which connectome measure set becomes the default scoring bundle (count + meanlength + sift2_count)?
- Should the objective combine multiple measures (e.g., average/robust aggregate of quality scores), or select the best-performing measure per theta?

