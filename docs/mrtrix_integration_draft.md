# MRtrix tractography-optimization backend (draft)

## Goal
Enable OptiConn to **optimize tractography parameters** for structural connectomics using an MRtrix3 workflow.

The intended execution path is:

1. **QSIPrep**: robust diffusion preprocessing and spatial references.
2. **QSIRecon** (MRtrix workflow, e.g. `mrtrix_multishell_msmt_ACT-hsvs`): produces a *tracking-ready bundle* (FODs + ACT tissues + parcellations).
3. **OptiConn (new MRtrix backend)**: repeatedly runs MRtrix tractography + connectome construction while tuning parameters (Bayesian optimization / sweeps), then scores/selects the best parameterizations.

This replaces the DSI Studio `.fz` / `.fib.gz` role with an MRtrix **bundle** (multiple files) that is stable across iterations.

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
- **ACT 5TT / hsvs tissue segmentation** aligned to the tracking space (DWI/ACPC): `*seg-hsvs*_dseg.nii.gz` (or the MRtrix 5TT `.mif` if available)
- **Parcellation label image** aligned to the same space as tractography endpoints: `*_space-ACPC_seg-<atlas>_dseg.nii.gz` (or `.mif.gz`)
- **Parcellation labels TSV**: `*_space-ACPC_seg-<atlas>_dseg.tsv`

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
   - outputs: weighted count matrix, mean length matrix (and/or others)
4. Convert output to the OptiConn Step01 layout (`results/<atlas>/*.connectivity.csv`) so existing aggregation/scoring works.

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

## Milestones
M0 — Draft/spec
- Write this doc + a config template + schema.

M1 — Bundle discovery + validation
- Given a QSIRecon derivatives directory, discover required files.
- Validate space compatibility (voxel sizes, affines, dimensions) conservatively.

M2 — Single-iteration runner
- Run one tractography iteration (tckgen → tcksift2 → tck2connectome) for one subject and one atlas.
- Export `.connectivity.csv` compatible with current aggregation.

M3 — Batch + caching
- Run multiple parameter sets per subject with caching.
- Avoid re-running expensive steps when outputs exist.

M4 — Bayesian optimization integration
- Plug the MRtrix backend into OptiConn’s `tune-bayes` flow.
- Define the parameter search space + objective function using existing QA metrics.

M5 — Reproducibility and testing
- Smoke tests that do not require MRtrix (schema + path validation).
- Optional integration tests when MRtrix is available.

M6 — Documentation + demo
- Document how to run QSIPrep/QSIRecon to produce the input bundle.
- Provide an OptiConn demo command that operates on a small dataset.

## Open questions
- Do we treat the **atlas choice** as part of the optimization, or fix it and only tune tractography?
- Do we always run `tcksift2`, or allow a “no-SIFT2” mode for speed?
- Which connectome measure(s) will be scored by default (sift-weighted count, mean length, etc.)?

