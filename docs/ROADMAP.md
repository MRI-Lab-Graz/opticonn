# OptiConn Roadmap

Updated: 2026-09-14 (replaces the 2026-05-28 audit)

## 1. Aim

Choose tractography settings from the data instead of by convention. There is
no gold standard, so OptiConn does not claim to find *the* optimal setting. It
**screens** settings with explicit, testable criteria (reliability, QA
confounds, plausibility), recommends a defensible one, and **reports how much
the choice matters** for the dataset (sensitivity). Workflow: sweep tracking
parameters on a subset of subjects, select as in section 3.4, then apply the
recommendation to all subjects.

**Publication targets: JOSS, then Aperture Neuro.** The earlier attempt
stalled on the DSI Studio dependency. It is a separately distributed binary
that reviewers and CI cannot install from a package manager, which works
against JOSS's open, installable, tested-software expectations. Consequence
for this roadmap: **MRtrix3 (open source, conda-installable) becomes the
primary backend**, the one used for examples, CI and the JOSS paper. DSI
Studio stays as an optional backend. JOSS reviews the software itself
(installable, documented, tested); once it's out, the fuller validation
study (section 3.3–3.4: held-out subjects, QA confounds, sensitivity
report, on a real dataset) is submitted separately to Aperture Neuro
(OHBM's open-access journal for research objects including software and
pipelines), which reviews the science and gives that study room a short
software paper doesn't. See section 4.1 (JOSS) and 4.2 (Aperture Neuro).

## 2. Where the code stands (verified 2026-09-14)

The plumbing works: sweep → per-combination extraction → selection → Phase 2
apply. The part that defines "best" does not.

| # | Finding | Evidence |
|---|---------|----------|
| F1 | **The score cannot separate settings.** `metric_optimizer` min-max normalises density/efficiency *within one combination's rows*, so the mean is ~0.5 whatever the parameters. The ranking is then decided by a `0.05 × density` term, i.e. "densest graph wins", which rewards false-positive edges. | `results/sweep_run_20251005_115304_EC98C4`: density 0.43–0.50 (fa 0.05) vs 0.07–0.14 (fa 0.20), quality score 0.495 vs 0.487. |
| F2 | **The swept `tract_count` never reaches DSI Studio.** Sweep configs write `tract_count`; the extractor reads `track_count` (default 100 000). | Combos with 100k and 2.5M tracts produce identical measures to 7 decimals; the 2.5M output folder is named `tracks_100k_…`. |
| F3 | **No reliability measure exists.** No bootstrap, no repeats, no retest. `top3_candidates.json` holds one entry with hard-coded `average_score: 1.0`, so `--candidate N` is a no-op. | `cross_validation_bootstrap_optimizer.py` main(); `results/optimize/optimization_results/top3_candidates.json`. |
| F4 | Phase 2 `--step all` calls `metric_optimizer -i … -o …`, but that script only accepts positional arguments. | `run_pipeline.run_step02` vs `metric_optimizer.main`. |
| F5 | Environment rot: not a git repository; `.venv` was built at `/Users/karl/work/github/opticonn` (pip shebang points there) on Python 3.9 (EOL); `activate.sh` targets a removed sibling repo. | `.venv/pyvenv.cfg`, `.venv/bin/pip`, `activate.sh`. |
| F6 | CLI: `--dsi-studio` is silently dropped whenever the `--config` path does not exist (all flag handling sat inside an `if config exists` branch, while `.opticonn_config` supplied a path at import); the global `--dry-run` is overwritten by each subcommand's own `--dry-run` default; validation fails unless `VIRTUAL_ENV` is exported. | `opticonn.py` main(). |

## 3. Direction

### 3.1 Scoring: reliability, not "quality"

For every (parameters, atlas, metric) candidate:

1. **Repeats.** Track each Phase 1 subject `repeats` times with different
   random seeds (DSI Studio `--random_seed`, MRtrix3 `MRTRIX_RNG_SEED`).
2. **Discriminability** (objective). Probability that a repeat of the same
   subject is closer (1 − Pearson r of log-compressed edge weights) than a
   repeat of another subject. 0.5 = chance, 1.0 = subjects always
   identifiable above tracking noise.
3. **Plausibility gates** (constraints, not rewards). Reject candidates whose
   mean density lies outside `density_range` or whose isolated-node fraction
   exceeds `max_isolated_fraction`. Both are calibration knobs in the sweep
   config.
4. **Tie-breaks.** Repeatability (mean within-subject r), then fewer tracts.
5. **Rank stability.** Leave-one-subject-out: how often each of the top 10
   stays top-1.

Known ceiling: seed repeats only measure tracking noise. With few subjects,
discriminability often saturates at 1.0 and repeatability decides. Scan–rescan
sessions (e.g. HCP retest) are the stronger test and are listed under later
work.

### 3.2 Backends

- **DSI Studio** (existing, optional after Phase 4): `.fz` / `.fib.gz` inputs.
- **MRtrix3** (new, primary for JOSS): preprocessed per-subject folders
  (`wmfod.mif`, `<atlas>.mif`, optional `5tt.mif`) → `tckgen` +
  `tck2connectome` → the same `.connectivity.mat` layout, so scoring and
  Phase 2 are shared. Preprocessing (response estimation, CSD, registration,
  parcellation) stays outside OptiConn.

### 3.3 Validation and confounds (no gold standard, so guard the criterion)

Reliability is not validity: a setting can reproduce the same wrong
connectome every time, and discriminability rewards anything that separates
subjects, including head motion, image quality or head size. Three guards:

1. **Held-out evaluation (cross-validation).** Select settings on one subset
   of subjects and report the score on subjects not used for selection
   (nested k-fold or leave-one-subject-out for small samples). The
   leave-one-subject-out *top-1 frequency* from 3.1 is a stability check, not
   this.
2. **QA confounds.** Accept a per-subject QA table supplied by the user
   (e.g. mean framewise displacement / eddy motion, DSI Studio
   neighbouring-DWI correlation, outlier slices); it is backend-agnostic.
   Use it to (a) exclude subjects failing QA thresholds before selection,
   (b) measure each candidate's confound sensitivity, i.e. how strongly
   between-subject connectome distance follows between-subject QA
   difference, and reject or penalise sensitive settings, and (c) report QA
   alongside results.
3. **Cost curve.** Report score against streamline count and prefer the point
   of diminishing returns, because repeatability rises monotonically with
   streamlines.
4. **Graph metrics as constraints, never as objectives.** Maximising
   small-worldness, efficiency or clustering mostly rewards density changes
   (the failure mode of the old score, F1). Nearly any network with some
   clustering is "small-world", and optimising a metric that later analyses
   test would make those analyses circular. Use graph metrics as (a)
   plausibility gates (single connected component, small-world index > 1,
   density band), (b) reliability targets: ICC across repeats of the
   measures a study will analyse (e.g. global efficiency, modularity), and
   (c) comparisons at matched density (proportional thresholding) so density
   is not mistaken for topology.

### 3.4 Selection rule (what the user gets)

No weighted sum: weights would be as arbitrary as the settings they replace.
Selection is staged, and every stage appears in the report.

1. **QA screen:** subjects failing user QA thresholds are excluded and listed.
2. **Gates (pass/fail, reason recorded):** plausibility (connected, no
   isolated regions, density band) and confound sensitivity below a
   threshold.
3. **Primary criterion on held-out subjects:** discriminability by default,
   or ICC across repeats of a user-declared target measure (e.g. global
   efficiency).
4. **Equivalence set:** candidates whose held-out score is statistically
   indistinguishable from the best (e.g. overlapping bootstrap intervals).
   Differences that cannot be measured are not claimed.
5. **Recommendation:** within the equivalence set, fewest streamlines, then
   highest repeatability. Until the equivalence set exists (Phase 5), the
   interim order of section 3.1 applies (discriminability, repeatability,
   fewer streamlines), because choosing the fewest streamlines without an
   equivalence set can pick a noticeably noisier setting.
6. **Sensitivity report:** spread of connectomes and of the target measure
   across the equivalence set. Small spread: use the recommendation. Large
   spread: run the downstream analysis on 2–3 representatives and report all
   (multiverse).

Deliverables: recommended configuration (Phase 2 input) with justification,
equivalence set, rejected candidates with reasons, sensitivity numbers.
Status: stages 2 (plausibility part) and 3 (on sweep subjects, not yet held
out) are built by the reliability-optimizer plan; stage 5 is interim-only in
this branch (no equivalence set yet, so it falls back to the section 3.1
order); stages 1, 4, 6, the confound gate and held-out scoring belong to
Phase 5.

## 4. Plan

| Phase | Scope | Plan | Status |
|-------|-------|------|--------|
| 0 | Git baseline, fresh venv, F2 fix | reliability-optimizer plan, Tasks 1–2 | `[x]` |
| 1 | Reliability scoring (F1, F3), honest top-N | same plan, Tasks 3–5 | `[x]` |
| 2 | Phase 2 = extraction + network-measure aggregation only (F4); remove the old quality-score scripts | same plan, Task 6 | `[x]` |
| 3 | CLI + docs cleanup (F5, F6) | same plan, Task 7 | `[x]` |
| 4 | MRtrix3 backend | mrtrix3-backend plan | `[ ]` |
| 5 | Validation: held-out subjects, QA confounds, cost curve (section 3.3; plan to be written) | — | `[ ]` |
| 6 | JOSS readiness (plan to be written after Phase 4 works on real MRtrix3 data) | see 4.1 | `[ ]` |
| 7 | Aperture Neuro submission: the validation study from Phase 5 on real data | see 4.2 | `[ ]` |

Implementation plans for completed and in-progress phases are tracked
outside this repository.

### 4.1 Phase 6: JOSS readiness (scope, not yet planned)

Check each item against the current JOSS author guidelines before starting;
they are revised periodically.

- OSI-approved `LICENSE` (MRtrix3 is MPL-2.0; MIT/BSD/MPL are all compatible
  for a wrapper that shells out).
- Installable package: `pyproject.toml` with an `opticonn` console script;
  MRtrix3 via conda-forge documented as the external dependency.
- Small **open** example dataset for MRtrix3 (a few subjects' FODs and
  parcellations, or a script that fetches and prepares them). CI and
  reviewers need it; the current `.fz` samples are DSI Studio–only.
- CI (GitHub Actions): `pytest` plus a tiny end-to-end MRtrix3 sweep on that
  dataset.
- Docs: statement of need, install, tutorial, config reference, and an
  explanation of the scoring (section 3.1).
- Community files: `CONTRIBUTING.md`, issue templates, and how to get support.
- `paper.md` + `paper.bib`: summary, statement of need, **state of the field**
  (discriminability-based pipeline selection, Bridgeford et al. 2021; existing
  tractography parameter studies), and acknowledgements.
- Public repository with a tagged release and archive DOI (Zenodo).

### 4.2 Phase 7: Aperture Neuro submission (scope, not yet planned)

After Phase 6. Submitted as a research object to Aperture Neuro (the OHBM's
open-access, non-traditional-research-object journal), not JOSS: the venue
for the validation study itself, reviewed by neuroimaging researchers rather
than for software installability.

- Phase 5's held-out evaluation, QA-confound sensitivity, cost curve and
  equivalence-set results, run on a real dataset (ideally with scan-rescan
  sessions).
- Reports how much connectomes and downstream graph measures change across
  equally defensible settings (the sensitivity finding is itself a result).
- State of the field: position as open, reusable screening software applying
  reliability-based selection to tractography parameters (Bridgeford et al.
  2021 is the closest prior art for the discriminability approach), not as a
  new statistic.
- Check Aperture Neuro's current author guidelines (article-processing
  charge and waiver policy, CC-BY 4.0 licensing, submission format) before
  starting.

### Later, when there is a reason

- **Scan–rescan sessions as repeats.** Add when retest data is at hand
  (strongest reliability evidence, and a strong validation figure for the
  paper).
- **More MRtrix3 edge weights** (SIFT2 `-tck_weights_in`, mean FA via
  `tcksample`). Add when count-based selection is validated.
- **Preprocessing wrapper for MRtrix3.** Add only if users keep failing at
  input preparation.
- **Region names in matrices** (instead of `region_001`). Add when Phase 2
  outputs feed statistics directly.

## 5. The 2026-05-28 audit, re-triaged

| Old item | Verdict |
|----------|---------|
| 1.1 activate.sh | Fix by deleting the script; Phase 3 |
| 1.2 fake top-3 | Phase 1 |
| 1.3 inverted DSI discovery | Real; fixed by deleting discovery and applying a simple precedence; Phase 3 |
| 1.4 missing configs | Phase 3 (point references at existing configs) |
| 1.5 `auto` crashes on list config | **Not reproducible from the code**: `"key" in list` does not raise |
| 2.1 README corrupted | Phase 3 (README rewritten) |
| 2.2 quick_start paths | Phase 3 (also regenerated by `install.sh`, fix both) |
| 2.3 artifacts in repo | Phase 0 `.gitignore` |
| 2.4 duplicated discovery | Deleted in Phase 3 |
| 2.5 / 2.6 scoring | Replaced in Phase 1 (was understated: see F1) |
| 2.7 `--candidate` no-op | Fixed by Phase 1 |
| 2.8 stale `sweep_parameters` / shared nested dict | Phase 1 (Task 4) |
| 2.9 fake LHS | Delete the option; grid/random suffice for 3–5 parameters |
| 2.10 thread oversubscription | Dropped; `max_parallel` defaults to 1 |
| 2.11 global `random.seed` | Phase 1 (`random.Random(seed)`) |
| 2.12 README paths | Phase 3 |
| 2.13 subprocess output bypasses filters | Dropped (cosmetic) |
| 2.14 `.tt.gz` deleted | Kept as is; tract files are large and not used downstream |
| 2.15 / 2.16 connectogram CSV, `quick_analysis.py` stub | Later (region names) |
| 2.17 no tests | Each plan task leaves its tests |
| 3.1–3.13, section 4 | Dropped unless touched by a task above; revisit before publishing |

## 6. Open questions

- **Novelty / state of the field (JOSS requires this section).**
  Discriminability-based pipeline selection is published (Bridgeford et al.,
  2021, *PLoS Comput Biol*). The paper should position OptiConn as open,
  reusable software that applies reliability-based selection to tractography
  parameters, not as a new statistic.
- **Keep DSI Studio at all?** Keeping it as an optional backend costs little
  once scoring is shared. Dropping it simplifies the paper, CI and docs.
  Decide before Phase 5.
- **Retest data.** Is scan–rescan data available for the target datasets?
- **Example data.** `examples/data/fib_samples` (80 MB) is git-ignored in
  Phase 0. Decide on Git LFS or a download script if others need it.
- **`utils/config_helper.py`, `utils/results_viewer.py`.** Unreferenced by
  code. Keep or delete after checking whether anyone uses them.
