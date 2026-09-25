---
title: "OptiConn: making tractography parameter dependence visible in structural connectomics"
tags:
  - neuroscience
  - diffusion MRI
  - connectomics
  - graph theory
  - optimization
  - bayesian optimization
  - reproducibility
authors:
  - name: Karl Koschutnig
    orcid: 0000-0001-6234-0498
    affiliation: 1
affiliations:
 - name: MRI-Lab-Graz, University of Graz, Austria
   index: 1
date: 21 September 2026
bibliography: paper.bib
---

## Summary

A structural connectome is not a measurement so much as the result of a long chain of choices — atlas, tracking algorithm, angular and FA thresholds, streamline count, connectivity metric. These are routinely fixed heuristically and reported in a single sentence, if at all. Because there is no ground-truth connectome, no procedure can identify the "correct" choice, and *OptiConn* does not attempt to. Its purpose is the complementary one: to make the dependence of a result on those choices **visible and quantified**, so that a published connectome is understood as one view of the data among several defensible ones.

*OptiConn* runs candidate parameter sets repeatedly over a cohort and reports three things. First, a **reliability screen** (repeat-run discriminability, a nearest-neighbour margin, and run-to-run repeatability, after gates on graph density and isolated nodes) that rejects degenerate candidates. Second, a **variance decomposition** placing the parameter effect on the same scale as tracking noise and between-subject differences. Third, the **reliability of the graph measures a study will actually analyse**, since those — not edge vectors — are what a paper reports.

Applied to two independent cohorts (117-126 subjects each, three-shell and two-shell acquisitions), the screen behaved as designed but told a more useful story than "which candidate wins": every plausible candidate was equally identifiable, the top candidates were separated by less than sampling noise, and yet switching between them displaced the connectome by 21-44% of the distance between two different people and re-ordered subjects on every graph measure far more than re-running the same settings did. The decision-relevant output is therefore not a recommended parameter set — *OptiConn* deliberately does not publish one — but a per-cohort statement of **which parameters the data can and cannot determine, and how much the remaining freedom moves the result**.

*OptiConn* is cross-sectional by design (one scan per subject, since repeat sessions usually carry the effect under study), proposes candidates by grid, random or Latin-hypercube sweep or by Gaussian-process search, and supports **DSI Studio** and **MRtrix3** backends.

## Statement of need

There is no consensus “best” parameterization for diffusion tractography–based connectomics; different atlases and thresholds materially shift graph topology, and no ground truth exists to say which topology is "correct". Existing pipelines (e.g. generic MRtrix or DIPY scripts) provide flexible building blocks but little built-in guidance for principled parameter selection. Exhaustive grid searches are computationally prohibitive (months of runtime), while manual tuning is subjective. *OptiConn* addresses this not by claiming to find an optimal answer, but by making candidate screening explicit and testable: it proposes candidates (via Bayesian or grid/random search), scores them by repeat-run discriminability, and rejects implausible ones — a transparent procedure a reader can inspect and reproduce, rather than a black-box "best" recommendation.

A second need became apparent in developing the tool, and is the more consequential one. A reliability screen answers "which candidate ranks first". A researcher needs to know something prior to that: **whether their conclusions depend on the choice at all.** That question is almost never asked, because nothing routinely answers it.

Measured on a 126-subject cohort (AAL3, 12 candidates spanning FA threshold, turning angle and track/voxel ratio, two independent 10-subject waves, 50,000 streamlines):

- Switching between candidates that all passed every plausibility gate displaced a connectome by **0.21x** (streamline count) to **0.44x** (FA- and QA-weighted) the dissimilarity between two different subjects.
- Subject *ordering* on global graph measures — the quantity a group analysis rests on — survived a re-run of identical settings at Spearman **0.93**, but only **0.66** across parameter settings. Roughly a quarter to a third of the ordering information is contingent on a choice that is rarely reported.
- The seven global graph measures had an **effective dimensionality of 1.8**, so a battery reported as seven findings is closer to two.

These quantities are cohort-specific and cannot be looked up; they must be measured on the data at hand. That is what *OptiConn* is for. It supports **DSI Studio** and **MRtrix3** (via QSIRecon-preprocessed derivatives) as tractography backends.

## State of the field

Foundational graph metrics for brain networks are well established [@Rubinov2010]. Tool ecosystems like MRtrix3, DIPY, and networkx-based wrappers enable tract generation and graph computation but integrate limited automated parameter ranking. Discriminability — the probability that repeated measurements of the same subject are more similar to each other than to another subject's — has been established as a general criterion for selecting analysis pipelines under exactly this kind of no-ground-truth constraint, including in connectomics [@Bridgeford2021]. *OptiConn* applies that principle specifically to tractography parameter selection, with one difference that bears stating: discriminability is conventionally computed over repeat *acquisitions*, whereas *OptiConn*'s repeats are repeat tracking runs of one acquisition. The noise floor it measures is correspondingly narrower — tractography stochasticity rather than the full measurement chain — which makes the screen easier to pass and is why it is positioned here as a rejection filter rather than a fine-grained ranking. Few existing tools implement (1) repeat-run discriminability as a screening criterion for parameter selection, (2) independent validation waves that allow the stability of that criterion to be checked rather than assumed, (3) efficient candidate proposal via Bayesian search [@Snoek2012] feeding the screen, and (4) a report of how far the parameter choice moves the connectome and the downstream graph measures, on the scale of between-subject differences. *OptiConn* targets that gap. The fourth is, to our knowledge, the least represented in existing tooling: it shifts the output from *which candidate to take* to *how much the choice matters, and which parts of it the data can settle*.

## Design and implementation

Core components:

1.  **Bayesian Optimization Engine**: A candidate *proposer* — supporting infrastructure for the discriminability screen in (7), not the selection criterion itself. It uses Gaussian Process regression (via `scikit-optimize`) to model the relationship between tracking parameters (FA, turning angle, step size, etc.) and network quality. It supports **Subject Sampling**, where a different random subject is used for each iteration, ensuring the learned parameters are robust across the population and not overfitted to a single anatomy.
2.  **Parameter Sweep Engine**: The engine that applies the discriminability screen in (7). It evaluates candidates — enumerated by grid, random, or Latin hypercube sampling, or handed over from the Bayesian proposer — with repeated tracking runs per subject, under a two-wave cross-validation design that separates optimization from validation.
3.  **Computation Integrity Validation**: A safety layer against silent failures. Structural checks — missing or partial connectivity matrices, wrong dimensions, NaN/Inf entries, empty matrices — apply to every run and mark the affected iteration failed rather than letting it score. Detection of artificially inflated single-subject scores is currently implemented in the Bayesian proposer only; the sweep engine relies on the plausibility gates and on reporting repeat counts and per-combination diagnostics for post-hoc inspection.
4.  **Graph Metric Acquisition**: For each candidate, global measures are extracted (density, global efficiency [weighted], small‑worldness, clustering/transitivity, path length, assortativity, rich‑club indices).
5.  **Variance decomposition**: A diagnostic report, computed from what a completed sweep already writes, that expresses three sources of variation on one scale — the dissimilarity (1 − *r*) between connectome edge vectors, using the same distance the selection criterion uses. The strata are `tracking_noise` (same scan, same candidate, different random seed), `parameter` (same scan, different candidate) and `between_subject` (different subjects, same candidate). Each is reported as a distribution (n, mean, median, IQR) rather than a point estimate, and strata with fewer than ten pairs are flagged as low-confidence. The headline ratios are `parameter / between_subject` — how far the parameter choice moves a connectome relative to the difference between two people — and `tracking_noise / parameter` — how much of the apparent parameter effect is merely stochastic. This decomposition is deliberately **not** wired into ranking: selecting parameters that minimise parameter-sensitivity would reward degenerate settings that flatten real differences, reintroducing the circularity described in (7) for the composite score.
6.  **Graph-measure reliability**: For each candidate and each global graph measure (density, global efficiency and clustering in binary and weighted form, small-worldness, Louvain modularity), a one-way ICC(1,1) of subjects against tracking repeats, with a 95% confidence interval from the F distribution. Discriminability saturates for plausible candidates because it compares whole edge vectors against a very small tracking-noise floor; a scalar graph measure compresses the connectome, so candidates that tie on discriminability can differ markedly in how reliably they preserve a given measure, and different measures can favour different candidates. ICC is therefore reported per measure and never ranked, and is flagged as low-confidence below ten subjects.
7.  **Selection criterion — discriminability**: Candidates are ranked primarily by repeat-run discriminability: the probability that two repeat connectomes from the same subject are more similar to each other than to a connectome from a different subject, above chance. This is computed across real repeat tracking runs and, where available, multiple subjects (`scripts/reliability.py`, `cross_validation_bootstrap_optimizer.py`, `mrtrix_tune.py`), and is the primary criterion driving selection in the sweep engine (`tune-grid`) on both the DSI Studio and MRtrix3 backends, as well as in the MRtrix3 backend's Bayesian mode. The DSI Studio Bayesian sampler proposes candidates by composite quality score; its proposals are screened by discriminability when handed to the sweep engine. A prior composite score, `quality_score_raw`, is still computed and reported alongside it for context — it aggregates five weighted components (Sparsity 25%, Small-Worldness 25%, Modularity 20%, Global Efficiency 20%, Reliability 10%) — but no longer drives selection: three of its five components (small-worldness, modularity, global efficiency) are maximized directly by the score, a known limitation for any metric used as a selection objective, since it rewards configurations for scoring well on the same quantities being optimized rather than for genuine, independently verified topological differences. Discriminability carries its own documented limitation: because between-subject anatomical variation is typically far larger than the within-scan tracking noise it is measured against, the statistic saturates at its ceiling once candidates are merely plausible, and ranking then falls through to run-to-run repeatability. Users screening a small number of similar candidates, or a cohort with limited between-subject diversity, should expect ties and consult the per-combination diagnostics rather than reading a winner from a saturated score. To give the criterion headroom above that ceiling, ranking proceeds through a **nearest-neighbour margin** (`discriminability_margin`): the mean, over within-subject repeat pairs, of the nearest between-subject distance minus the within-subject distance. Unlike discriminability it is unbounded above, so candidates tied at 1.0 are ordered by how much separation they achieved rather than falling straight through to repeatability. The full ordering is discriminability, then margin, then repeatability, then fewer streamlines; an undefined margin sorts last.
8.  **Cohort handling**: Scans are discovered by subject and session identifiers parsed from file names, with DataLad/git-annex datasets handled explicitly (annex objects are excluded so a scan is not counted twice under two names). Staging is cross-sectional: each subject contributes one scan, its first session, in both the sweep and the Bayesian proposer, and a subject whose first-session scan fails QC is dropped rather than replaced by a later session. A separate screen (`scripts/qc_gate.py`) flags scans whose *preprocessed* DWI is a robust outlier within the cohort — median/MAD on QSIPrep's final-image contrast metric — which, on our cohort, identified the two scans exhibiting implausible tractography while their raw data appeared normal. Flagged scans are reported, not silently dropped; exclusion is an explicit entry in the wave configuration.
9.  **Automation & Performance**: Single high-level commands: `tune-bayes` (smart search), `tune-grid` (grid/random search), `select` (promotion), and `apply` (full-cohort extraction). Parallel execution is supported via `--max-workers`.
10.  **Reproducibility**: Deterministic seeds, config echoing, parameter snapshots, and machine‑readable artifacts (JSON/CSV) ensure every step is traceable.

## Empirical behaviour, and why no parameter recipe is provided

Two independent cohorts were screened with an identical 12-candidate grid: a three-shell acquisition (b = 0/1000/2000/3000, 117 directions, 126 subjects) and a two-shell acquisition (b = 0/1000/3000, 129 directions, 117 subjects), each with two independent subject waves.

**The screen saturates, for structural reasons.** Repeat-run discriminability was exactly 1.000 for every candidate, in every wave, in both cohorts, at every streamline count tested — and this was not because tracking noise was negligible. Noise reached 20-39% of the between-subject distance, and the statistic still pinned at its ceiling, because it asks only whether a within-subject distance is smaller than a between-subject one, never by how much. It is a rejection filter, not a ranking.

**The candidate ordering is stable; the winner is not.** Rank agreement between independent waves was 0.87-0.92, and the ordering transferred across a tenfold change in streamline count (0.63-0.93), across tracking algorithms (0.94-0.97) and across the two acquisitions (0.94-0.97). Yet in both cohorts the top two candidates were separated by 0.0013 against a wave-to-wave variation of 0.0090 — a sevenfold noise excess. They differed in exactly one parameter. The defensible conclusion is that FA threshold and turning angle were determined by the data while track/voxel ratio was not, and *OptiConn* reports it that way rather than naming a winner.

**We therefore do not publish recommended parameters, and would discourage readers from extracting any.** Three reasons. The separation between leading candidates was smaller than the noise on it, so a recommendation would propagate a distinction the data do not support. The criterion measures identifiability, not anatomical validity, so "ranked first" is not "more correct" [@Zalesky2016]. And the two cohorts here share a site, scanner and population, so their agreement is evidence that screening is stable and cheap — not that a parameter set generalises to other scanners, field strengths or clinical populations. What generalises, and what we offer as calibration, are the expected magnitudes above: a parameter effect of roughly a fifth to a half of the between-subject difference, a quarter to a third of graph-measure subject ordering contingent on the choice, and a graph battery with an effective dimensionality near two.

**Limitations.** Every criterion *OptiConn* reports measures reliability, and reliability is not validity: dense tractography can reproducibly generate false-positive connections, which distort graph measures more than missed connections do [@Zalesky2016]. The density gate is the only safeguard for specificity; a high ICC is necessary for a trustworthy graph analysis, not sufficient. Because repeats are tracking re-runs of a single scan, they carry algorithmic but not measurement noise. Attempts to rank individual graph measures by parameter-fragility did not replicate even at 79 subjects, which we attribute to their low effective dimensionality; *OptiConn* therefore reports the aggregate contrast and the per-measure values, but no per-measure verdict.

## Quality control

Validation measures:

-   **Integrity Checks**: Automatic detection of failed extractions or invalid metrics prevents "garbage in, garbage out".
-   **Automated Test Suite**: A pytest suite (`tests/`, 174 tests) runs without a tractography backend installed and covers the reliability statistics on synthetic data with known injected structure, the variance decomposition strata and ratios, file discovery under git-annex layouts, cross-sectional staging, configuration validation, and the CLI surface. A separate script (`scripts/test_integrity_checks.py`) exercises the scoring and failure-detection logic.
-   **Cohort-level data screening**: The QC gate described in (8) flags scans degraded during preprocessing before they enter a sweep, since a single corrupted scan distorts every between-subject comparison it participates in.
-   **Subject Sampling**: The Bayesian optimizer's ability to sample different subjects per iteration prevents overfitting to specific anatomical idiosyncrasies.
-   **Cross-wave Stability**: (Sweep mode) Score divergence between waves flags fragile candidates.
-   **Density Corridor**: Rejects extreme sparsity or saturation early.
-   **Diagnostics**: Comprehensive JSON/CSV logs for every iteration allow post-hoc analysis of the optimization trajectory.

## Reuse potential

The toolkit enables:

-   **Rapid Screening**: Screen candidate parameters for a new dataset in a few hours; rankings proved stable at low streamline counts, so screening need not run at production scale.
-   **Study planning**: The variance decomposition answers, before a full analysis is committed to, whether the parameter choice is negligible or comparable to the effect under study.
-   **Sensitivity reporting**: An existing analysis can be re-examined under an alternative parameter set to test whether its conclusions are parameter-dependent — a robustness check that is currently rare in connectomics papers, and which we consider the tool's most useful application.
-   **Method Harmonization**: Share `selected_candidate.json` files to document, rather than prescribe, the settings a study used.
-   **Benchmarking**: Compare atlases or connectivity metrics under standardized scoring.
-   **Extension**: The modular design allows adding new scorers or optimization strategies.

## Availability

-   Source repository: [MRI-Lab-Graz/opticonn](https://github.com/MRI-Lab-Graz/opticonn)
-   License: MIT
-   Dependencies: Python ≥3.10, DSI Studio or MRtrix3 (external, backend-dependent), scikit-optimize, numpy/pandas/networkx/scipy.
-   **Demo Data**: To facilitate testing and demonstration, we recommend using the open-access diffusion dataset available on OpenNeuro (ds003138) [https://openneuro.org/datasets/ds003138/versions/1.0.1](https://openneuro.org/datasets/ds003138/versions/1.0.1). This dataset is compatible with the BIDS standard and suitable for testing the pipeline's capabilities.

## Acknowledgements

We acknowledge the support of the MRI-Lab-Graz team for testing and feedback.

## References
