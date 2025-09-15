---
title: "OptiConn (braingraph-pipeline): A data-driven, two-wave optimization framework for structural connectomics"
authors:
  - name: "Karl Koschutnig"
    affiliation: "MRI-Lab-Graz"
    orcid: ""
    email: ""
    github: "MRI-Lab-Graz"
date: 2025-09-13
tags:
  - neuroscience
  - diffusion MRI
  - connectomics
  - graph theory
  - optimization
  - reproducibility
---

# Summary
Structural connectome construction depends on numerous interlocking choices (atlas, tracking parameters, streamline count, connectivity threshold, metric). These are often fixed heuristically, reducing reproducibility and risking biased network structure. *OptiConn* (the optimization layer inside `braingraph-pipeline`) provides a fully automated, data-driven workflow that (i) performs a systematic parameter sweep (grid, random, or Latin hypercube sampling), (ii) evaluates candidate configurations via global graph-theoretic measures, (iii) applies a two-wave cross-validation procedure on independently sampled subject subsets to mitigate overfitting, and (iv) produces a ranked shortlist (Top‑3) of robust atlas–metric–parameter combinations. A selected configuration is then applied uniformly to the full cohort, yielding study-specific, reproducible structural connectomes.

# Statement of need
There is no consensus “best” parameterization for diffusion tractography–based connectomics; different atlases and thresholds materially shift graph topology. Existing pipelines (e.g. generic MRtrix or DIPY scripts) provide flexible building blocks but little built-in guidance for principled parameter selection. *OptiConn* reframes methodological setup as an explicit optimization problem isolated from downstream hypothesis testing, reducing circularity and encouraging reproducible, study-tailored configurations.

# State of the field
Foundational graph metrics for brain networks are well established [@Rubinov2010]. Tool ecosystems (MRtrix3, DIPY, networkx-based wrappers) enable tract generation and graph computation, but integrate limited automated parameter ranking. Some recent works explore reliability screens or density control, yet few implement (1) multi-parameter sweeps, (2) independent bootstrap-like validation waves, and (3) composite scoring across normalized global metrics in a single cohesive, reproducible workflow. *OptiConn* targets that gap.

# Design and implementation
Core components:
1. **Parameter sweep engine**: User-declared ranges (list or `start:step:end` syntax) for tractography (e.g. FA/Otsu thresholds, min/max length, streamline count, turning angle, smoothing) and connectivity (atlas set, metric: count/qa/fa, connectivity threshold, type). Sampling modes: exhaustive grid, random, Latin hypercube; optional “quick sweep” for coarse pruning.
2. **Two-wave cross-validation**: Two independent random subject subsets (Wave 1, Wave 2) run the identical sweep; scores are aggregated (mean or robust statistic) to dampen overfitting to a single sample.
3. **Graph metric acquisition**: For each candidate (atlas × metric × parameter set) global measures are extracted (density, global efficiency, small‑worldness, clustering/transitivity, path length, assortativity, rich‑club indices; optionally diameter, radius, modularity).
4. **Scoring framework**: Raw measures are robustly normalized (quantile clipping + z or min–max) and mapped to thematic sub-scores (efficiency, integration–segregation balance, small‑world structure, sparsity appropriateness, robustness). A weighted composite yields a quality score in [0,1]. Weights are configurable.
5. **Ranking & selection**: Full ranked table (`all_candidates_ranked.json`) plus stability metadata; Top‑3 shortlist (`top3_candidates.json`) supports interactive or scripted choice. Selected candidate serialized (`extraction_from_selection.json`).
6. **Automation**: Single high-level commands: `optimize` (sweep + waves + aggregation) and `apply`/`analyze` (full-cohort extraction with chosen configuration). Manifests (available vs. selected subjects), config hashes, and logs are persisted.
7. **Reproducibility features**: Deterministic seeds, machine‑readable artifacts, version stamping, optional quiet mode for batch use.

# Quality control
Validation measures:
- **Cross-wave stability**: Score divergence between waves flags fragile candidates.
- **Density corridor**: Reject extreme sparsity or saturation early.
- **Composite normalization diagnostics** (planned): Outlier clipping summaries.
- **Determinism**: Re-running with identical seeds reproduces Top‑3 ordering (empirically verified in internal tests).
- **Transparent selection**: Human choice occurs only after unbiased ranking; no group labels used during optimization.

# Reuse potential
The toolkit enables:
- Rapid study-specific optimization for new cohorts.
- Method harmonization across centers by sharing final JSON configs.
- Benchmarking of atlases or connectivity metrics under standardized scoring.
- Extension via plugin scorers (planned) to incorporate reliability or modularity stability.
Outputs (ranked candidates, final matrices, metrics) integrate with downstream statistical frameworks or machine learning pipelines.

# Limitations
- **Compute cost**: Large grids can be expensive; mitigated by sampling modes and quick sweeps.
- **Global metric focus**: Edge-level or community stability not yet part of core score (planned).
- **Two-wave design**: Improves robustness but does not fully eliminate overfitting in very small datasets.
- **No clinical outcome optimization**: Intentional separation to avoid leakage; downstream modeling remains user responsibility.

# Roadmap (selected)
Planned enhancements: edge reliability bootstrapping, Pareto multi-objective front, Bayesian adaptive re‑sampling, provenance hashing (environment + binary versions), plugin-based scorers, normative reference integration, optional modularity stability metrics.

# Availability
- Source repository: (add GitHub URL)
- License: (add OSI-approved license, e.g. MIT/BSD)
- Installation: `pip install opticonn` (post-publication) or from source.
- Dependencies: Python ≥3.10, DSI Studio (external), numpy/pandas/networkx/scipy.

# Acknowledgements
Developed at MRI-Lab-Graz. Community feedback and early testing by lab members acknowledged.

# References
- Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: Uses and interpretations. *NeuroImage*, 52(3), 1059–1069. https://doi.org/10.1016/j.neuroimage.2009.10.003

# Statement of authorship
All listed authors meet authorship criteria; the submitting author accepts responsibility for the manuscript.

# Conflicts of interest
None declared.

# Funding
(Add funding sources or state “No external funding.”)

# Data availability
Demonstration scripts operate on user-provided diffusion reconstructions; no bundled subject-level data.
