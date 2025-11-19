---
title: "OptiConn (braingraph-pipeline): A Bayesian-optimized framework for structural connectomics"
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
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: MRI-Lab-Graz, University of Graz, Austria
   index: 1
date: 19 November 2025
bibliography: paper.bib
---

## Summary

Structural connectome construction depends on numerous interlocking choices (atlas, tracking parameters, streamline count, connectivity threshold, metric). These are often fixed heuristically, reducing reproducibility and risking biased network structure. *OptiConn* (the optimization layer inside `braingraph-pipeline`) provides a fully automated, data-driven workflow that discovers optimal tractography parameters. Built exclusively as a wrapper for **DSI Studio**, it features two complementary approaches: (i) a highly efficient **Bayesian Optimization** engine that uses Gaussian Processes to find optimal parameters in 20-50 iterations, and (ii) a systematic **Cross-Validation Sweep** (grid/random search) for rigorous baseline comparisons. The framework evaluates candidate configurations via global graph-theoretic measures, applies subject sampling strategies to mitigate overfitting, and produces a ranked shortlist of robust atlas–metric–parameter combinations. A selected configuration is then applied uniformly to the full cohort, yielding study-specific, reproducible structural connectomes.

## Statement of need

There is no consensus “best” parameterization for diffusion tractography–based connectomics; different atlases and thresholds materially shift graph topology. Existing pipelines (e.g. generic MRtrix or DIPY scripts) provide flexible building blocks but little built-in guidance for principled parameter selection. Exhaustive grid searches are computationally prohibitive (months of runtime), while manual tuning is subjective. *OptiConn* solves this by reframing setup as an explicit optimization problem. Its Bayesian optimizer reduces search time from months to hours, making data-driven parameter selection feasible for routine studies. Note that *OptiConn* currently relies entirely on **DSI Studio** for the underlying reconstruction and tracking operations; it does not support other tools (e.g., MRtrix3) at this time.

## State of the field

Foundational graph metrics for brain networks are well established [@Rubinov2010]. Tool ecosystems like MRtrix3, DIPY, and networkx-based wrappers enable tract generation and graph computation but integrate limited automated parameter ranking. Some recent works explore reliability screens or density control, yet few implement (1) intelligent Bayesian parameter discovery [@Snoek2012], (2) independent bootstrap-like validation waves, and (3) composite scoring across normalized global metrics in a single cohesive, reproducible workflow. *OptiConn* targets that gap.

## Design and implementation

Core components:

1.  **Bayesian Optimization Engine**: The primary discovery tool. It uses Gaussian Process regression (via `scikit-optimize`) to model the relationship between tracking parameters (FA, turning angle, step size, etc.) and network quality. It supports **Subject Sampling**, where a different random subject is used for each iteration, ensuring the learned parameters are robust across the population and not overfitted to a single anatomy.
2.  **Parameter Sweep Engine (Baseline)**: A traditional engine for exhaustive grid, random, or Latin hypercube sampling. It uses a two-wave cross-validation design to separate optimization from validation, serving as a rigorous baseline for the Bayesian results.
3.  **Computation Integrity Validation**: A safety layer that validates every optimization run. It detects silent failures (e.g., partial connectivity matrices, artificial 1.0 scores) and flags faulty iterations, ensuring that high scores reflect genuine network quality, not artifacts.
4.  **Graph Metric Acquisition**: For each candidate, global measures are extracted (density, global efficiency [weighted], small‑worldness, clustering/transitivity, path length, assortativity, rich‑club indices).
5.  **Scoring Framework**: Component scores are combined into an absolute‑scale composite in [0,1], producing `quality_score_raw`. Selection uses this absolute score to avoid deceptive relative maxima.
6.  **Automation & Performance**: Single high-level commands: `bayesian` (smart search), `sweep` (grid search), `review` (selection), and `apply` (full-cohort extraction). Parallel execution is supported via `--max-workers`.
7.  **Reproducibility**: Deterministic seeds, config echoing, parameter snapshots, and machine‑readable artifacts (JSON/CSV) ensure every step is traceable.

## Quality control

Validation measures:

-   **Integrity Checks**: Automatic detection of failed extractions or invalid metrics prevents "garbage in, garbage out".
-   **Automated Test Suite**: A dedicated test suite (`scripts/test_integrity_checks.py`) validates the internal logic of the scoring and failure detection systems, ensuring that edge cases (e.g., NaN values, artificial scores) are handled correctly.
-   **Subject Sampling**: The Bayesian optimizer's ability to sample different subjects per iteration prevents overfitting to specific anatomical idiosyncrasies.
-   **Cross-wave Stability**: (Sweep mode) Score divergence between waves flags fragile candidates.
-   **Density Corridor**: Rejects extreme sparsity or saturation early.
-   **Diagnostics**: Comprehensive JSON/CSV logs for every iteration allow post-hoc analysis of the optimization trajectory.

## Reuse potential

The toolkit enables:

-   **Rapid Study Optimization**: Find optimal parameters for a new dataset in <3 hours using Bayesian optimization.
-   **Method Harmonization**: Share `selected_candidate.json` files to standardize methods across centers.
-   **Benchmarking**: Compare atlases or connectivity metrics under standardized scoring.
-   **Extension**: The modular design allows adding new scorers or optimization strategies.

## Availability

-   Source repository: [MRI-Lab-Graz/braingraph-pipeline](https://github.com/MRI-Lab-Graz/braingraph-pipeline)
-   License: MIT
-   Dependencies: Python ≥3.10, DSI Studio (external), scikit-optimize, numpy/pandas/networkx/scipy.
-   **Demo Data**: To facilitate testing and demonstration, we recommend using the open-access diffusion dataset available on OpenNeuro (ds003138) [https://openneuro.org/datasets/ds003138/versions/1.0.1](https://openneuro.org/datasets/ds003138/versions/1.0.1). This dataset is compatible with the BIDS standard and suitable for testing the pipeline's capabilities.

## Acknowledgements

We acknowledge the support of the MRI-Lab-Graz team for testing and feedback.

## References
