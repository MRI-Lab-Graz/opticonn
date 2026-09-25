# Methods: Selection and Optimization Strategies

## Selection: repeat-run discriminability

There is no ground truth for "correct" tractography parameters, so OptiConn does not optimize toward one. Instead, the parameter set that is **selected** is the one whose repeated tracking runs of the same subject can be told apart from other subjects' runs (above tracking noise) — **discriminability**, computed from `repeats` tracking runs per subject with varying random seeds. Candidates that fail basic plausibility gates (density, isolated nodes) are excluded before ranking, not penalized within it. See the composite score's status below — it is reported for context but no longer drives selection.

### The criterion saturates — and that is structural

Discriminability was exactly 1.000 for every candidate, in both waves, in two independent cohorts, at 5,000 and 50,000 streamlines, under both DSI Studio tracking methods. Pooling waves to 20 subjects did not change it.

This is *not* because tracking noise is negligible. Measured on study 129 (AAL3, edge-vector correlation, `log1p` of the upper triangle, 50k streamlines, n=10 per wave):

| Connectivity metric | tracking noise | parameter | between subject | noise as share of between-subject |
| --- | --- | --- | --- | --- |
| count | 0.0114 | 0.0511 | 0.2461 | 20% |
| fa | 0.0922 | 0.1910 | 0.4325 | 39% |
| qa | 0.0924 | 0.1929 | 0.4392 | 39% |

Noise is large, and the statistic still pins at its ceiling, because discriminability is **ordinal**: it asks only whether a within-subject distance is smaller than a between-subject one, never by how much. Any candidate that is not broken passes. More subjects do not help.

**How to read this as a user.** Treat discriminability as a pass/fail filter. Ties at 1.0 are the expected outcome and do not mean the candidates are equivalent — the same table shows a parameter change moving a connectome 21% (count) to 44% (fa, qa) as far as the difference between two people. Consult the margin, the variance decomposition and the graph-measure reports rather than reading a winner off a saturated score.

### Streamline count changes what the screen can see

At 5,000 streamlines, 65-76% of the apparent parameter effect was tracking noise; at 50,000, 22-48%. Screening too cheaply therefore *overstates* how much parameters matter, and can make a criterion unusable: the count margin had a cross-wave rank agreement of 0.03 at 5k (pure noise) and 0.87 at 50k. If a criterion looks unstable, check `tracking_noise / parameter` before adding subjects — the remedy is usually more streamlines.

Candidate *ordering*, by contrast, is robust: it transferred across a tenfold streamline change (rho 0.63-0.93), across tracking methods (0.94-0.97) and across two different acquisitions (0.94-0.97). Screening at a low streamline count is therefore defensible, and this is testable on your own data by sweeping `tract_count_range`.

### What the data can and cannot determine

High rank agreement does not imply a trustworthy winner. In both cohorts the two leading candidates were separated by 0.0013 while wave-to-wave variation was 0.0090 — sevenfold larger. They differed in exactly one parameter (track/voxel ratio), agreeing on FA threshold and turning angle. The honest report is that the data determined FA and angle and left track/voxel ratio open, not that one candidate won. Before quoting a winner, check that its lead over the runner-up exceeds the wave-to-wave variation.

### Tie-breaking: nearest-neighbour margin

Ranking is discriminability, then `discriminability_margin`, then repeatability, then fewer tracts. The margin is the mean, over within-subject repeat pairs, of (distance to the nearest other-subject scan) minus (distance between the repeats). Unlike discriminability it does not cap at 1.0, so candidates tied at 1.0 are ordered by how far apart subjects sit relative to tracking noise. A candidate whose margin is undefined sorts after any candidate with a known margin. Like discriminability it rewards subject separation, not correctness, and is reported in `discriminability_margin` in the sweep CSV/JSON.

### Cross-sectional by design

OptiConn uses one scan per subject: the first session in natural order (`ses-2` before `ses-10`), for both `tune-grid` staging and `tune-bayes` sampling. Datasets with repeat DWI almost always acquired it to measure change, often an intervention effect, so a later session is never used to choose parameters, neither as a repeat nor as a benchmark. When `exclude_scans` removes a subject's baseline scan, the subject drops out rather than falling back to a later session. `--subjects` counts subjects (default 10). Configs that still set `data_selection.sessions_per_subject` fail at load with an explanation.

### Graph-measure reliability (ICC)

Discriminability works on whole edge vectors and saturates. The graph measures a study analyses do not necessarily: after a two-wave sweep, `graph_icc.csv` reports, per candidate and per global measure (density, global efficiency and clustering, binary and weighted, small-worldness, Louvain modularity), a one-way ICC(1,1) of subjects against tracking repeats with a 95% confidence interval. At 50k streamlines with n=10 per wave, ICCs were uniformly high (0.95-1.00 across all candidates and measures), so the ICC does not by itself separate plausible candidates at this scale. Below 10 subjects every ICC is flagged as low confidence; below 3 none is reported.

ICC is reported, never ranked: measures can disagree on the better candidate, which measures matter is a study decision, and a setting that flattens individual differences can still score well on some measures. Modularity's ICC includes the Louvain algorithm's own variability.

Reliability is not validity. A setting can reproducibly produce false-positive connections, which distort graph measures more than missed connections do (Zalesky et al., 2016). Treat a high ICC as necessary, not sufficient.

### Parameter fragility: the finding that matters most for your paper

The ICC above asks whether a graph measure survives *re-running the same settings*. The more consequential question is whether it survives *choosing different settings*. Measured on both cohorts (subject ordering, Spearman, AAL3/count):

| Cohort | re-run, same settings | across parameter settings |
| --- | --- | --- |
| 129 (three-shell) | 0.93 | 0.66 |
| 134 (two-shell) | 0.94 | 0.75 |

Every measure, both cohorts, no exceptions: parameter choice disrupts the subject ordering far more than tracking noise does. Roughly a quarter to a third of the ordering a group analysis rests on is contingent on the parameter choice.

Two consequences:

- **A high ICC is not enough.** Binary clustering reached ICC 0.95 while being among the most parameter-sensitive measures. A measure can be perfectly reproducible under identical settings and still re-order your subjects when the settings change.
- **The measures are redundant.** A PCA of the seven global measures gives an effective dimensionality of 1.8 (study 129) and 2.0 (study 134), with PC1 explaining 68-74%. Reporting them as independent findings inflates one result into seven.

Per-measure fragility *verdicts* are not offered. At 79 subjects the per-measure values compress into 0.69-0.81 and their ranking anti-correlates across split halves, which is expected when the battery holds only about two independent quantities. OptiConn reports the aggregate contrast, the per-measure values and the effective dimensionality; it does not label individual measures robust or fragile.

### Using this for an existing analysis

If an analysis has already been run with one parameter set, the useful move is not to redo it with a "better" one — the leading candidates differ by less than the noise between them — but to re-run a subset under an alternative setting and check whether the *conclusions* hold. Reporting that robustness check is stronger than claiming optimal parameters, and it is what the fragility numbers above imply is needed.

## Candidate proposal strategies

The two strategies below decide *which parameter sets get evaluated*; discriminability (above) decides which evaluated candidate wins. `tune-grid`'s two-wave sweep is screened by discriminability directly; `tune-bayes`'s proposals are not automatically re-screened today (see [Workflows](workflows.md)) — its own acquisition function still optimizes the composite score described below.

## 1. Bayesian Optimization (Gaussian Processes)

A candidate-proposal strategy (`tune-bayes`). It treats the connectome quality as a "black box" function $f(x)$ where $x$ is the vector of tracking parameters (FA threshold, turning angle, etc.) and $f(x)$ is the composite quality score.

### Algorithm
We use **Gaussian Process (GP) Regression** to model $f(x)$.
1.  **Prior**: A GP defines a prior distribution over functions.
2.  **Acquisition Function**: We use Expected Improvement (EI) to decide which point $x_{next}$ to evaluate next. EI balances *exploitation* (sampling where the model predicts high quality) and *exploration* (sampling where uncertainty is high).
3.  **Update**: After evaluating $f(x_{next})$ (by running tracking and scoring the network), the GP is updated to produce a posterior distribution.

### Subject Sampling
To avoid overfitting to a single subject's anatomy, we implement **Stochastic Bayesian Optimization**.
-   In standard optimization, $f(x)$ is deterministic.
-   In our case, $f(x, s)$ depends on the subject $s$.
-   By sampling a new random subject $s_i$ at each iteration $i$, the optimizer learns parameters that maximize the *expected* quality $E_s[f(x, s)]$ across the population.

## 2. Cross-Validation Sweep (Grid Search)

This method serves as a rigorous baseline and validation tool.

### Design
-   **Split-Half Validation**: The cohort is split into two "waves" (Wave 1 and Wave 2).
-   **Exhaustive Search**: A grid of parameters is defined (e.g., FA $\in \\{0.1, 0.2\\}$, Angle $\in \\{30, 60\\}$).
-   **Evaluation**: Every combination is run on both waves.
-   **Selection**: We select parameters that:
    1.  Score highly in Wave 1.
    2.  Score highly in Wave 2.
    3.  Show low variance between waves (high stability).

This method is computationally expensive ($O(N^k)$ where $k$ is the number of parameters) but provides a complete landscape of the parameter space.

## Composite quality score (reported, not the selection objective)

Still computed and reported alongside discriminability, and still what `tune-bayes`'s acquisition function optimizes — but no longer what `tune-grid`'s or the MRtrix3 backend's selection ranks by. It directly maximizes small-worldness, modularity/efficiency, which tends to reward similar graphs regardless of real topological differences between settings; treat it as descriptive context, not as a second valid ranking criterion.

$$
\mathrm{Score} =
w_1 \cdot \mathrm{Density}_{\mathrm{score}} +
w_2 \cdot \mathrm{Efficiency}_{\mathrm{score}} +
w_3 \cdot \mathrm{SmallWorld}_{\mathrm{score}} + \cdots
$$

-   **Density**: Penalizes unconnected or fully connected graphs.
-   **Global Efficiency**: Measures integration.
-   **Small-Worldness**: Measures the balance of segregation and integration.
-   **Rich Club**: Measures the connectivity of high-degree nodes.

All metrics are normalized to a [0, 1] scale before combination.
