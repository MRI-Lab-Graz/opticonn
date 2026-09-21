# Methods: Selection and Optimization Strategies

## Selection: repeat-run discriminability

There is no ground truth for "correct" tractography parameters, so OptiConn does not optimize toward one. Instead, the parameter set that is **selected** is the one whose repeated tracking runs of the same subject can be told apart from other subjects' runs (above tracking noise) — **discriminability**, computed from `repeats` tracking runs per subject with varying random seeds. Candidates that fail basic plausibility gates (density, isolated nodes) are excluded before ranking, not penalized within it. See the composite score's status below — it is reported for context but no longer drives selection.

### Known limitation: the criterion saturates

Because repeats are re-runs of one scan, the noise floor discriminability is measured against is tractography stochasticity alone — which is very small at converged streamline counts. Between-subject anatomy differs far more. The consequence is that every merely-plausible candidate passes, and discriminability pins at 1.0.

Measured on a 150-subject cohort (AAL3, edge-vector correlation, `log1p` of the upper triangle):

| Comparison | r | dissimilarity (1-r) |
| --- | --- | --- |
| Same scan, same parameters, different random seed — the noise floor | 0.997 | 0.003 |
| Same scan, different parameters (fa 0->0.1, angle->45) | 0.864 | 0.136 |
| Different subjects, same parameters | 0.684 (median 0.703) | 0.316 |

The noise floor is roughly 2% of the parameter effect and 1% of the between-subject difference. A sweep of four candidates bracketing a production setting returned discriminability 1.0 for all four; only repeatability separated them, across a span of 0.0032 (0.9959 to 0.9991 on edge-count connectivity, 2 repeats each).

**How to read this as a user.** Discriminability is a rejection filter, not a fine-grained ranking. Ties at 1.0 are the expected outcome for a set of reasonable candidates, not a sign that the candidates are equivalent — the same table shows a modest parameter change moving the connectome about 0.43x as far as the difference between two people. When candidates tie, consult the per-combination diagnostics (density, repeatability, graph measures) rather than reading a winner off the saturated score, and widen the candidate range if you need the screen to discriminate.

### Tie-breaking: nearest-neighbour margin

Ranking is discriminability, then `discriminability_margin`, then repeatability, then fewer tracts. The margin is the mean, over within-subject repeat pairs, of (distance to the nearest other-subject scan) minus (distance between the repeats). Unlike discriminability it does not cap at 1.0, so candidates tied at 1.0 are ordered by how far apart subjects sit relative to tracking noise. A candidate whose margin is undefined sorts after any candidate with a known margin. Like discriminability it rewards subject separation, not correctness, and is reported in `discriminability_margin` in the sweep CSV/JSON.

### Cross-sectional by design

OptiConn uses one scan per subject: the first session in natural order (`ses-2` before `ses-10`), for both `tune-grid` staging and `tune-bayes` sampling. Datasets with repeat DWI almost always acquired it to measure change, often an intervention effect, so a later session is never used to choose parameters, neither as a repeat nor as a benchmark. When `exclude_scans` removes a subject's baseline scan, the subject drops out rather than falling back to a later session. `--subjects` counts subjects (default 10). Configs that still set `data_selection.sessions_per_subject` fail at load with an explanation.

### Graph-measure reliability (ICC)

Discriminability works on whole edge vectors and saturates. The graph measures a study analyses do not necessarily: after a two-wave sweep, `graph_icc.csv` reports, per candidate and per global measure (density, global efficiency and clustering, binary and weighted, small-worldness, Louvain modularity), a one-way ICC(1,1) of subjects against tracking repeats with a 95% confidence interval. On a pre-release exploratory study-129 AAL3 sweep (n = 5–6 scans, 2 repeats; these figures will be replaced by a fresh cross-sectional sweep), two candidates tied on discriminability (1.000), margin (0.163 vs 0.158) and repeatability (0.951 vs 0.950), yet their binary global-efficiency ICC was 0.62 vs 0.92 while binary clustering favoured the other candidate (0.84 vs 0.71). The 95% intervals overlap, so this is suggestive only. Below 10 subjects every ICC is flagged as low confidence; below 3 none is reported.

ICC is reported, never ranked: measures can disagree on the better candidate, which measures matter is a study decision, and a setting that flattens individual differences can still score well on some measures. Modularity's ICC includes the Louvain algorithm's own variability.

Reliability is not validity. A setting can reproducibly produce false-positive connections, which distort graph measures more than missed connections do (Zalesky et al., 2016). Treat a high ICC as necessary, not sufficient.

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
