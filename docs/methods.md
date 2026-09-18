# Methods: Selection and Optimization Strategies

## Selection: repeat-run discriminability

There is no ground truth for "correct" tractography parameters, so OptiConn does not optimize toward one. Instead, the parameter set that is **selected** is the one whose repeated tracking runs of the same subject can be told apart from other subjects' runs (above tracking noise) — **discriminability**, computed from `repeats` tracking runs per subject with varying random seeds. Candidates that fail basic plausibility gates (density, isolated nodes) are excluded before ranking, not penalized within it. See the composite score's status below — it is reported for context but no longer drives selection.

### Known limitation: the criterion saturates

Because repeats are re-runs of one scan, the noise floor discriminability is measured against is tractography stochasticity alone — which is very small at converged streamline counts. Between-subject anatomy differs far more. The consequence is that every merely-plausible candidate passes, and discriminability pins at 1.0.

Measured on a 150-subject longitudinal cohort (AAL3, edge-vector correlation, `log1p` of the upper triangle):

| Comparison | r | dissimilarity (1-r) |
| --- | --- | --- |
| Same scan, same parameters, different random seed — the noise floor | 0.997 | 0.003 |
| Same subject, different session, same parameters | 0.889 (median 0.900) | 0.111 |
| Same scan, different parameters (fa 0->0.1, angle->45) | 0.864 | 0.136 |
| Different subjects, same parameters | 0.684 (median 0.703) | 0.316 |

The noise floor is roughly 2.5% of the between-session effect and 2% of the parameter effect. A sweep of four candidates bracketing a production setting returned discriminability 1.0 for all four; only repeatability separated them, across a span of 0.0032 (0.9959 to 0.9991 on edge-count connectivity, 2 repeats each).

**How to read this as a user.** Discriminability is a rejection filter, not a fine-grained ranking. Ties at 1.0 are the expected outcome for a set of reasonable candidates, not a sign that the candidates are equivalent — the same table shows a modest parameter change moving the connectome slightly *more* than a real between-session change does. When candidates tie, consult the per-combination diagnostics (density, repeatability, graph measures) rather than reading a winner off the saturated score, and widen the candidate range if you need the screen to discriminate.

### Session-aware wave staging

By default `tune-grid` stages each subject's sessions together: `--sessions-per-subject` (default 2) makes `--subjects` count subjects rather than scans, so a wave stages up to subjects x sessions scans and costs proportionally more compute. If no subject has enough sessions, staging falls back to the legacy scan-level sampling; `0` or `1` selects it explicitly.

Caveat: discriminability's unit is the scan, so a subject's second session is compared against the first as a "different subject". This makes the test harder than before but is not a test-retest reliability estimate.

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
