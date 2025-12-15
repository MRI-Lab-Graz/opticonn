# Methods: Optimization Strategies

OptiConn employs two distinct strategies to solve the problem of parameter selection in structural connectomics.

## 1. Bayesian Optimization (Gaussian Processes)

This is the primary and recommended method. It treats the connectome quality as a "black box" function $f(x)$ where $x$ is the vector of tracking parameters (FA threshold, turning angle, etc.) and $f(x)$ is the composite quality score.

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

## Scoring Function

The objective function maximizes a composite score derived from graph-theoretic metrics:

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
