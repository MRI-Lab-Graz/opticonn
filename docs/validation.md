# Validation and Quality Control

Ensuring the validity of the generated connectomes is the core mission of OptiConn. We employ several layers of quality control.

## 1. Computation Integrity Validation

Before any scientific scoring occurs, the pipeline checks for computational failures.

-   **File Existence**: Checks if `.connectivity.mat` files were actually generated.
-   **Dimension Check**: Verifies the matrix dimensions match the atlas (e.g., 84x84 for FreeSurferDKT).
-   **NaN/Inf Check**: Scans for invalid numerical values in the connectivity matrix.
-   **Empty Matrix Check**: Flags matrices with zero connections.

If any of these checks fail, the iteration is marked as `failed` and assigned a score of 0.0, preventing it from being selected.

## 2. Graph Metric Validation

We compute a suite of global graph metrics. We validate these metrics against expected biological ranges.

-   **Density Corridor**: We define a "plausible density" range (typically 5% - 30% for weighted structural networks, though this is configurable). Networks outside this range are penalized.
-   **Small-Worldness**: Real brain networks typically exhibit small-world properties ($\sigma > 1$).

## 3. Cross-Validation (Sweep Mode)

In the Sweep mode, we explicitly validate parameter stability.

-   **Wave Consistency**: We compare the scores of Parameter Set $P$ on Wave 1 (Subjects A, B, C) vs Wave 2 (Subjects D, E, F).
-   **Selection**: We prefer parameters where $|Score(P, W1) - Score(P, W2)|$ is minimized, ensuring the parameters are not overfitting to the specific anatomy of the test subjects.

## 4. Subject Sampling (Bayesian Mode)

In Bayesian mode, validation is implicit in the stochastic process. By evaluating parameters on different subjects, "fragile" parameters (those that work only for one specific brain) will yield high variance in the GP model and are less likely to be chosen as the global optimum.

## 5. Output Verification

The `validate_setup.py` script allows users to verify their environment and configuration before starting a long run.

```bash
python scripts/validate_setup.py --config configs/braingraph_default_config.json
```

## 6. Automated Test Suite

OptiConn includes a comprehensive test suite to verify the integrity of its internal logic, particularly the scoring and failure detection mechanisms.

To run the integrity checks:

```bash
python scripts/test_integrity_checks.py
```

This suite validates:
-   Detection of artificial single-subject scores.
-   Handling of partial extraction failures.
-   Metric range validation (e.g., density > 1.0).
-   NaN/Inf detection in connectivity matrices.
