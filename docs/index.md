# OptiConn Docs

![OptiConn logo](img/opticonn_logo.png)

White matter tractography lacks a gold standard for parameter settings, and most publications offer little rationale for their choices—parameters are often selected arbitrarily or by convention. This becomes critical when deriving structural connectomes for graph-theoretic analyses, where parameter decisions directly influence network topology and derived measures. OptiConn addresses this gap by automating tractography parameter discovery through Bayesian optimization, validating selections via cross-validation bootstrap, and applying optimal parameters to produce analysis-ready connectivity outputs with a principled, data-driven foundation.

- For setup, see [Installation](installation.md).
- For day-to-day runs, see [Workflows](workflows.md) and [Demos](demos.md).
- For configuration details, see [Configuration](configuration.md) and [Validation Notes](validation.md).
- For background, see [User Guide](user_guide.md) and [Methods](methods.md).

## Quick links

- Bayesian + Apply demo: `python scripts/opticonn_demo.py --step all`
- Cross-validation demo (seeds from Bayes): `python scripts/opticonn_cv_demo.py --workspace demo_workspace_cv`
- DSI Studio download: https://github.com/frankyeh/DSI-Studio/releases

---

**Affiliation**

- MRI-Lab Graz
- Contact: karl.koschutnig@uni-graz.at
