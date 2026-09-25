# A multiverse analysis of structural connectome construction

**Working draft — target: Aperture Neuro (OHBM)**
Status: framing + measured results from cohorts 1–2; cohort 3 and the OpenNeuro extension pending.

> Drafting notes are marked `[NOTE]` and must not survive to submission.
> `[VERIFY]` marks citations stated from memory that must be checked against the source.

---

## Title (candidates)

1. *What your data cannot tell you: a multiverse analysis of structural connectome construction*
2. *One dataset, many connectomes: quantifying analytic flexibility in diffusion tractography*
3. *Parameter dependence in structural connectomics, and how to measure it on your own data*

`[NOTE]` Preference for (1): it states the contribution (what is *not* determined) rather than promising a recipe.

## Abstract (draft)

Structural connectomes are not measured; they are constructed through a chain of analytic choices — tracking algorithm, angular and anisotropy thresholds, streamline count, parcellation, edge weighting — that are typically fixed once, reported briefly, and never varied. Because no ground-truth connectome exists, the field has no way to declare any of these choices correct, and consequently little practice of reporting how much a published result depends on them. We treat the choice space as a **multiverse** and ask, for a fixed dataset and analysis, how far the results move across defensible specifications.

Screening N candidate specifications across M independent cohorts spanning different acquisition schemes, we find: (i) conventional reliability screening **saturates** — every non-degenerate specification is equally able to identify individuals, so reliability cannot select among them; (ii) the leading specifications are separated by **less than the sampling noise on the estimate**, so the data determine some parameters and leave others open; (iii) nevertheless, moving within the multiverse displaces a connectome by 21–44% of the distance between two different people, and re-orders subjects on every global graph measure substantially more than re-running the identical specification does; (iv) the standard global-graph battery carries an **effective dimensionality near two**, so multi-measure reports overstate the number of independent findings.

We argue that the appropriate output of parameter exploration is not a recommended specification — we deliberately publish none — but a per-dataset statement of which choices the data can determine and how much the residual freedom moves the result. We release OptiConn, an open-source implementation that produces this report for an arbitrary cohort.

---

## 1. Introduction

### 1.1 Analytic flexibility as a measurement problem

A published neuroimaging result is conditional on a long chain of analytic decisions, most of which are defensible and few of which are reported in enough detail to reproduce. The NARPS study made the consequences concrete for task fMRI: independent teams analysing a single dataset reached materially different conclusions on the same hypotheses (Botvinik-Nezer et al., 2020) `[VERIFY]`. The statistical framing predates it — *multiverse analysis* (Steegen et al., 2016) `[VERIFY]` proposes reporting the distribution of results across all reasonable specifications rather than a single path through them.

Diffusion tractography is an unusually severe case, for three reasons:

1. **The choice space is large and continuous.** Anisotropy and angular thresholds, step size, streamline count, seeding strategy, algorithm family, parcellation and edge weighting each admit a defensible range.
2. **There is no ground truth.** Unlike a statistical model that can be checked against held-out data, a connectome has no reference against which a specification can be declared correct. Tractography is known to produce both false positives and false negatives at rates that depend on these choices (Maier-Hein et al., 2017; Zalesky et al., 2016) `[VERIFY]`.
3. **The output feeds a second analysis.** Graph-theoretic measures are computed *on top of* the constructed connectome, so specification-induced variation propagates into the quantities actually reported.

### 1.2 Why reliability screening is not enough

`[NOTE] This subsection is the pivot: it motivates why we do not simply recommend the winner.`

A natural response is to select a specification by reliability — prefer settings under which repeated measurements of the same individual are most distinguishable from other individuals. Discriminability formalises this and has been applied to pipeline selection (Bridgeford et al., 2021) `[VERIFY]`. We show below that under repeated tracking of the same acquisition this criterion **saturates**: it is an ordinal statistic, asking only whether a within-subject distance is smaller than a between-subject distance and never by how much, so every non-degenerate specification attains its ceiling. It functions as a rejection filter, not a ranking.

This matters beyond a technical detail. If reliability cannot order defensible specifications, then a study that selects one and reports it as optimal is reporting a choice the data did not make.

### 1.3 Contribution

We reframe the question from *which specification is best* to *what does this dataset determine, and how much does the remainder matter*. Specifically we report, per cohort: the saturation behaviour of the reliability screen; the separation between leading specifications relative to its sampling noise; the displacement of the connectome across the multiverse relative to between-subject distance; the disruption of subject ordering on downstream graph measures; and the effective dimensionality of the graph battery. We release the implementation so that any group can produce this report for their own data, which — we argue — is the only place these quantities can legitimately come from.

---

## 2. Materials and methods

### 2.1 Cohorts

| Cohort | Acquisition | Directions | Subjects | Role |
| --- | --- | --- | --- | --- |
| 1 | three-shell, b = 0/1000/2000/3000 | 117 | 126 | primary |
| 2 | two-shell, b = 0/1000/3000 | 129 | 117 | independent replication |
| 3 | free q-space, 15 b-values 200–3000 | 103 | 52 | `[NOTE] pending reconstruction` |
| 4 | OpenNeuro, multi-site | — | — | `[NOTE] planned; see §5.3` |

All diffusion data were preprocessed with QSIPrep `[VERIFY version]` and reconstructed in DSI Studio (QSDR). Cohorts 1–3 originate from a single site; this is a limitation addressed by cohort 4 (§5.3).

`[NOTE] Add scanner vendor, field strength, voxel size, TE/TR per cohort before submission.`

### 2.2 Cohort screening

Scans whose preprocessed diffusion signal was a robust outlier within its cohort (median/MAD on QSIPrep's final-image contrast metric, |z| > 3.5) were excluded prior to analysis. This flagged 7/408 scans in cohort 1 and 4/306 in cohort 2. In cohort 1 the two most extreme scans were the two independently identified as producing implausible tractography, while their raw data appeared normal — motivating the screen's inclusion as a routine step.

One scan per subject (the first session) entered every analysis. Later sessions were excluded by design: in longitudinal cohorts they typically carry the effect under study, so treating them as repeats would discard signal.

### 2.3 The specification multiverse

We varied anisotropy threshold, turning angle and track/voxel ratio in a full factorial (12 specifications), holding algorithm, step size and smoothing at the values used in the cohort's own published analysis. Each specification was tracked twice per subject with different random seeds, producing a within-specification noise estimate.

Streamline count was treated as a separate axis (5,000 and 50,000) to test whether conclusions obtained cheaply transfer to expensive settings. Tracking algorithm was varied in a further comparison.

Each analysis was run in **two independent subject waves**, which provides the replication needed to distinguish a real effect from a sampling artefact — a check we apply to every statistic reported below, including our own.

### 2.4 Quantities reported

All distances are `1 − r` between `log1p`-transformed upper-triangular edge vectors.

- **Discriminability** — probability that a repeat of a subject is closer to that subject than to another subject.
- **Margin** — mean over within-subject repeat pairs of (nearest between-subject distance − within-subject distance); unbounded above, so it does not share discriminability's ceiling.
- **Variance decomposition** — `tracking_noise` (same scan, same specification, different seed), `parameter` (same scan, different specification) and `between_subject`, each as a distribution.
- **Graph-measure reliability** — one-way ICC(1,1) of subjects against tracking repeats, per global measure.
- **Parameter fragility** — Spearman agreement of the *subject ordering* on each graph measure, between specifications, contrasted against the same agreement between repeats of one specification.
- **Effective dimensionality** — from the eigenvalue spectrum of the z-scored graph-measure battery.

---

## 3. Results

### 3.1 The reliability screen saturates

Discriminability was exactly 1.000 for every specification, in both waves, in both cohorts, at both streamline counts and under both tracking algorithms. Pooling waves to 20 subjects did not change this.

Saturation was not a consequence of negligible noise. In cohort 1 at 50,000 streamlines, tracking noise reached 20% (streamline count), 39% (FA-weighted) and 39% (QA-weighted) of the corresponding between-subject distance, and the statistic still attained its ceiling.

### 3.2 Ordering is stable; the winner is not

Specification ordering by margin agreed across independent waves at ρ = 0.87–0.92, and transferred across a tenfold change in streamline count (ρ = 0.63–0.93), across tracking algorithms (ρ = 0.94–0.97) and across the two acquisition schemes of cohorts 1 and 2 (ρ = 0.94–0.97).

The top two specifications, however, were separated by 0.0013 against a wave-to-wave variation of 0.0090 — a sevenfold noise excess — in **both** cohorts independently. They differed in exactly one parameter. The data therefore determined the anisotropy threshold and turning angle, and did not determine the track/voxel ratio.

### 3.3 The multiverse moves the connectome

| Cohort 1, 50k | tracking noise | parameter | between subject | parameter / between-subject |
| --- | --- | --- | --- | --- |
| count | 0.0114 | 0.0511 | 0.2461 | **0.21** |
| FA-weighted | 0.0922 | 0.1910 | 0.4325 | **0.44** |
| QA-weighted | 0.0924 | 0.1929 | 0.4392 | **0.44** |

At 5,000 streamlines the same ratios were inflated (0.31–0.51) because 65–76% of the apparent parameter effect was tracking noise, against 22–48% at 50,000. Under-powered screening therefore *overstates* parameter sensitivity.

### 3.4 Subject ordering is specification-dependent

| Cohort | re-run, identical specification | across specifications |
| --- | --- | --- |
| 1 (three-shell) | ρ = 0.93 | ρ = 0.66 |
| 2 (two-shell) | ρ = 0.94 | ρ = 0.75 |

Every measure, both cohorts, without exception: the specification disrupts subject ordering far more than tracking stochasticity does. A quarter to a third of the ordering on which a group analysis rests is contingent on the specification.

Notably, high test-retest reliability does not protect against this. Binary clustering attained ICC = 0.95 while being among the most specification-sensitive measures — reproducible under a fixed specification, and re-ordered by changing it.

### 3.5 The graph battery is about two-dimensional

Effective dimensionality of the seven global measures was 1.8 (cohort 1) and 2.0 (cohort 2), with the first component explaining 68–74%. Reporting the battery as independent outcomes inflates one finding into several.

We attempted to rank individual measures by fragility. The ranking did not replicate across split halves even at 79 subjects, which is the expected consequence of a battery carrying two independent dimensions. We therefore report the aggregate contrast and per-measure values, and decline to label individual measures robust or fragile.

`[NOTE] §3.6 — cohort 3 (free q-space) results, pending reconstruction.`
`[NOTE] §3.7 — sensitivity analysis: published specification vs screened alternative, currently running.`

---

## 4. Discussion

### 4.1 Why we publish no recommended specification

Three reasons, each sufficient on its own. The separation between leading specifications was smaller than the noise on that separation, so a recommendation would propagate a distinction the data do not support. The criterion measures identifiability rather than anatomical validity, so "ranked first" does not mean "closer to the truth". And cohorts 1–3 share a site, scanner and population, so their agreement is evidence that screening is *stable and cheap* — not that a specification transfers to other scanners, field strengths or clinical populations.

What we offer instead are expected magnitudes, as calibration for what a study should anticipate before measuring its own.

### 4.2 Implications for practice

- **Report the specification in full.** It is not a detail; it moves the connectome by a fifth to a half of a between-subject difference.
- **Report a sensitivity analysis, not an optimality claim.** Re-running under one alternative specification and showing the conclusion holds is achievable, and is stronger evidence than any selection procedure.
- **Do not treat a global graph battery as independent findings.** Its effective dimensionality is near two.
- **Check that a chosen specification's lead exceeds its own sampling noise** before describing it as preferred.

### 4.3 Limitations

Repeats are re-runs of a single acquisition, so they capture algorithmic but not measurement noise; the true multiverse is therefore wider than the one measured here. All criteria assess reliability, and reliability is not validity: a specification can reproducibly generate false-positive connections. Cohorts 1–3 share a site. The multiverse explored is a factorial over three tracking parameters within one software package, not over parcellations, algorithms or packages.

---

## 5. Software and reproducibility

### 5.1 Availability

OptiConn (MIT licence) implements every quantity reported here. `[NOTE] Add Zenodo DOI on release.`

### 5.2 Reproducibility

Deterministic seeds, configuration snapshots and machine-readable outputs per run.

### 5.3 Planned extension: a cross-protocol test battery

`[NOTE] Design pending — see the companion spec. Intent: OpenNeuro cohorts chosen for maximal acquisition diversity (field strength, coil, vendor, b-value scheme, resolution), processed on SLURM, to test whether the magnitudes in §3.3–3.5 hold outside a single site. This is the principal limitation of the present draft.`

---

## References

`[NOTE] All entries to be verified against source before submission.`

- Botvinik-Nezer et al. (2020). Variability in the analysis of a single neuroimaging dataset by many teams. *Nature*. `[VERIFY]`
- Steegen et al. (2016). Increasing transparency through a multiverse analysis. *Perspectives on Psychological Science*. `[VERIFY]`
- Maier-Hein et al. (2017). The challenge of mapping the human connectome based on diffusion tractography. *Nature Communications*. `[VERIFY]`
- Zalesky et al. (2016). Connectome sensitivity or specificity: which is more important? *NeuroImage*, 142, 407–420. `[VERIFY]`
- Bridgeford et al. (2021). Eliminating accidental deviations to minimize generalization error. `[VERIFY]`
- Rubinov & Sporns (2010). Complex network measures of brain connectivity. *NeuroImage*. `[VERIFY]`
