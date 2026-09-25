# Subject-ordering viewer: design

Status: approved, not implemented.
Target: interactive supplement to the multiverse paper
(`docs/papers/2026-09-25-multiverse-methods-paper.md`, submitting to Aperture Neuro).

## Problem

The multiverse paper's most consequential claim is (iii): moving within the multiverse
re-orders subjects on global graph measures far more than re-running an identical
specification does. On our pilot cohort, subject ordering survived a re-run at Spearman
0.93 but only 0.66 across specifications.

That is a number in a table. A reader has no way to feel it, and no way to ask what it
looks like on a measure they care about. Nothing in OptiConn currently shows a user how
far their own choice displaced their own subjects.

## What this builds

A generator that reads any completed sweep directory and emits one self-contained HTML
file. A reader drags a slider across candidate specifications and watches subject ranks
reshuffle against a fixed baseline, with the tracking-noise floor shown alongside so the
reshuffle can be judged against what re-running the same settings already costs.

The same generator produces the committed public demo, so the demo cannot drift from the
tool.

## Baseline: DSI Studio defaults

The slider measures displacement from **DSI Studio's untouched default parameters**, not
from a grid cell.

This requires a change to the OpenNeuro battery
(`docs/superpowers/specs/2026-09-25-openneuro-multiverse-battery-design.md`): every
dataset runs a **13th combo, DSI Studio's defaults, as a fixed reference rather than a
grid cell**. It is excluded from candidate ranking — it is not a screened candidate — and
exists only as the origin displacement is measured from.

The 12-candidate grid varies FA threshold, turning angle and track/voxel ratio around
OptiConn's chosen values. `configs/braingraph_default_config.json` (`fa_threshold=0.1`,
`track_voxel_ratio=3.0`) is OptiConn's default, not DSI Studio's. Without the added
reference combo there is nothing on disk to slide away from.

The anchor also strengthens the paper: claim (iii) becomes a statement about distance from
the out-of-the-box setting that most published connectomes silently used, rather than
about distance between two arbitrary candidates.

## Components

### `scripts/ordering_viewer.py`

`collect(sweep_dir) -> payload`

Reuses `scripts.reliability.collect_sweep_matrices()` and
`scripts.compute_network_measures_from_connectivity.measures_from_matrix()` — the path
`scripts/graph_icc.py` already walks. Per-subject graph measures are not stored on disk;
they are computed from the connectivity matrices on demand, as ICC already does.

Payload holds, for each combo: its parameters, and for each connectivity metric and graph
measure, per-subject per-repeat values. Plus, precomputed per combo/metric/measure:

- `rho_noise`: Spearman of subject ordering, repeat 1 vs repeat 2, within that combo.
- `rho_vs_reference`: Spearman of subject ordering, combo vs the defaults reference,
  using repeat 1 of each.

Spearman is computed in Python (scipy is already a dependency); the JavaScript only draws.

Subject labels are emitted as `S01..Sn`. The demo ships public-data results only, but the
generator anonymizes unconditionally so a user sharing a viewer built from their own
cohort does not leak identifiers.

`render(payload) -> str`

Inlines the payload as JSON into one HTML template. No fetch, no server, no CDN, no build
step, no JavaScript dependency. The file opens over `file://`.

### CLI

`opticonn view -i <optimize dir> [-o viewer.html]`, added as an argparse subparser in
`scripts/opticonn_hub.py` beside `select`.

## The page

Controls:

- connectivity metric: `count` / `fa` / `qa`
- graph measure: whichever measures `measures_from_matrix()` returned for all combos
- order-by parameter: whichever parameters actually vary across the candidates (for the
  battery grid: `fa_threshold`, `turning_angle`, `track_voxel_ratio`) — determines the
  order candidates occupy along the slider
- the slider itself, one stop per candidate

Main mark: a **slopegraph**. Left column is subject rank under the DSI Studio defaults
reference, right column is rank under the currently selected candidate, one line per
subject. Both columns use repeat 1, the same repeat `rho_vs_reference` is computed from. Reshuffle is visible directly as crossing lines.

Two readouts above it: `rho_noise` for the current candidate, and `rho_vs_reference`.

Below: a static strip showing `rho_vs_reference` for every candidate at once, with a
marker at the slider position and a horizontal band at the noise floor. The strip is the
paper figure; the slider is what makes a reader believe it.

Payload size for the battery shape (13 combos x 3 metrics x ~7 measures x 10 subjects x
2 repeats) is roughly 60 KB inline.

## Failure modes

- **Defaults reference combo absent**: fail with an error naming the combo id expected.
  Never silently substitute a nearby candidate — the whole quantity is relative to that
  origin.
- **Fewer than 3 subjects**: refuse. Rank correlation over two subjects is not
  interpretable.
- **Non-finite value for a subject on one measure**: drop that subject from that measure's
  panel and report the count, matching `scripts/graph_icc.py`.
- **A measure missing from some combos**: omit it from the dropdown rather than showing a
  panel that changes population as the slider moves.

## Testing

One pytest against synthetic connectivity matrices with a known injected subject ordering,
in the style of the existing reliability tests:

- a candidate identical to the reference gives `rho_vs_reference` ~ 1
- a candidate built by shuffling subjects gives a markedly lower value
- a payload missing the reference combo raises, naming it

## Build order

The tool is written and verified against `studies/study129/cross_sectional_v1/optimize`
locally; that output is **not** committed. The committed demo HTML is generated once the
OpenNeuro battery has run, from public data only.

## Deliberately excluded

- **PCA view of connectome edge vectors.** Considered and dropped: visually striking, but
  a step removed from what a paper actually reports. Add it if the slopegraph turns out
  not to carry the argument.
- **Any server, bundler or JavaScript dependency.** A supplement a reader cannot open by
  double-clicking has failed at its only job.
- **Edge-level display.** Different tool.
- **Interactive re-ranking or candidate selection.** The viewer reports; it does not
  select. Selection stays in `tune-grid`.
