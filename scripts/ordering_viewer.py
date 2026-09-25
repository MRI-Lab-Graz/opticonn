"""Subject-ordering viewer: how far a parameter choice moves your subjects.

Emits one self-contained HTML file per sweep. A reader drags a slider across
candidate specifications and watches subject ranks on a graph measure reshuffle
against DSI Studio's untouched defaults, with the tracking-noise floor -- what
re-running the identical specification already costs -- shown alongside.

This is reporting, not selection: nothing here feeds scripts.reliability.rank().
See docs/superpowers/specs/2026-09-25-ordering-viewer-design.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from scripts.compute_network_measures_from_connectivity import measures_from_matrix
from scripts.variance_decomposition import collect_sweep_matrices

MIN_SUBJECTS_FOR_ORDERING = 3


class ReferenceMissing(RuntimeError):
    """No combo in a wave was flagged as the DSI Studio defaults reference."""


def _combo_meta(sweep_optimize_dir: Path) -> dict[str, dict]:
    """{combo_id: diagnostics.json}, keyed exactly as collect_sweep_matrices keys."""
    meta = {}
    for path in sorted(Path(sweep_optimize_dir).glob("*/combos/sweep_*/diagnostics.json")):
        combo_dir = path.parent
        meta[f"{combo_dir.parent.parent.name}/{combo_dir.name}"] = json.loads(path.read_text())
    return meta


def _references(meta: dict[str, dict], waves: list[str]) -> dict[str, str]:
    refs = {}
    for wave in waves:
        flagged = [cid for cid, m in meta.items()
                   if m.get("wave", cid.split("/")[0]) == wave and m.get("reference")]
        if not flagged:
            raise ReferenceMissing(
                f"{wave} has no combo flagged reference. Re-run the sweep with "
                f"sweep_parameters.reference_candidate set to DSI Studio's defaults, "
                f"or the reference combo failed; check {wave}/combos/*/diagnostics.json. "
                f"Displacement is meaningless without that origin."
            )
        refs[wave] = sorted(flagged)[0]
    return refs


def _varying(meta: dict[str, dict]) -> list[str]:
    """Parameters that take more than one value across non-reference combos."""
    seen: dict[str, set] = {}
    for m in meta.values():
        if m.get("reference"):
            continue
        for key, value in (m.get("parameters") or {}).items():
            seen.setdefault(key, set()).add(value)
    return sorted(k for k, v in seen.items() if len(v) > 1)


def collect(sweep_optimize_dir: Path) -> dict:
    """Viewer payload for a completed sweep. See the module docstring."""
    sweep_optimize_dir = Path(sweep_optimize_dir)
    grouped = collect_sweep_matrices(sweep_optimize_dir)
    if not grouped:
        raise ValueError(
            f"no */combos/sweep_* directories with connectivity matrices under {sweep_optimize_dir}"
        )
    meta = _combo_meta(sweep_optimize_dir)
    waves = sorted({cid.split("/")[0] for cid in meta})
    references = _references(meta, waves)

    # {wave: [scan key]}, sorted -- the order that anonymised labels stand for.
    scan_keys: dict[str, list[str]] = {}
    for (atlas, metric), combos in grouped.items():
        for cid, scans in combos.items():
            wave = cid.split("/")[0]
            scan_keys.setdefault(wave, sorted(scans))
    for wave, keys in scan_keys.items():
        if len(keys) < MIN_SUBJECTS_FOR_ORDERING:
            raise ValueError(
                f"{wave} has {len(keys)} subject(s); rank correlation needs at least "
                f"{MIN_SUBJECTS_FOR_ORDERING}"
            )
    subjects = {w: [f"S{i:02d}" for i in range(1, len(k) + 1)] for w, k in scan_keys.items()}

    # values[combo_id][pair][measure] -> [rep1 values, rep2 values], one per subject
    values: dict[str, dict[str, dict[str, list[list[float]]]]] = {}
    measure_names: set[str] | None = None
    for (atlas, metric), combos in sorted(grouped.items()):
        pair = f"{atlas}/{metric}"
        for cid, scans in combos.items():
            keys = scan_keys[cid.split("/")[0]]
            if any(len(scans.get(k, [])) < 2 for k in keys):
                logging.warning("ordering viewer: %s lacks 2 repeats for some subject; skipped", cid)
                continue
            per_repeat = [[measures_from_matrix(scans[k][rep]) for k in keys] for rep in (0, 1)]
            names = set(per_repeat[0][0])
            measure_names = names if measure_names is None else measure_names & names
            slot = values.setdefault(cid, {}).setdefault(pair, {})
            for name in names:
                slot[name] = [[tbl[name] for tbl in rep] for rep in per_repeat]

    measures = sorted(measure_names or ())
    pairs = sorted({p for c in values.values() for p in c})

    for wave, ref_id in references.items():
        if ref_id not in values:
            raise ReferenceMissing(
                f"{wave}: reference combo {ref_id} has no usable data (missing "
                f"connectivity matrices or fewer than 2 repeats for some subject); "
                f"check {ref_id}/diagnostics.json"
            )

    combos_out = []
    for cid in sorted(values):
        wave = cid.split("/")[0]
        ref_id = references[wave]
        cmeta = meta.get(cid)
        if cmeta is None:
            raise ValueError(f"{cid} has connectivity matrices but no diagnostics.json")
        entry = {
            "id": cid,
            "wave": wave,
            "reference": bool(cmeta.get("reference")),
            "params": cmeta.get("parameters") or {},
            "ranks": {}, "rho_noise": {}, "rho_vs_reference": {},
        }
        for pair in pairs:
            if pair not in values[cid]:
                continue
            entry["ranks"][pair] = {}
            entry["rho_noise"][pair] = {}
            entry["rho_vs_reference"][pair] = {}
            for name in measures:
                rep1, rep2 = values[cid][pair][name]
                entry["ranks"][pair][name] = _ranks(rep1)
                entry["rho_noise"][pair][name] = _rho(rep1, rep2)
                ref = values.get(ref_id, {}).get(pair, {}).get(name)
                entry["rho_vs_reference"][pair][name] = _rho(rep1, ref[0]) if ref else None
        combos_out.append(entry)

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "pairs": pairs,
        "measures": measures,
        "order_params": _varying(meta),
        "waves": waves,
        "subjects": subjects,
        "reference": references,
        "combos": combos_out,
    }


def _ranks(values: list[float]) -> list[float | None]:
    """rankdata over finite values only; a non-finite subject is dropped from the
    panel (None, not a bogus bottom rank) -- the same population _rho() computes
    over, matching the drop-and-report-the-count failure mode scripts/graph_icc.py
    uses for the identical situation."""
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    dropped = int((~finite).sum())
    if dropped:
        logging.warning("ordering viewer: %d subject(s) with a non-finite value dropped", dropped)
    out: list[float | None] = [None] * len(arr)
    if finite.any():
        ranked = iter(rankdata(arr[finite]))
        for i, ok in enumerate(finite):
            if ok:
                out[i] = float(next(ranked))
    return out


def _rho(a: list[float], b: list[float]) -> float | None:
    """Spearman of two subject orderings; None when either is constant or too short."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < MIN_SUBJECTS_FOR_ORDERING:
        return None
    a, b = a[finite], b[finite]
    if a.std() == 0 or b.std() == 0:
        return None
    value = float(spearmanr(a, b)[0])
    return None if np.isnan(value) else value


_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Subject ordering under parameter choice</title>
<style>
  :root { --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#bbb; --accent:#c2410c; --ok:#0369a1; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
    --bg:#15171a; --fg:#e8e8e8; --muted:#9aa0a6; --line:#555; --accent:#fb923c; --ok:#38bdf8; } }
  body { background:var(--bg); color:var(--fg); margin:0; padding:16px;
         font:15px/1.5 system-ui, -apple-system, sans-serif; }
  .wrap { max-width:900px; margin:0 auto; }
  .controls { display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin:16px 0; }
  select, input[type=range] { font:inherit; }
  .readout { display:flex; gap:24px; flex-wrap:wrap; margin:12px 0; }
  .readout div { font-variant-numeric:tabular-nums; }
  .big { font-size:1.6em; font-weight:600; }
  .muted { color:var(--muted); font-size:.85em; }
  svg { max-width:100%; height:auto; }
</style></head><body><div class="wrap">
<h1>How far does the parameter choice move your subjects?</h1>
<p class="muted">Left column: subject rank under DSI Studio's untouched defaults.
Right column: rank under the selected specification. Crossing lines are subjects the
choice re-ordered. <span id="src"></span></p>
<div class="controls">
  <label>Wave <select id="wave"></select></label>
  <label>Atlas / metric <select id="pair"></select></label>
  <label>Graph measure <select id="measure"></select></label>
  <label>Order by <select id="orderby"></select></label>
</div>
<div class="controls">
  <label style="flex:1">Specification
    <input type="range" id="slider" min="0" value="0" style="width:100%"></label>
</div>
<p id="params" class="muted"></p>
<div class="readout">
  <div><span class="big" id="rhoRef">-</span><br><span class="muted">vs. DSI Studio defaults</span></div>
  <div><span class="big" id="rhoNoise">-</span><br><span class="muted">same settings, re-run (noise floor)</span></div>
</div>
<svg id="slope" viewBox="0 0 640 420" role="img" aria-label="Subject rank slopegraph"></svg>
<h2>Every specification at once</h2>
<svg id="strip" viewBox="0 0 640 180" role="img" aria-label="Rank correlation per specification"></svg>
</div>
<script>
const PAYLOAD = __PAYLOAD__;
const $ = id => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
// Presentation attributes (fill="var(--x)") are not guaranteed to resolve CSS
// custom properties in every engine; move any var(...) color onto style=
// instead, where var() substitution is reliable.
const el = (n, a) => { const e = document.createElementNS(NS, n);
  let style = "";
  for (const k in a) {
    const v = a[k];
    if ((k === "fill" || k === "stroke") && typeof v === "string" && v.indexOf("var(") === 0)
      style += k + ":" + v + ";";
    else
      e.setAttribute(k, v);
  }
  if (style) e.setAttribute("style", style);
  return e; };

$("src").textContent = "Generated " + PAYLOAD.generated_at + ".";
const fill = (sel, vals) => { sel.innerHTML = "";
  vals.forEach(v => { const o = document.createElement("option");
    o.value = v; o.textContent = v; sel.appendChild(o); }); };
fill($("wave"), PAYLOAD.waves);
fill($("pair"), PAYLOAD.pairs);
fill($("measure"), PAYLOAD.measures);
fill($("orderby"), PAYLOAD.order_params.length ? PAYLOAD.order_params : ["(none)"]);

function candidates() {
  const wave = $("wave").value, key = $("orderby").value;
  const list = PAYLOAD.combos.filter(c => c.wave === wave && !c.reference);
  if (PAYLOAD.order_params.includes(key))
    list.sort((a, b) => (a.params[key] - b.params[key]) || a.id.localeCompare(b.id));
  return list;
}

function drawSlope(combo, pair, measure, subjects) {
  const svg = $("slope"); svg.innerHTML = "";
  const refId = PAYLOAD.reference[combo.wave];
  const ref = PAYLOAD.combos.find(c => c.id === refId);
  const a = ref.ranks[pair] && ref.ranks[pair][measure];
  const b = combo.ranks[pair] && combo.ranks[pair][measure];
  if (!a || !b) { svg.appendChild(el("text", {x:20, y:40, fill:"var(--muted)"}))
    .textContent = "measure unavailable for this specification"; return; }
  const n = a.length, top = 30, bottom = 400, xl = 120, xr = 520;
  const y = r => top + (bottom - top) * (r - 1) / Math.max(1, n - 1);
  svg.appendChild(el("text", {x:xl, y:16, fill:"var(--muted)", "text-anchor":"middle",
    "font-size":"13"})).textContent = "DSI Studio defaults";
  svg.appendChild(el("text", {x:xr, y:16, fill:"var(--muted)", "text-anchor":"middle",
    "font-size":"13"})).textContent = "selected specification";
  for (let i = 0; i < n; i++) {
    const moved = a[i] !== b[i];
    svg.appendChild(el("line", {x1:xl, y1:y(a[i]), x2:xr, y2:y(b[i]),
      stroke: moved ? "var(--accent)" : "var(--line)",
      "stroke-width": moved ? 2 : 1, "stroke-opacity": moved ? 0.9 : 0.45}));
    svg.appendChild(el("text", {x:xl-10, y:y(a[i])+4, "text-anchor":"end",
      fill:"var(--muted)", "font-size":"12"})).textContent = subjects[i];
    svg.appendChild(el("text", {x:xr+10, y:y(b[i])+4, fill:"var(--muted)",
      "font-size":"12"})).textContent = subjects[i];
  }
}

function drawStrip(list, current, pair, measure) {
  const svg = $("strip"); svg.innerHTML = "";
  const top = 20, bottom = 140, x0 = 60, x1 = 600, naY = bottom + 12;
  // Spearman's rho is genuinely [-1, 1]; map the full range onto the plot area
  // instead of clamping to [0, 1], which would collapse every negative rho
  // (a real, displayed result) onto the 0.0 gridline.
  const y = rho => bottom - (bottom - top) * (Math.max(-1, Math.min(1, rho)) + 1) / 2;
  const x = i => list.length < 2 ? (x0+x1)/2 : x0 + (x1-x0) * i / (list.length-1);
  [-1, 0, 1].forEach(t => {
    svg.appendChild(el("line", {x1:x0, y1:y(t), x2:x1, y2:y(t),
      stroke:"var(--line)", "stroke-opacity":0.3}));
    svg.appendChild(el("text", {x:x0-10, y:y(t)+4, "text-anchor":"end",
      fill:"var(--muted)", "font-size":"12"})).textContent = t.toFixed(1);
  });
  const noise = list.map(c => (c.rho_noise[pair]||{})[measure]).filter(v => v !== null && v !== undefined);
  if (noise.length) {
    const lo = Math.min(...noise), hi = Math.max(...noise);
    svg.appendChild(el("rect", {x:x0, y:y(hi), width:x1-x0, height:Math.max(1, y(lo)-y(hi)),
      fill:"var(--ok)", "fill-opacity":0.15}));
    svg.appendChild(el("text", {x:x1, y:y(hi)-6, "text-anchor":"end", fill:"var(--ok)",
      "font-size":"12"})).textContent = "noise floor (same settings, re-run)";
  }
  let d = "", hasNA = false;
  list.forEach((c, i) => {
    const v = (c.rho_vs_reference[pair]||{})[measure];
    const isCurrent = c.id === current.id;
    if (v === null || v === undefined) {
      // rho can be genuinely undefined (degenerate or too-short ranking).
      // Still mark the selected point's column so the reader can find it,
      // at a reserved position below the axis rather than dropping it.
      hasNA = true;
      svg.appendChild(el("circle", {cx:x(i), cy:naY, r: isCurrent ? 6 : 3,
        fill:"none", stroke:"var(--muted)", "stroke-width":1.5}));
      return;
    }
    d += (d ? " L" : "M") + x(i) + " " + y(v);
    svg.appendChild(el("circle", {cx:x(i), cy:y(v), r: isCurrent ? 6 : 3,
      fill:"var(--accent)"}));
  });
  if (d) svg.appendChild(el("path", {d, fill:"none", stroke:"var(--accent)", "stroke-width":1.5}));
  if (hasNA) svg.appendChild(el("text", {x:x0-10, y:naY+4, "text-anchor":"end",
    fill:"var(--muted)", "font-size":"11"})).textContent = "n/a";
  svg.appendChild(el("text", {x:x0, y:170, fill:"var(--muted)", "font-size":"12"}))
    .textContent = "each point is one specification, ordered by " + $("orderby").value;
}

function draw() {
  const list = candidates();
  $("slider").max = Math.max(0, list.length - 1);
  const combo = list[Math.min($("slider").value, list.length - 1)];
  if (!combo) {
    // Nothing to show for this wave: clear the previous wave's chart and
    // readouts rather than leaving them on screen mislabeled as the new one.
    $("slope").innerHTML = ""; $("strip").innerHTML = "";
    $("slope").appendChild(el("text", {x:20, y:40, fill:"var(--muted)"}))
      .textContent = "no candidate specifications in this wave";
    $("rhoRef").textContent = "-";
    $("rhoNoise").textContent = "-";
    $("params").textContent = "";
    return;
  }
  const pair = $("pair").value, measure = $("measure").value;
  const fmt = v => (v === null || v === undefined) ? "n/a" : v.toFixed(2);
  $("rhoRef").textContent = fmt((combo.rho_vs_reference[pair]||{})[measure]);
  $("rhoNoise").textContent = fmt((combo.rho_noise[pair]||{})[measure]);
  $("params").textContent = Object.entries(combo.params)
    .map(([k, v]) => k + "=" + v).join(", ");
  drawSlope(combo, pair, measure, PAYLOAD.subjects[combo.wave]);
  drawStrip(list, combo, pair, measure);
}

["wave","pair","measure","orderby","slider"].forEach(id =>
  $(id).addEventListener("input", draw));
draw();
</script></body></html>
"""


def render(payload: dict) -> str:
    """One self-contained HTML page. No server, no CDN, no JavaScript dependency."""
    # Escape "</script>" so a payload string can never terminate the inline
    # <script> block early. "<" -> "<" is valid inside a JSON string and
    # leaves JSON.parse/JS-literal semantics unchanged.
    escaped = json.dumps(payload).replace("<", "\\u003c")
    return _TEMPLATE.replace("__PAYLOAD__", escaped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Self-contained subject-ordering viewer for a completed OptiConn sweep"
    )
    parser.add_argument("-i", "--input", required=True, help="a sweep's optimize/ directory")
    parser.add_argument("-o", "--output", default=None,
                        help="output HTML (default: <input>/ordering_viewer.html)")
    args = parser.parse_args(argv)

    out = Path(args.output) if args.output else Path(args.input) / "ordering_viewer.html"
    try:
        payload = collect(Path(args.input))
    except (ReferenceMissing, ValueError) as exc:
        logging.error("ordering viewer: %s", exc)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(payload), encoding="utf-8")
    logging.info("Ordering viewer written to %s", out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
