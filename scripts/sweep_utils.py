# Vendored copy of sweep_utils (minimal surface) - adapted from upstream sources
from __future__ import annotations

import itertools
import json
from typing import Any, Dict, List, Tuple


def grid_product(param_values: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Return all combinations for a dict of lists."""
    if not param_values:
        return []
    keys = list(param_values.keys())
    pools = [param_values[k] for k in keys]
    combos = []
    for prod in itertools.product(*pools):
        combos.append({k: v for k, v in zip(keys, prod)})
    return combos


def build_param_grid_from_config(
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, List[Any]], Dict[str, str]]:
    """Builds a simple param grid from a config dict. Supports keys ending with _range being lists."""
    sp = cfg.get("sweep_parameters") or {}
    param_values = {}
    mapping = {}
    for k, v in sp.items():
        if k.endswith("_range"):
            name = k[:-6]
            # If it's a string like '0.1:0.05:0.2', try to parse as start:step:end
            if isinstance(v, str) and ":" in v:
                try:
                    start, step, end = [float(x) for x in v.split(":")]
                    vals = []
                    x = start
                    while x <= end + 1e-12:
                        vals.append(round(x, 6))
                        x += step
                    param_values[name] = vals
                    mapping[name] = k
                except Exception:
                    # fallback to single string
                    param_values[name] = [v]
                    mapping[name] = k
            elif isinstance(v, list):
                param_values[name] = v
                mapping[name] = k
            else:
                param_values[name] = [v]
                mapping[name] = k
    return param_values, mapping


def random_sampling(param_values: Dict[str, List[Any]], n_samples: int, seed: int):
    import random

    rng = random.Random(seed)
    keys = list(param_values.keys())
    return [{k: rng.choice(param_values[k]) for k in keys} for _ in range(n_samples)]


def apply_param_choice_to_config(
    cfg: Dict[str, Any], choice: Dict[str, Any], mapping: Dict[str, str]
) -> Dict[str, Any]:
    """Return a standalone config for one combination (deep copy, ranges removed)."""
    out = json.loads(json.dumps(cfg))
    out.pop("sweep_parameters", None)
    tp = out.setdefault("tracking_parameters", {})
    for k, v in choice.items():
        if k not in out and k in tp:
            tp[k] = v
        else:
            out[k] = v
    return out
