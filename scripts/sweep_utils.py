#!/usr/bin/env python3
# Supports: --dry-run (prints intended actions without running)
# When run without arguments the script prints help: parser.print_help()
"""
Sweep Utilities
===============

Hilfsfunktionen für Parameter-Sweeps:
- MATLAB-Range-Parser (start:step:end)
- Expansion von Ranges (Listen oder Strings)
- Sampling-Strategien: grid, random, (leichte) lhs
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Tuple


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def parse_matlab_range(expr: str) -> List[float]:
    """Parse "start:step:end" in string form into a list of numbers (inclusive end).

    Supports int and float. Keeps float precision with a small epsilon.
    """
    parts = [p.strip() for p in expr.split(':')]
    if len(parts) != 3 or not all(_is_float(p) for p in parts):
        raise ValueError(f"Invalid MATLAB range expression: {expr}")
    start, step, end = map(float, parts)
    if step == 0:
        raise ValueError("step must be non-zero")
    n = int(math.floor((end - start) / step + 0.0000001))
    seq = [start + i * step for i in range(n + 1)]
    # Ensure inclusive end within tolerance
    if (step > 0 and seq and seq[-1] < end - 1e-9) or (step < 0 and seq and seq[-1] > end + 1e-9):
        seq.append(start + (n + 1) * step)
    # If both endpoints are close to ints and step is int-like, cast to int
    as_int = _float_list_maybe_int(seq)
    return as_int


def _float_list_maybe_int(seq: List[float]) -> List[Any]:
    def is_int_like(v: float) -> bool:
        return abs(v - round(v)) < 1e-9
    if all(is_int_like(v) for v in seq):
        return [int(round(v)) for v in seq]
    return [float(v) for v in seq]


def expand_range(value: Any) -> List[Any]:
    """Expand a range value which can be:
    - list/tuple of numbers
    - string in MATLAB notation "start:step:end"
    Returns a flat list of numbers (int or float).
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        return parse_matlab_range(value)
    # Single value
    return [value]


def grid_product(param_values: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product over parameter values.
    Returns list of dicts mapping param->choice.
    """
    import itertools
    keys = list(param_values.keys())
    value_lists = [param_values[k] for k in keys]
    combos = []
    for tpl in itertools.product(*value_lists):
        combos.append({k: v for k, v in zip(keys, tpl)})
    return combos


def random_sampling(param_values: Dict[str, List[Any]], n_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Random sampling across provided discrete value lists.
    Assumes each param has a discrete set (already expanded list)."""
    rng = random.Random(seed)
    keys = list(param_values.keys())
    combos = []
    for _ in range(max(1, n_samples)):
        choice = {k: rng.choice(param_values[k]) for k in keys if param_values[k]}
        combos.append(choice)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in combos:
        key = tuple((k, c[k]) for k in sorted(c.keys()))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def lhs_sampling(param_values: Dict[str, List[Any]], n_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Lightweight Latin Hypercube Sampling over discrete value lists.

    Strategy: For each dim, shuffle indices and map ranks to available discrete levels.
    This is an approximation suitable when continuous ranges have already
    been discretized via expand_range.
    """
    rng = random.Random(seed)
    keys = list(param_values.keys())
    n = max(1, n_samples)
    # Build per-dimension index sequences
    index_sequences = {}
    for k in keys:
        m = max(1, len(param_values[k]))
        # Create n ranks and map to m levels
        ranks = list(range(n))
        rng.shuffle(ranks)
        # Map each rank to nearest level index
        idxs = [min(int(round(r / max(1, n - 1) * (m - 1))), m - 1) for r in ranks]
        index_sequences[k] = idxs

    combos = []
    for i in range(n):
        choice = {}
        for k in keys:
            vals = param_values[k]
            if not vals:
                continue
            idx = index_sequences[k][i % len(index_sequences[k])]
            choice[k] = vals[idx]
        combos.append(choice)
    # Deduplicate
    seen = set()
    unique = []
    for c in combos:
        key = tuple((k, c[k]) for k in sorted(c.keys()))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def build_param_grid_from_config(cfg: Dict[str, Any]) -> Tuple[Dict[str, List[Any]], Dict[str, str]]:
    """Extract sweepable parameters from config['sweep_parameters'].

    Returns a tuple (param_values, mapping), where param_values maps logical param keys
    to value lists, and mapping maps logical keys to (config path) descriptors:
    - 'tract_count' -> root key 'tract_count'
    - 'connectivity_threshold' -> 'connectivity_options.connectivity_threshold'
    - tracking params -> 'tracking_parameters.<name>'
    """
    sp = cfg.get('sweep_parameters', {}) or {}
    param_values: Dict[str, List[Any]] = {}
    mapping: Dict[str, str] = {}

    def add(name: str, values_any: Any, target: str):
        values = expand_range(values_any)
        if values:
            param_values[name] = values
            mapping[name] = target

    add('otsu_threshold', sp.get('otsu_range'), 'tracking_parameters.otsu_threshold')
    add('fa_threshold', sp.get('fa_threshold_range'), 'tracking_parameters.fa_threshold')
    add('min_length', sp.get('min_length_range'), 'tracking_parameters.min_length')
    add('max_length', sp.get('max_length_range'), 'tracking_parameters.max_length')
    add('track_voxel_ratio', sp.get('track_voxel_ratio_range'), 'tracking_parameters.track_voxel_ratio')
    add('turning_angle', sp.get('turning_angle_range'), 'tracking_parameters.turning_angle')
    add('step_size', sp.get('step_size_range'), 'tracking_parameters.step_size')
    add('smoothing', sp.get('smoothing_range'), 'tracking_parameters.smoothing')
    add('dt_threshold', sp.get('dt_threshold_range'), 'tracking_parameters.dt_threshold')

    add('connectivity_threshold', sp.get('connectivity_threshold_range'), 'connectivity_options.connectivity_threshold')
    add('tract_count', sp.get('tract_count_range'), 'tract_count')

    return param_values, mapping


def apply_param_choice_to_config(base_cfg: Dict[str, Any], choice: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    """Create a derived config dict with choice applied according to mapping."""
    import copy
    cfg = copy.deepcopy(base_cfg)
    for logical_name, value in choice.items():
        target = mapping.get(logical_name)
        if not target:
            continue
        path = target.split('.')
        cur = cfg
        for key in path[:-1]:
            if key not in cur or not isinstance(cur[key], dict):
                cur[key] = {}
            cur = cur[key]
        cur[path[-1]] = value
    return cfg
