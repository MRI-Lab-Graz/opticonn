import pandas as pd
import sys
import os

# Usage: python check_metric_uniqueness.py <path_to_combo_diagnostics.csv>
if len(sys.argv) < 2:
    print("Usage: python check_metric_uniqueness.py <path_to_combo_diagnostics.csv>")
    sys.exit(1)

csv_path = sys.argv[1]
if not os.path.exists(csv_path):
    print(f"File not found: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)

# Identify parameter columns (including flattened param_* columns)
param_cols = [c for c in df.columns if c.startswith("param_")]
for base_col in ["tract_count", "thread_count", "total_combinations"]:
    if base_col in df.columns and base_col not in param_cols:
        param_cols.append(base_col)

# Metrics are numeric columns not classified as parameters
numeric_dtypes = ["float64", "int64"]
metric_cols = [
    c for c in df.columns if df[c].dtype in numeric_dtypes and c not in param_cols
]

if not param_cols:
    print("⚠️  No parameter columns detected. Showing sweep identifiers only.\n")

print(f"Checking uniqueness for metrics in: {csv_path}\n")
problem_found = False
for metric in metric_cols:
    vals = df[metric].dropna().tolist()
    unique_vals = set(vals)
    n_unique = len(unique_vals)
    n_total = len(vals)
    if n_unique < n_total:
        print(
            f"⚠️  Metric '{metric}': {n_total-n_unique} duplicates out of {n_total} values. Only {n_unique} unique."
        )
        problem_found = True
        duplicated_groups = df.groupby(metric)
        for value, group in duplicated_groups:
            if len(group) <= 1:
                continue
            display_value = value
            if pd.api.types.is_float_dtype(df[metric]):
                display_value = f"{value:.6f}"
            if "sweep_id" in group.columns:
                sweeps = [str(x) for x in group["sweep_id"].tolist()]
            else:
                sweeps = [str(x) for x in group.index.tolist()]
            print(f"   ↳ Value {display_value} appears in sweeps: {', '.join(sweeps)}")
            if param_cols:
                for _, row in group.iterrows():
                    param_bits = []
                    for col in param_cols:
                        val = row.get(col)
                        if pd.isna(val):
                            continue
                        param_bits.append(f"{col}={val}")
                    summary = (
                        ", ".join(param_bits)
                        if param_bits
                        else "no parameter columns available"
                    )
                    identifier = row.get("sweep_id", row.name)
                    print(f"      • {identifier}: {summary}")
    else:
        print(f"✅ Metric '{metric}': All {n_total} values are unique.")

if not problem_found:
    print("\nAll metrics are unique for all parameter combinations.")
else:
    print("\nReview your analysis pipeline and parameter grid for possible issues.")
