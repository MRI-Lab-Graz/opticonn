import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import os

# Path to diagnostics CSV (update if needed)
DEFAULT_CSV = os.path.join(
    "results", "optimize", "comprehensive_optimization", "combo_diagnostics.csv"
)

# Try quick_sweep if comprehensive not present
if not os.path.exists(DEFAULT_CSV):
    DEFAULT_CSV = os.path.join(
        "results", "optimize", "comprehensive_optimization", "combo_diagnostics.csv"
    )
    if not os.path.exists(DEFAULT_CSV):
        DEFAULT_CSV = os.path.join("results", "optimize", "combo_diagnostics.csv")

# Load data
if not os.path.exists(DEFAULT_CSV):
    raise FileNotFoundError(
        f"Could not find diagnostics CSV at {DEFAULT_CSV}. Run a sweep first."
    )

df = pd.read_csv(DEFAULT_CSV)

exclude_params = ["thread_count"]
param_cols = [
    c
    for c in df.columns
    if any(
        x in c
        for x in ["threshold", "count", "tract", "fa", "min_length", "max_length"]
    )
    and c not in exclude_params
]
metric_cols = [
    c for c in df.columns if c not in param_cols and df[c].dtype in ["float64", "int64"]
]

app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.H2("OptiConn Parameter Sweep Interactive Dashboard"),
        html.Div(
            [
                html.Label("X Parameter:"),
                dcc.Dropdown(
                    id="xparam",
                    options=[{"label": c, "value": c} for c in param_cols],
                    value=param_cols[0] if param_cols else None,
                ),
                html.Label("Y Metric:"),
                dcc.Dropdown(
                    id="ymetric",
                    options=[{"label": c, "value": c} for c in metric_cols],
                    value=metric_cols[0] if metric_cols else None,
                ),
            ],
            style={"width": "40%", "display": "inline-block", "verticalAlign": "top"},
        ),
        dcc.Graph(id="mainplot"),
        html.Div(id="summary", style={"marginTop": 20}),
    ]
)


@app.callback(
    Output("mainplot", "figure"),
    Output("summary", "children"),
    Input("xparam", "value"),
    Input("ymetric", "value"),
)
def update_plot(xparam, ymetric):
    if not xparam or not ymetric:
        return {}, "Select both parameter and metric."
    fig = px.scatter(df, x=xparam, y=ymetric, hover_data=df.columns)
    fig.update_layout(title=f"{ymetric} vs {xparam}", height=500)
    # Uniqueness check
    metric_vals = df[ymetric].dropna().tolist()
    unique_vals = set(metric_vals)
    n_unique = len(unique_vals)
    n_total = len(metric_vals)
    if n_unique < n_total:
        summary = (
            f"Showing {n_total} combinations. X: {xparam}, Y: {ymetric}.\n"
            f"⚠️ Only {n_unique} unique {ymetric} values. {n_total-n_unique} duplicates detected.\n"
            f"Some parameter combinations yield identical metrics!\n"
            f"Note: This may indicate parameter insensitivity or redundant sweeps. Consider reviewing your parameter grid or metric definition."
        )
    else:
        summary = f"Showing {n_total} combinations. X: {xparam}, Y: {ymetric}. All values unique."
    return fig, summary


if __name__ == "__main__":
    app.run(debug=True)
