import dash
from dash import dcc, html, Input, Output
from dash.dash_table import DataTable
import pandas as pd
import plotly.express as px
import os


def find_pareto_csv(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "pareto_front.csv":
                return os.path.join(root, file)
    return None


def find_optimal_combinations(base_dir):
    """Find optimal_combinations.json files from wave selections"""
    import json
    import glob

    pattern = os.path.join(base_dir, "**/03_selection/optimal_combinations.json")
    files = glob.glob(pattern, recursive=True)

    all_combinations = []
    for file_path in files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    # Add wave info from path
                    wave_name = (
                        file_path.split("/")[-3] if "/" in file_path else "unknown"
                    )
                    for item in data:
                        item["wave"] = wave_name
                    all_combinations.extend(data)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    return all_combinations


def load_data(output_folder):
    """Load candidate data - prefer optimal_combinations.json over pareto CSV"""
    # First try to load optimal combinations (has atlas/metric info)
    combinations = find_optimal_combinations(output_folder)
    if combinations:
        df = pd.DataFrame(combinations)

        # Add cross-wave consistency metrics
        if (
            "wave" in df.columns
            and "atlas" in df.columns
            and "connectivity_metric" in df.columns
        ):
            # Create unique key for each candidate (atlas + metric combination)
            df["candidate_key"] = df["atlas"] + "_" + df["connectivity_metric"]

            # Count how many waves each candidate appears in
            wave_counts = df.groupby("candidate_key")["wave"].nunique().to_dict()
            df["waves_present"] = df["candidate_key"].map(wave_counts)

            # Calculate average QA score across waves
            if "pure_qa_score" in df.columns:
                avg_qa = df.groupby("candidate_key")["pure_qa_score"].mean().to_dict()
                df["avg_qa_across_waves"] = df["candidate_key"].map(avg_qa)

                # Calculate QA standard deviation (consistency metric)
                qa_std = (
                    df.groupby("candidate_key")["pure_qa_score"]
                    .std()
                    .fillna(0)
                    .to_dict()
                )
                df["qa_std"] = df["candidate_key"].map(qa_std)
                df["qa_consistency"] = df["candidate_key"].map(
                    lambda k: 1.0 - min(qa_std.get(k, 0), 1.0)
                )

        return df

    # Fallback to Pareto CSV
    pareto_csv = find_pareto_csv(output_folder)
    if pareto_csv:
        return pd.read_csv(pareto_csv)

    return pd.DataFrame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OptiConn Sweep Explorer (Dash)")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
        help="Sweep output folder containing pareto_front.csv",
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Port to run Dash app on"
    )
    args = parser.parse_args()

    df = load_data(args.output)

    # Build wave consistency summary and recommendation
    wave_summary = ""
    recommendation = ""
    if "waves_present" in df.columns and "pure_qa_score" in df.columns:
        total_waves = df["wave"].nunique()
        consistent_candidates = df[df["waves_present"] == total_waves][
            "candidate_key"
        ].nunique()

        # Find best candidate: appears in all waves with highest avg QA
        best_candidates = df[df["waves_present"] == total_waves].copy()
        if (
            not best_candidates.empty
            and "avg_qa_across_waves" in best_candidates.columns
        ):
            best_idx = best_candidates["avg_qa_across_waves"].idxmax()
            best = df.loc[best_idx]
            recommendation = (
                f"🏆 **Recommended:** {best['atlas']} + {best['connectivity_metric']} "
                f"(QA: {best.get('avg_qa_across_waves', 0):.3f}, appears in {int(best['waves_present'])} waves)"
            )

        wave_summary = (
            f"📊 **Cross-Wave Analysis:** {total_waves} waves compared | "
            f"{consistent_candidates} candidates consistent across all waves"
        )

    app = dash.Dash(__name__)
    app.title = "OptiConn Interactive Review"

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H1(
                        "🎯 OptiConn Interactive Review",
                        style={"color": "#2c3e50", "marginBottom": "10px"},
                    ),
                    html.H3(
                        "Select Your Optimal Tractography Parameters",
                        style={"color": "#7f8c8d", "marginTop": "0"},
                    ),
                ],
                style={
                    "backgroundColor": "#ecf0f1",
                    "padding": "20px",
                    "borderRadius": "10px",
                    "marginBottom": "20px",
                },
            ),
            html.Div(
                [
                    html.H4("📋 Instructions", style={"color": "#2980b9"}),
                    html.Ol(
                        [
                            html.Li("Review the cross-wave analysis summary below"),
                            html.Li(
                                "Explore candidates using the interactive visualization"
                            ),
                            html.Li(
                                "Click on a row in the table to select a candidate"
                            ),
                            html.Li(
                                "Click the green 'Select This Candidate' button to save your choice"
                            ),
                            html.Li(
                                "Use the displayed command to apply settings to your full dataset"
                            ),
                        ],
                        style={"fontSize": "16px", "lineHeight": "1.8"},
                    ),
                ],
                style={
                    "backgroundColor": "#e8f4f8",
                    "padding": "15px",
                    "borderRadius": "8px",
                    "marginBottom": "20px",
                    "border": "2px solid #3498db",
                },
            ),
            (
                dcc.Markdown(
                    wave_summary,
                    style={
                        "fontSize": "18px",
                        "fontWeight": "bold",
                        "color": "#34495e",
                    },
                )
                if wave_summary
                else html.Div()
            ),
            (
                dcc.Markdown(
                    recommendation,
                    style={
                        "backgroundColor": "#d4edda",
                        "padding": "15px",
                        "borderRadius": "8px",
                        "marginTop": "10px",
                        "fontSize": "18px",
                        "border": "2px solid #28a745",
                    },
                )
                if recommendation
                else html.Div()
            ),
            html.Hr(),
            html.H3("📊 Interactive Visualization", style={"color": "#2c3e50"}),
            html.P(
                "Explore parameter relationships by selecting axes:",
                style={"fontSize": "14px", "color": "#7f8c8d"},
            ),
            dcc.Dropdown(
                id="x-axis",
                options=[{"label": col, "value": col} for col in df.columns],
                value=df.columns[0] if not df.empty else None,
                placeholder="Select X axis parameter",
            ),
            dcc.Dropdown(
                id="y-axis",
                options=[{"label": col, "value": col} for col in df.columns],
                value=df.columns[1] if not df.empty else None,
                placeholder="Select Y axis parameter",
            ),
            dcc.Graph(id="pareto-plot"),
            html.Hr(),
            html.H3("📑 Candidate Selection Table", style={"color": "#2c3e50"}),
            html.Div(
                [
                    html.P(
                        [
                            "✨ ",
                            html.Strong("How to select: "),
                            "Click on any row in the table below to select it (row will turn green). Then click the 'Select This Candidate' button to save your choice.",
                        ],
                        style={
                            "fontSize": "16px",
                            "color": "#2c3e50",
                            "backgroundColor": "#fff3cd",
                            "padding": "12px",
                            "borderRadius": "6px",
                            "border": "1px solid #ffc107",
                        },
                    )
                ],
                style={"marginBottom": "15px"},
            ),
            dcc.Markdown("No data found." if df.empty else ""),
            dcc.Loading(
                [
                    DataTable(
                        id="pareto-table",
                        columns=[{"name": i, "id": i} for i in df.columns],
                        data=df.to_dict("records"),
                        page_size=20,
                        style_table={"overflowX": "auto"},
                        row_selectable="single",
                        selected_rows=[],
                        sort_action="native",  # Enable native sorting
                        sort_mode="multi",  # Allow sorting by multiple columns
                        filter_action="native",  # Enable filtering
                        style_cell={
                            "textAlign": "left",
                            "padding": "10px",
                            "minWidth": "100px",
                        },
                        style_header={
                            "backgroundColor": "#f0f0f0",
                            "fontWeight": "bold",
                            "border": "1px solid #ddd",
                        },
                        style_data_conditional=[
                            {"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"},
                            {
                                "if": {"state": "selected"},
                                "backgroundColor": "#d4edda",
                                "border": "2px solid #28a745",
                            },
                        ],
                    ),
                    html.Button(
                        "✅ Select This Candidate for Full Dataset",
                        id="export-btn",
                        n_clicks=0,
                        style={
                            "marginTop": "20px",
                            "backgroundColor": "#28a745",
                            "color": "white",
                            "padding": "15px 30px",
                            "fontSize": "18px",
                            "fontWeight": "bold",
                            "border": "none",
                            "borderRadius": "8px",
                            "cursor": "pointer",
                            "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
                        },
                    ),
                    html.Div(
                        id="export-status",
                        style={
                            "marginTop": "15px",
                            "fontSize": "16px",
                            "padding": "15px",
                            "borderRadius": "8px",
                        },
                    ),
                ]
                if not df.empty
                else html.Div()
            ),
        ]
    )

    @app.callback(
        Output("pareto-plot", "figure"),
        Input("x-axis", "value"),
        Input("y-axis", "value"),
    )
    def update_plot(x, y):
        if df.empty or not x or not y:
            return {}
        fig = px.scatter(df, x=x, y=y, color=df.columns[0], hover_data=df.columns)
        fig.update_layout(title=f"Pareto Front: {x} vs {y}")
        return fig

    import json
    from dash.dependencies import State

    @app.callback(
        Output("export-status", "children"),
        Input("export-btn", "n_clicks"),
        State("pareto-table", "selected_rows"),
    )
    def export_selected(n_clicks, selected_rows):
        if n_clicks > 0 and selected_rows:
            # Since row_selectable='single', selected_rows will have only one element
            selected_idx = selected_rows[0]
            selected_candidate = df.iloc[selected_idx].to_dict()

            # Check if this is an optimal_combinations format (has atlas/metric)
            # If so, wrap it in a list for compatibility with apply command
            has_atlas = (
                "atlas" in selected_candidate
                and "connectivity_metric" in selected_candidate
            )

            out_path = os.path.join(args.output, "selected_candidate.json")
            try:
                with open(out_path, "w") as f:
                    # Save as list if it has atlas/metric (optimal_combinations format)
                    # This way apply command can process it correctly
                    if has_atlas:
                        json.dump([selected_candidate], f, indent=2)
                    else:
                        json.dump(selected_candidate, f, indent=2)

                # Build helpful next steps message
                atlas_info = (
                    f"{selected_candidate.get('atlas', 'N/A')} + {selected_candidate.get('connectivity_metric', 'N/A')}"
                    if has_atlas
                    else "Selected configuration"
                )
                qa_score = (
                    f" (QA: {selected_candidate.get('pure_qa_score', 0):.3f})"
                    if "pure_qa_score" in selected_candidate
                    else ""
                )

                return html.Div(
                    [
                        html.H4(
                            "✅ Selection Saved Successfully!",
                            style={"color": "#28a745", "marginBottom": "10px"},
                        ),
                        html.P(
                            [html.Strong("Selected: "), f"{atlas_info}{qa_score}"],
                            style={"fontSize": "16px", "marginBottom": "10px"},
                        ),
                        html.P(
                            [
                                html.Strong("Saved to: "),
                                html.Code(
                                    out_path,
                                    style={
                                        "backgroundColor": "#f8f9fa",
                                        "padding": "4px 8px",
                                        "borderRadius": "4px",
                                    },
                                ),
                            ],
                            style={"fontSize": "14px", "marginBottom": "15px"},
                        ),
                        html.Div(
                            [
                                html.P(
                                    html.Strong("🚀 Next Step: Apply to Full Dataset"),
                                    style={"color": "#2c3e50", "marginBottom": "8px"},
                                ),
                                html.Pre(
                                    f"opticonn apply \\\n  --data-dir <your_full_dataset_directory> \\\n  --optimal-config {out_path} \\\n  --output-dir <output_directory>",
                                    style={
                                        "backgroundColor": "#2c3e50",
                                        "color": "#ecf0f1",
                                        "padding": "15px",
                                        "borderRadius": "6px",
                                        "fontSize": "14px",
                                        "fontFamily": "monospace",
                                        "overflowX": "auto",
                                    },
                                ),
                            ],
                            style={
                                "backgroundColor": "#e8f4f8",
                                "padding": "15px",
                                "borderRadius": "8px",
                                "border": "2px solid #3498db",
                            },
                        ),
                    ],
                    style={"backgroundColor": "#d4edda", "border": "2px solid #28a745"},
                )

            except Exception as e:
                return html.Div(
                    [
                        html.H4("❌ Export Failed", style={"color": "#dc3545"}),
                        html.P(f"Error: {e}", style={"fontSize": "14px"}),
                    ],
                    style={
                        "backgroundColor": "#f8d7da",
                        "color": "#721c24",
                        "border": "2px solid #dc3545",
                    },
                )
        elif n_clicks > 0:
            return html.Div(
                [
                    html.H4("⚠️  No Candidate Selected", style={"color": "#856404"}),
                    html.P(
                        "Please click on a row in the table above to select a candidate first.",
                        style={"fontSize": "14px"},
                    ),
                ],
                style={
                    "backgroundColor": "#fff3cd",
                    "color": "#856404",
                    "border": "2px solid #ffc107",
                },
            )
        return ""

    print("\n" + "=" * 70)
    print("✅ OptiConn Interactive Review Dashboard is running!")
    print("=" * 70)
    print(f"🌐 Open your browser and navigate to: http://localhost:{args.port}")
    print(f"📂 Reviewing sweep results from: {args.output}")
    print("\n💡 Press Ctrl+C to stop the server when done")
    print("=" * 70 + "\n")

    app.run(debug=True, port=args.port)
