
import dash
from dash import dcc, html, Input, Output
from dash.dash_table import DataTable
import pandas as pd
import plotly.express as px
import os
import sys

def find_pareto_csv(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'pareto_front.csv':
                return os.path.join(root, file)
    return None

def find_optimal_combinations(base_dir):
    """Find optimal_combinations.json files from wave selections"""
    import json
    import glob
    pattern = os.path.join(base_dir, '**/03_selection/optimal_combinations.json')
    files = glob.glob(pattern, recursive=True)
    
    all_combinations = []
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    # Add wave info from path
                    wave_name = file_path.split('/')[-3] if '/' in file_path else 'unknown'
                    for item in data:
                        item['wave'] = wave_name
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
        if 'wave' in df.columns and 'atlas' in df.columns and 'connectivity_metric' in df.columns:
            # Create unique key for each candidate (atlas + metric combination)
            df['candidate_key'] = df['atlas'] + '_' + df['connectivity_metric']
            
            # Count how many waves each candidate appears in
            wave_counts = df.groupby('candidate_key')['wave'].nunique().to_dict()
            df['waves_present'] = df['candidate_key'].map(wave_counts)
            
            # Calculate average QA score across waves
            if 'pure_qa_score' in df.columns:
                avg_qa = df.groupby('candidate_key')['pure_qa_score'].mean().to_dict()
                df['avg_qa_across_waves'] = df['candidate_key'].map(avg_qa)
                
                # Calculate QA standard deviation (consistency metric)
                qa_std = df.groupby('candidate_key')['pure_qa_score'].std().fillna(0).to_dict()
                df['qa_std'] = df['candidate_key'].map(qa_std)
                df['qa_consistency'] = df['candidate_key'].map(lambda k: 1.0 - min(qa_std.get(k, 0), 1.0))
        
        return df
    
    # Fallback to Pareto CSV
    pareto_csv = find_pareto_csv(output_folder)
    if pareto_csv:
        return pd.read_csv(pareto_csv)
    
    return pd.DataFrame()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="OptiConn Sweep Explorer (Dash)")
    parser.add_argument('--output', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
                        help='Sweep output folder containing pareto_front.csv')
    parser.add_argument('--port', type=int, default=8050, help='Port to run Dash app on')
    args = parser.parse_args()

    df = load_data(args.output)
    
    # Build wave consistency summary and recommendation
    wave_summary = ""
    recommendation = ""
    if 'waves_present' in df.columns and 'pure_qa_score' in df.columns:
        total_waves = df['wave'].nunique()
        consistent_candidates = df[df['waves_present'] == total_waves]['candidate_key'].nunique()
        
        # Find best candidate: appears in all waves with highest avg QA
        best_candidates = df[df['waves_present'] == total_waves].copy()
        if not best_candidates.empty and 'avg_qa_across_waves' in best_candidates.columns:
            best_idx = best_candidates['avg_qa_across_waves'].idxmax()
            best = df.loc[best_idx]
            recommendation = f"🏆 **Recommended:** {best['atlas']} + {best['connectivity_metric']} " \
                           f"(QA: {best.get('avg_qa_across_waves', 0):.3f}, appears in {int(best['waves_present'])} waves)"
        
        wave_summary = f"📊 **Cross-Wave Analysis:** {total_waves} waves compared | " \
                      f"{consistent_candidates} candidates consistent across all waves"

    app = dash.Dash(__name__)
    app.title = "OptiConn Sweep Explorer"

    app.layout = html.Div([
        html.H1("OptiConn Sweep Explorer"),
        html.P("Interactive dashboard for exploring sweep parameter interactions and selecting optimal candidates."),
        dcc.Markdown(wave_summary) if wave_summary else html.Div(),
        dcc.Markdown(recommendation, style={'backgroundColor': '#d4edda', 'padding': '10px', 'borderRadius': '5px', 'marginTop': '10px'}) if recommendation else html.Div(),
        html.Hr(),
        html.H3("Visualization Controls"),
        dcc.Dropdown(
            id='x-axis',
            options=[{'label': col, 'value': col} for col in df.columns],
            value=df.columns[0] if not df.empty else None,
            placeholder='Select X axis parameter'
        ),
        dcc.Dropdown(
            id='y-axis',
            options=[{'label': col, 'value': col} for col in df.columns],
            value=df.columns[1] if not df.empty else None,
            placeholder='Select Y axis parameter'
        ),
        dcc.Graph(id='pareto-plot'),
        html.Hr(),
        html.H3("Candidate Selection Table"),
        html.P("📌 Click on a row to select it, then click the button below to save your choice."),
        dcc.Markdown("No data found." if df.empty else ""),
            dcc.Loading(
                [
                    DataTable(
                        id='pareto-table',
                        columns=[{"name": i, "id": i} for i in df.columns],
                        data=df.to_dict('records'),
                        page_size=20,
                        style_table={'overflowX': 'auto'},
                        row_selectable='single',
                        selected_rows=[],
                        sort_action='native',  # Enable native sorting
                        sort_mode='multi',      # Allow sorting by multiple columns
                        filter_action='native', # Enable filtering
                        style_cell={
                            'textAlign': 'left',
                            'padding': '10px',
                            'minWidth': '100px',
                        },
                        style_header={
                            'backgroundColor': '#f0f0f0',
                            'fontWeight': 'bold',
                            'border': '1px solid #ddd'
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': '#f9f9f9'
                            },
                            {
                                'if': {'state': 'selected'},
                                'backgroundColor': '#d4edda',
                                'border': '2px solid #28a745'
                            }
                        ]
                    ),
                    html.Button('Select This Candidate for Apply', id='export-btn', n_clicks=0, style={'marginTop': '20px', 'backgroundColor': '#4CAF50', 'color': 'white', 'padding': '10px 20px', 'fontSize': '16px', 'border': 'none', 'cursor': 'pointer'}),
                    html.Div(id='export-status', style={'marginTop': '10px', 'color': 'green', 'fontSize': '14px'})
                ] if not df.empty else html.Div()
            )
    ])


    @app.callback(
        Output('pareto-plot', 'figure'),
        Input('x-axis', 'value'),
        Input('y-axis', 'value')
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
        Output('export-status', 'children'),
        Input('export-btn', 'n_clicks'),
        State('pareto-table', 'selected_rows')
    )
    def export_selected(n_clicks, selected_rows):
        if n_clicks > 0 and selected_rows:
            # Since row_selectable='single', selected_rows will have only one element
            selected_idx = selected_rows[0]
            selected_candidate = df.iloc[selected_idx].to_dict()
            
            # Check if this is an optimal_combinations format (has atlas/metric)
            # If so, wrap it in a list for compatibility with apply command
            has_atlas = 'atlas' in selected_candidate and 'connectivity_metric' in selected_candidate
            
            out_path = os.path.join(args.output, 'selected_candidate.json')
            try:
                with open(out_path, 'w') as f:
                    # Save as list if it has atlas/metric (optimal_combinations format)
                    # This way apply command can process it correctly
                    if has_atlas:
                        json.dump([selected_candidate], f, indent=2)
                    else:
                        json.dump(selected_candidate, f, indent=2)
                        
                return f"✅ Selected candidate saved to selected_candidate.json. You can now run: opticonn apply --optimal-config {out_path} -i <your_data_dir> -o <output_dir>"
            except Exception as e:
                return f"❌ Export failed: {e}"
        elif n_clicks > 0:
            return "⚠️  Please select a candidate from the table above."
        return ""

    app.run(debug=True, port=args.port)
