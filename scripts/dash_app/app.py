
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

def load_data(output_folder):
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

    app = dash.Dash(__name__)
    app.title = "OptiConn Sweep Explorer"

    app.layout = html.Div([
        html.H1("OptiConn Sweep Explorer"),
        html.P("Interactive dashboard for exploring sweep parameter interactions and Pareto fronts."),
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
        html.H3("Pareto Front Table"),
        dcc.Markdown("No data found." if df.empty else ""),
            dcc.Loading(
                [
                    DataTable(
                        id='pareto-table',
                        columns=[{"name": i, "id": i} for i in df.columns],
                        data=df.to_dict('records'),
                        page_size=10,
                        style_table={'overflowX': 'auto'},
                        row_selectable='multi',
                        selected_rows=[],
                    ),
                    html.Button('Export Selected Candidates', id='export-btn', n_clicks=0, style={'marginTop': '20px'}),
                    html.Div(id='export-status', style={'marginTop': '10px', 'color': 'green'})
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
            selected = df.iloc[selected_rows].to_dict('records')
            out_path = os.path.join(args.output, 'selected_candidates.json')
            try:
                with open(out_path, 'w') as f:
                    json.dump(selected, f, indent=2)
                return f"Exported {len(selected)} candidates to selected_candidates.json."
            except Exception as e:
                return f"Export failed: {e}"
        elif n_clicks > 0:
            return "No candidates selected."
        return ""

    app.run(debug=True, port=args.port)
