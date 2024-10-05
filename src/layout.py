from dash import dcc
from dash.html import Div, H1, Label
from dash_bootstrap_components import Container, Row, Col

__all__ = [
    "create_layout"
]

infection_rate_slider = Row([
    Col(Label('Infection Rate:'), width=3),
    Col(dcc.Slider(id='infection-rate', min=0, max=1, step=0.01,
                    value=0.1, marks={i/10: f"{i/10}" for i in range(11)}), width=9)
], className="mb-3")

recovery_rate_slider = Row([
    Col(Label('Recovery Rate:'), width=3),
    Col(dcc.Slider(id='recovery-rate', min=0, max=1, step=0.01,
                    value=0.05, marks={i/10: f"{i/10}" for i in range(11)}), width=9)
], className="mb-3")

population_input = Row([
    Col(Label('Population:'), width=3),
    Col(dcc.Input(id='population-input', type='number',
                    value=100, min=10, max=500, className="form-control"), width=9)
], className="mb-3")

sir_scatterplot = Row([
    Col(dcc.Graph(id='sir-scatter'), width=12)
], className="mb-4")

population_graph = Row([
    Col(dcc.Graph(id='population-graph'), width=12)
])

def create_layout() -> Div:
    """Generates the application layout"""
    return Div([
        Container([
            H1("SIR Epidemic Simulation", className="text-center mb-4"),
            infection_rate_slider,
            recovery_rate_slider,
            population_input,
            sir_scatterplot,
            population_graph,
            dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
        ])
    ])
