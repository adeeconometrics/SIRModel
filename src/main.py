
from dash import Input, Output, State, Dash
from dash_bootstrap_components.themes import BOOTSTRAP

import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from src.layout import create_layout
from src.model import SIRModel
    
app = Dash(__name__, external_stylesheets=[BOOTSTRAP])


app.layout = create_layout()

model = SIRModel(width=20, height=20, population=1500,
                 infection_rate=0.1, recovery_rate=0.05)


@app.callback(
    [Output('sir-scatter', 'figure'),
     Output('population-graph', 'figure')],
    [Input('interval-component', 'n_intervals'),
     Input('infection-rate', 'value'),
     Input('recovery-rate', 'value'),
     Input('population-input', 'value')],
    [State('interval-component', 'n_intervals')]
)
def update_graph(_, infection_rate: float, 
                 recovery_rate: float, 
                 population: int, 
                 last_step: int):
    """Callback function for rendering response"""
    if last_step == 0:
        
        global model
        model = SIRModel(width=20, height=20, population=population,
                         infection_rate=infection_rate, recovery_rate=recovery_rate)

    model.step()

    # Get positions and states for scatter plot
    positions = np.array([agent.pos for agent in model.schedule.agents])
    states = [agent.state.name for agent in model.schedule.agents]

    # Scatter plot for agent positions
    scatter_fig = px.scatter(
        x=positions[:, 0], y=positions[:, 1], color=states,
        color_discrete_map={
            'SUSCEPTIBLE': 'blue',
            'INFECTED': 'red',
            'RECOVERED': 'green'
        },
        title='Agent Position (SIR)'
    )

    # Line graph for population counts
    population_counts = model.data_collector
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(
        y=population_counts["SUSCEPTIBLE"], mode='lines', name='Susceptible', line={'color': 'blue'}))
    line_fig.add_trace(go.Scatter(
        y=population_counts["INFECTED"], mode='lines', name='Infected', line={'color': 'red'}))
    line_fig.add_trace(go.Scatter(
        y=population_counts["RECOVERED"], mode='lines', name='Recovered', line={'color': 'green'}))
    line_fig.update_layout(title='SIR Population Over Time',
                           xaxis_title='Step', yaxis_title='Population')

    return scatter_fig, line_fig


if __name__ == '__main__':
    app.run_server(debug=True)
