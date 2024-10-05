from enum import Enum
import random

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from dash import Input, Output, State, Dash
from dash import dcc, html

import plotly.express as px
import plotly.graph_objects as go
import numpy as np


class HealthState(Enum):
    SUSCEPTIBLE = 1
    INFECTED = 2
    RECOVERED = 3


class SIRAgent(Agent):
    def __init__(self, unique_id: int, model, initial_state: HealthState) -> None:
        super().__init__(unique_id, model)
        self.state = initial_state


    def move(self) -> None:
        """Move the agent to a random neighboring cell, only vertically or horizontally."""
        # Get neighbors in cardinal directions (N, S, E, W)
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)


    def step(self) -> None:
        """Each agent moves randomly and may infect neighbors if infected."""
        self.move()  # Move randomly

        if self.state == HealthState.INFECTED:
            # Infect susceptible neighbors
            neighbors = self.model.grid.get_neighbors(
                self.pos, moore=True, include_center=False)
            for neighbor in neighbors:
                if neighbor.state == HealthState.SUSCEPTIBLE and random.random() < self.model.infection_rate:
                    neighbor.state = HealthState.INFECTED
            # Recover after some time with a certain probability
            if random.random() < self.model.recovery_rate:
                self.state = HealthState.RECOVERED


class SIRModel(Model):
    def __init__(self, width: int, height: int, population: int, infection_rate: float, recovery_rate: float) -> None:
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.population = population

        # Initialize agent populations
        for i in range(self.population):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            initial_state = HealthState.SUSCEPTIBLE
            if i == 0:  # Start with one infected agent
                initial_state = HealthState.INFECTED
            agent = SIRAgent(i, self, initial_state)
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)

        # Track population counts over time
        self.data_collector: dict[str, list[int]] = {
            "SUSCEPTIBLE": [],
            "INFECTED": [],
            "RECOVERED": []
        }

    def step(self) -> None:
        """Advance the model by one step and record population data."""
        self.schedule.step()

        # Count the number of agents in each state
        susceptible = infected = recovered = 0
        for agent in self.schedule.agents:
            if agent.state == HealthState.SUSCEPTIBLE:
                susceptible += 1
            elif agent.state == HealthState.INFECTED:
                infected += 1
            elif agent.state == HealthState.RECOVERED:
                recovered += 1

        # Record counts
        self.data_collector["SUSCEPTIBLE"].append(susceptible)
        self.data_collector["INFECTED"].append(infected)
        self.data_collector["RECOVERED"].append(recovered)

    def get_population_counts(self) -> dict[str, int]:
        """Return the current counts of each population state."""
        return {
            "SUSCEPTIBLE": len([agent for agent in self.schedule.agents if agent.state == HealthState.SUSCEPTIBLE]),
            "INFECTED": len([agent for agent in self.schedule.agents if agent.state == HealthState.INFECTED]),
            "RECOVERED": len([agent for agent in self.schedule.agents if agent.state == HealthState.RECOVERED])
        }

    
app = Dash(__name__)

app.layout = html.Div([
    html.H1("SIR Epidemic Simulation"),

    # Control buttons for adjusting parameters
    html.Label('Infection Rate:'),
    dcc.Slider(id='infection-rate', min=0, max=1, step=0.01,
               value=0.1, marks={i/10: f"{i/10}" for i in range(11)}),

    html.Label('Recovery Rate:'),
    dcc.Slider(id='recovery-rate', min=0, max=1, step=0.01,
               value=0.05, marks={i/10: f"{i/10}" for i in range(11)}),

    html.Label('Population:'),
    dcc.Input(id='population-input', type='number',
              value=100, min=10, max=500),

    # Real-time scatter plot for agent positions
    dcc.Graph(id='sir-scatter'),

    # Real-time population graph (S, I, R)
    dcc.Graph(id='population-graph'),

    # Interval component to update the simulation
    dcc.Interval(id='interval-component', interval=500, n_intervals=0)
])

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
def update_graph(n_intervals: int, infection_rate: float, recovery_rate: float, population: int, last_step: int):
    if last_step == 0:
        # Initialize the model with new parameters
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
        y=population_counts["SUSCEPTIBLE"], mode='lines', name='Susceptible', line=dict(color='blue')))
    line_fig.add_trace(go.Scatter(
        y=population_counts["INFECTED"], mode='lines', name='Infected', line=dict(color='red')))
    line_fig.add_trace(go.Scatter(
        y=population_counts["RECOVERED"], mode='lines', name='Recovered', line=dict(color='green')))
    line_fig.update_layout(title='SIR Population Over Time',
                           xaxis_title='Step', yaxis_title='Population')

    return scatter_fig, line_fig


if __name__ == '__main__':
    app.run_server(debug=True)