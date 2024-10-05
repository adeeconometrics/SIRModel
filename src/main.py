from enum import Enum
import random

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from dash import Input, Output, Dash
from dash import dcc, html

import plotly.express as px
import numpy as np


class HealthState(Enum):
    SUSCEPTIBLE = 1
    INFECTED = 2
    RECOVERED = 3


class SIRAgent(Agent):
    def __init__(self, 
                 unique_id: int, 
                 model, 
                 initial_state: HealthState) -> None:
        super().__init__(unique_id, model)
        self.state = initial_state

    def step(self) -> None:
        """Each agent will try to infect its neighbors if it is infected."""
        if self.state == HealthState.INFECTED:
            # Try to infect neighbors
            neighbors = self.model.grid.get_neighbors(
                self.pos, moore=True, include_center=False)
            for neighbor in neighbors:
                if neighbor.state == HealthState.SUSCEPTIBLE:
                    # Probability-based infection
                    if random.random() < self.model.infection_rate:
                        neighbor.state = HealthState.INFECTED
            # Random recovery after being infected
            if random.random() < self.model.recovery_rate:
                self.state = HealthState.RECOVERED


class SIRModel(Model):
    def __init__(self, 
                 width: int, 
                 height: int, 
                 population: int, 
                 infection_rate: float, 
                 recovery_rate: float) -> None:
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate

        # Create agents
        for i in range(population):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            initial_state = HealthState.SUSCEPTIBLE
            if i == 0:  # Infect the first agent
                initial_state = HealthState.INFECTED
            agent = SIRAgent(i, self, initial_state)
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)

    def step(self) -> None:
        """Advance the model by one step."""
        self.schedule.step()

    def get_agent_states(self) -> list[HealthState]:
        """Return the current states of all agents for visualization or analysis."""
        return [agent.state for agent in self.schedule.agents]
    
app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='sir-graph'),
    dcc.Interval(id='interval-component', interval=100,
                 n_intervals=0)  # Update every second
])

model = SIRModel(width=20, height=20, population=100,
                 infection_rate=0.2, recovery_rate=0.05)

@app.callback(
    Output('sir-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n: int):
    # Run one step of the model
    model.step()

    # Extract agent positions and states
    positions = np.array([agent.pos for agent in model.schedule.agents])
    states = [agent.state.name for agent in model.schedule.agents]

    # Create scatter plot
    fig = px.scatter(
        x=positions[:, 0], y=positions[:, 1], color=states,
        color_discrete_map={
            'SUSCEPTIBLE': 'blue',
            'INFECTED': 'red',
            'RECOVERED': 'green'
        },
        title=f'SIR Model - Step {n}'
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)