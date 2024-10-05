from src.const import HealthState
from random import random, choice

from mesa import Agent


class SIRAgent(Agent):
    """Represents the SIR agent and contain its corresponding actions"""

    def __init__(self,
                 unique_id: int,
                 t_model,
                 initial_state: HealthState) -> None:
        super().__init__(unique_id, t_model)
        self.state = initial_state

    def move(self) -> None:
        """
        Move the agent to a random neighboring cell, only vertically or horizontally.
        Get neighbors in cardinal directions (N, S, E, W).
        """
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False)
        new_position = choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def step(self) -> None:
        """Each agent moves randomly and may infect neighbors if infected."""
        self.move()

        if self.state == HealthState.INFECTED:
            # Infect susceptible neighbors
            neighbors = self.model.grid.get_neighbors(
                self.pos, moore=True, include_center=False)
            for neighbor in neighbors:
                if neighbor.state == HealthState.SUSCEPTIBLE and random() < self.model.infection_rate:
                    neighbor.state = HealthState.INFECTED
            # Recover after some time with a certain probability
            if random() < self.model.recovery_rate:
                self.state = HealthState.RECOVERED
