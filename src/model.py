from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid

from src.agent import SIRAgent
from src.const import HealthState


class SIRModel(Model):
    """Represents the SIR Model generation"""
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
        """
        Advance the model by one step and record population data by 
        counting the number of agents in each state.
        """
        self.schedule.step()

        susceptible = infected = recovered = 0
        for agent in self.schedule.agents:
            if agent.state == HealthState.SUSCEPTIBLE:
                susceptible += 1
            elif agent.state == HealthState.INFECTED:
                infected += 1
            elif agent.state == HealthState.RECOVERED:
                recovered += 1

        self.data_collector["SUSCEPTIBLE"].append(susceptible)
        self.data_collector["INFECTED"].append(infected)
        self.data_collector["RECOVERED"].append(recovered)
