from enum import Enum


class HealthState(Enum):
    """Represents different Health States for the SIR model"""
    SUSCEPTIBLE = 1
    INFECTED = 2
    RECOVERED = 3
