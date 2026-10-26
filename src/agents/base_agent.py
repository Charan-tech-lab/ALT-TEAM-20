"""Abstract Base Agent for Smart Building Energy Control."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseAgent(ABC):
    """Abstract interface defining required methods for all RL and baseline controllers."""

    def __init__(self, state_dim: int, action_dim: int, name: str = "BaseAgent"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = name

    @abstractmethod
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> Any:
        """Selects an action given the current observation."""
        pass

    def update(self, *args, **kwargs) -> dict[str, float]:
        """Performs a gradient or learning update step."""
        return {}

    def save(self, filepath: str) -> None:
        """Saves model weights / parameters."""
        pass

    def load(self, filepath: str) -> None:
        """Loads model weights / parameters."""
        pass
