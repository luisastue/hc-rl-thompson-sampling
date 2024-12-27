from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from custom_sepsis import n_states, n_actions


class Simplification(Enum):
    NONE = 0
    MEDIUM = 1
    SIMPLE = 2


class DirModel(ABC):
    def __init__(self, type: Simplification, state_counts):
        self.type = type
        self.state_counts = state_counts

    @abstractmethod
    def update_state_counts(self, episode, state_counts):
        pass

    @abstractmethod
    def transition_model(self, dirichlet_counts):
        pass

    @abstractmethod
    def get_state_counts(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def to_dict_counts(self, state_counts):
        pass

    @abstractmethod
    def from_dict_counts(state_counts):
        pass
