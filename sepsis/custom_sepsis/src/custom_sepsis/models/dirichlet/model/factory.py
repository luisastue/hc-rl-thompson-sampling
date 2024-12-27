

from .full import FullModel
from .medium import MediumModel
from .simple import SimpleModel
from .dirichlet_model import Simplification


def create_model(type: Simplification, state_counts=None):
    if type == Simplification.NONE:
        return FullModel(state_counts)
    elif type == Simplification.MEDIUM:
        return MediumModel(state_counts)
    elif type == Simplification.SIMPLE:
        return SimpleModel(state_counts)
    else:
        raise ValueError("Invalid simplification type")
