from typing import List, Any, Optional
from collections import namedtuple
import random
from enum import Enum
import numpy as np

Action = namedtuple('Action', ['abx', 'vaso', 'vent'])

State = namedtuple('State', ['hr', 'bp', 'o2', 'glu',
                   'diabetic', 'abx', 'vaso', 'vent'])


EnvParameters = namedtuple('EnvParameters', [
    'abx_on_hr_H_N',
    'abx_on_bp_H_N',
    'abx_withdrawn_hr_N_H',
    'abx_withdrawn_bp_N_H',
    'vent_on_o2_L_N',
    'vent_withdrawn_o2_N_L',
    'nond_vaso_on_bp_L_N',
    'nond_vaso_on_bp_N_H',
    'diab_vaso_on_bp_L_N',
    'diab_vaso_on_bp_L_H',
    'diab_vaso_on_bp_N_H',
    'diab_vaso_on_glu_up',
    'nond_vaso_withdrawn_bp_N_L',
    'nond_vaso_withdrawn_bp_H_N',
    'diab_vaso_withdrawn_bp_N_L',
    'diab_vaso_withdrawn_bp_H_N',
    'fluct',
    'diab_fluct_glu'
])


class Level(Enum):
    LOW = -1
    NORMAL = 0
    HIGH = 1
    SUPER_LOW = -2
    SUPER_HIGH = 2


ACTIONS = [
    Action(abx, vaso, vent)
    for abx in [True, False]
    for vaso in [True, False]
    for vent in [True, False]
]

STATES = [
    State(hr, bp, o2, glu, diabetic, abx, vaso, vent)
    for hr in [Level.LOW, Level.NORMAL, Level.HIGH]
    for bp in [Level.LOW, Level.NORMAL, Level.HIGH]
    for o2 in [Level.LOW, Level.NORMAL]
    for glu in [Level.SUPER_LOW, Level.LOW, Level.NORMAL, Level.HIGH, Level.SUPER_HIGH]
    for diabetic in [True, False]
    for abx in [True, False]
    for vaso in [True, False]
    for vent in [True, False]
]


class Episode:
    def __init__(self,
                 policy: Optional[List[int]] = None,
                 rewards: List[float] = None,
                 visited: Optional[List[int]] = None):
        # Default to an empty list if None
        self.policy: Optional[List[int]] = policy
        # Default to an empty list if None
        self.rewards: List[float] = rewards or []
        # Default to an empty list if None
        self.visited: Optional[List[int]] = visited or []
        self.date: str = str(np.datetime64('now'))


state_to_index = {state: i for i, state in enumerate(STATES)}
action_to_index = {action: i for i, action in enumerate(ACTIONS)}

Policy = List[int]  # for each state, index of the action to take


def random_policy():
    return [action_to_index[random.choice(ACTIONS)] for _ in STATES]


def random_initial_state():
    # returns a random initial state with all actions set to False

    return State(
        Level(random.randint(-1, 1)),
        Level(random.randint(-1, 1)),
        Level(random.randint(-1, 0)),
        Level(random.randint(-2, 2)),
        random.choice([True, False]),
        False,
        False,
        False
    )
