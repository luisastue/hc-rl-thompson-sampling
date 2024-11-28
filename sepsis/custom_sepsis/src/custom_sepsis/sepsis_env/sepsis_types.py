from typing import List, Any, Optional
from collections import namedtuple
import random
from enum import Enum
import numpy as np

Action = namedtuple('Action', ['abx', 'vaso', 'vent'])

State = namedtuple('State', ['hr', 'bp', 'o2', 'glu',
                   'diabetic', 'abx', 'vaso', 'vent'])

EnvParameters = List[float]

param_names = [
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
]
param_to_index = {param: i for i, param in enumerate(param_names)}


class Level(Enum):
    SUPER_LOW = -2
    LOW = -1
    NORMAL = 0
    HIGH = 1
    SUPER_HIGH = 2


ACTIONS = [
    Action(abx, vaso, vent)
    for abx in [True, False]
    for vaso in [True, False]
    for vent in [True, False]
]

STATES = [
    State(hr, bp, o2, glu, diabetic, abx, vaso, vent)
    for hr in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]
    for bp in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]
    for o2 in [Level.LOW.value, Level.NORMAL.value]
    for glu in [Level.SUPER_LOW.value, Level.LOW.value, Level.NORMAL.value, Level.HIGH.value, Level.SUPER_HIGH.value]
    for diabetic in [True, False]
    for abx in [True, False]
    for vaso in [True, False]
    for vent in [True, False]
]

Policy = dict[State, Action]


class Episode:
    def __init__(self,
                 policy: Optional[Policy] = None,
                 rewards: List[float] = None,
                 visited: Optional[List[int]] = None):
        # Default to an empty list if None
        self.policy: Optional[Policy] = policy
        # Default to an empty list if None
        self.rewards: List[float] = rewards or []
        # Default to an empty list if None
        self.visited: Optional[List[State]] = visited or []
        self.date: str = str(np.datetime64('now'))


state_to_index = {state: i for i, state in enumerate(STATES)}
action_to_index = {action: i for i, action in enumerate(ACTIONS)}


def random_policy():
    return {state: random.choice(ACTIONS) for state in STATES}
