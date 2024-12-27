import numpy as np
from ...sepsis_env import *
from .dirichlet_model import *

n_states = len(STATES)
n_actions = len(ACTIONS)


def init_hr_counts():
    return {
        (hr, action, hr_next): 1
        for hr in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]
        for action in ACTIONS
        for hr_next in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]
    }


def hr_to_list(hr_counts):
    return [hr_counts[(hr, action, hr_next)]
            for hr in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]
            for action in ACTIONS
            for hr_next in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]
            ]


def list_to_hr(hr_list):
    hr_counts = {}
    i = 0
    for hr in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]:
        for action in ACTIONS:
            for hr_next in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]:
                hr_counts[(hr, action, hr_next)] = hr_list[i]
                i += 1
    return hr_counts


def init_bp_counts():
    return {
        (bp, action, bp_next): 1
        for bp in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]
        for action in ACTIONS
        for bp_next in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]
    }


def bp_to_list(bp_counts):
    return [bp_counts[(bp, action, bp_next)]
            for bp in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]
            for action in ACTIONS
            for bp_next in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]
            ]


def list_to_bp(bp_list):
    bp_counts = {}
    i = 0
    for bp in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]:
        for action in ACTIONS:
            for bp_next in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]:
                bp_counts[(bp, action, bp_next)] = bp_list[i]
                i += 1
    return bp_counts


def init_o2_counts():
    return {
        (o2, action, o2_next): 1
        for o2 in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]
        for action in ACTIONS
        for o2_next in [Level.LOW.value, Level.NORMAL.value]
    }


def o2_to_list(o2_counts):
    return [o2_counts[(o2, action, o2_next)]
            for o2 in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]
            for action in ACTIONS
            for o2_next in [Level.LOW.value, Level.NORMAL.value]
            ]


def list_to_o2(o2_list):
    o2_counts = {}
    i = 0
    for o2 in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]:
        for action in ACTIONS:
            for o2_next in [Level.LOW.value, Level.NORMAL.value]:
                o2_counts[(o2, action, o2_next)] = o2_list[i]
                i += 1
    return o2_counts


def init_glu_counts():
    return {
        (glu, action, glu_next): 1
        for glu in [Level.SUPER_LOW.value, Level.LOW.value, Level.NORMAL.value, Level.HIGH.value, Level.SUPER_HIGH.value]
        for action in ACTIONS
        for glu_next in [Level.SUPER_LOW.value, Level.LOW.value, Level.NORMAL.value, Level.HIGH.value, Level.SUPER_HIGH.value]
    }


def glu_to_list(glu_counts):
    return [glu_counts[(glu, action, glu_next)]
            for glu in [Level.SUPER_LOW.value, Level.LOW.value, Level.NORMAL.value, Level.HIGH.value, Level.SUPER_HIGH.value]
            for action in ACTIONS
            for glu_next in [Level.SUPER_LOW.value, Level.LOW.value, Level.NORMAL.value, Level.HIGH.value, Level.SUPER_HIGH.value]
            ]


def list_to_glu(glu_list):
    glu_counts = {}
    i = 0
    for glu in [Level.SUPER_LOW.value, Level.LOW.value, Level.NORMAL.value, Level.HIGH.value, Level.SUPER_HIGH.value]:
        for action in ACTIONS:
            for glu_next in [Level.SUPER_LOW.value, Level.LOW.value, Level.NORMAL.value, Level.HIGH.value, Level.SUPER_HIGH.value]:
                glu_counts[(glu, action, glu_next)] = glu_list[i]
                i += 1
    return glu_counts


class SimpleModel(DirModel):
    def __init__(self, state_counts=None):
        if state_counts is None:
            state_counts = init_hr_counts(), init_bp_counts(
            ), init_o2_counts(), init_glu_counts()
        super().__init__(Simplification.SIMPLE, state_counts)

    def update_state_counts(self, episode):
        for i, state in enumerate(episode.visited[:-1]):
            if i > 0 and episode.rewards[i-1] != 0:
                break
            action = episode.policy[state]
            next_state = episode.visited[i + 1]
            hr_counts, bp_counts, o2_counts, glu_counts = self.state_counts
            hr_counts[(state.hr, action, next_state.hr)] += 1
            bp_counts[(state.bp, action, next_state.bp)] += 1
            o2_counts[(state.o2, action, next_state.o2)] += 1
            glu_counts[(state.glu, action, next_state.glu)] += 1
            self.state_counts = (hr_counts, bp_counts, o2_counts, glu_counts)
        return self.state_counts

    def transition_model(self):
        hr_counts, bp_counts, o2_counts, glu_counts = self.state_counts
        model = np.zeros((n_states, n_actions, n_states))
        for i, state in enumerate(STATES):
            for j, action in enumerate(ACTIONS):
                hr = hr_counts[(state.hr, action, Level.LOW.value)], hr_counts[(
                    state.hr, action, Level.NORMAL.value)], hr_counts[(state.hr, action, Level.HIGH.value)]
                p = np.random.dirichlet(hr)
                hr_probs = {
                    Level.LOW.value: p[0],
                    Level.NORMAL.value: p[1],
                    Level.HIGH.value: p[2]
                }
                bp = bp_counts[(state.bp, action, Level.LOW.value)], bp_counts[(
                    state.bp, action, Level.NORMAL.value)], bp_counts[(state.bp, action, Level.HIGH.value)]
                p = np.random.dirichlet(bp)
                bp_probs = {
                    Level.LOW.value: p[0],
                    Level.NORMAL.value: p[1],
                    Level.HIGH.value: p[2]
                }
                o2 = o2_counts[(state.o2, action, Level.LOW.value)
                               ], o2_counts[(state.o2, action, Level.NORMAL.value)]
                p = np.random.dirichlet(o2)
                o2_probs = {
                    Level.LOW.value: p[0],
                    Level.NORMAL.value: p[1]
                }
                glu = glu_counts[(state.glu, action, Level.SUPER_LOW.value)], glu_counts[(state.glu, action, Level.LOW.value)], glu_counts[(
                    state.glu, action, Level.NORMAL.value)], glu_counts[(state.glu, action, Level.HIGH.value)], glu_counts[(state.glu, action, Level.SUPER_HIGH.value)]
                p = np.random.dirichlet(glu)
                glu_probs = {
                    Level.SUPER_LOW.value: p[0],
                    Level.LOW.value: p[1],
                    Level.NORMAL.value: p[2],
                    Level.HIGH.value: p[3],
                    Level.SUPER_HIGH.value: p[4]
                }
                diab_probs = {True: 1 if state.diabetic else 0,
                              False: 1 if not state.diabetic else 0}
                abx_probs = {True: 1 if action.abx else 0,
                             False: 1 if not action.abx else 0}
                vaso_probs = {True: 1 if action.vaso else 0,
                              False: 1 if not action.vaso else 0}
                vent_probs = {True: 1 if action.vent else 0,
                              False: 1 if not action.vent else 0}

                for k, next_state in enumerate(STATES):
                    model[i, j, k] = hr_probs[next_state.hr] * \
                        bp_probs[next_state.bp] * \
                        o2_probs[next_state.o2] * \
                        glu_probs[next_state.glu] * \
                        diab_probs[next_state.diabetic] * \
                        abx_probs[next_state.abx] * \
                        vaso_probs[next_state.vaso] * \
                        vent_probs[next_state.vent]

        return model

    def get_state_counts(self):
        return tuple(en.copy() for en in self.state_counts)

    def to_dict(self):
        return {
            "type": Simplification.SIMPLE.value,
            "state_counts": self.to_dict_counts(self.state_counts)
        }

    def to_dict_counts(self, state_counts):
        hr_counts, bp_counts, o2_counts, glu_counts = state_counts
        return {
            "hr_counts": hr_to_list(hr_counts),
            "bp_counts": bp_to_list(bp_counts),
            "o2_counts": o2_to_list(o2_counts),
            "glu_counts": glu_to_list(glu_counts)
        }

    @staticmethod
    def from_dict_counts(state_counts):
        hr_counts = list_to_hr(state_counts["hr_counts"])
        bp_counts = list_to_bp(state_counts["bp_counts"])
        o2_counts = list_to_o2(state_counts["o2_counts"])
        glu_counts = list_to_glu(state_counts["glu_counts"])
        return hr_counts, bp_counts, o2_counts, glu_counts
