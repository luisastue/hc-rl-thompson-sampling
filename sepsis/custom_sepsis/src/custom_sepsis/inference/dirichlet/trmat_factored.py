import numpy as np
from ...sepsis_env import *

n_states = len(STATES)
n_actions = len(ACTIONS)


def init_hr_counts():
    return {
        (hr, action, hr_next): 1
        for hr in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]
        for action in ACTIONS
        for hr_next in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]
    }


def init_bp_counts():
    return {
        (bp, action, bp_next): 1
        for bp in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]
        for action in ACTIONS
        for bp_next in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]
    }


def init_o2_counts():
    return {
        (o2, action, o2_next): 1
        for o2 in [Level.LOW.value, Level.NORMAL.value, Level.HIGH.value]
        for action in ACTIONS
        for o2_next in [Level.LOW.value, Level.NORMAL.value]
    }


def init_glu_counts():
    return {
        (glu, action, glu_next): 1
        for glu in [Level.SUPER_LOW.value, Level.LOW.value, Level.NORMAL.value, Level.HIGH.value, Level.SUPER_HIGH.value]
        for action in ACTIONS
        for glu_next in [Level.SUPER_LOW.value, Level.LOW.value, Level.NORMAL.value, Level.HIGH.value, Level.SUPER_HIGH.value]
    }


def update_factored_counts(episode, hr_counts, bp_counts, o2_counts, glu_counts):
    for i, state in enumerate(episode.visited[:-1]):
        action = episode.policy[state]
        next_state = episode.visited[i + 1]
        hr_counts[(state.hr, action, next_state.hr)] += 1
        bp_counts[(state.bp, action, next_state.bp)] += 1
        o2_counts[(state.o2, action, next_state.o2)] += 1
        glu_counts[(state.glu, action, next_state.glu)] += 1
    return (hr_counts, bp_counts, o2_counts, glu_counts)


def transition_model(hr_counts=init_hr_counts(), bp_counts=init_bp_counts(), o2_counts=init_o2_counts(), glu_counts=init_glu_counts()):
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
