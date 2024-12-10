import numpy as np
from ...sepsis_env import *

n_states = len(STATES)
n_actions = len(ACTIONS)


def update_full_counts(episode, state_counts=np.ones((n_states, n_actions, n_states))):
    for i, state in enumerate(episode.visited[:-1]):
        action = episode.policy[state]
        next_state = episode.visited[i + 1]
        state_counts[state_to_index[state], action_to_index[action],
                     state_to_index[next_state]] += 1
    return state_counts


def transition_model(dirichlet_counts=np.ones((n_states, n_actions, n_states))):
    model = np.zeros((n_states, n_actions, n_states))
    for i in range(n_states):
        for j in range(n_actions):
            counts = dirichlet_counts[i, j, :]
            transition_probs = np.random.dirichlet(counts)
            model[i, j, :] = transition_probs
    return model
