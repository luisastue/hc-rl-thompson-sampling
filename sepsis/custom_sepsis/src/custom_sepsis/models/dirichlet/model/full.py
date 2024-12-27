import numpy as np
from ....sepsis_env import *
from .dirichlet_model import *


class FullModel(DirModel):
    def __init__(self, state_counts):
        if state_counts is None:
            state_counts = np.ones((n_states, n_actions, n_states))
        super().__init__(Simplification.NONE, state_counts)

    def update_state_counts(self, episode):
        for i, state in enumerate(episode.visited[:-1]):
            if i > 0 and episode.rewards[i-1] != 0:
                break
            action = episode.policy[state]
            next_state = episode.visited[i + 1]
            self.state_counts[state_to_index[state], action_to_index[action],
                              state_to_index[next_state]] += 1
        return self.state_counts

    def transition_model(self):
        model = np.zeros((n_states, n_actions, n_states))
        for i in range(n_states):
            for j in range(n_actions):
                counts = self.state_counts[i, j, :]
                transition_probs = np.random.dirichlet(counts)
                model[i, j, :] = transition_probs
        return model

    def get_state_counts(self):
        return self.state_counts.copy()
