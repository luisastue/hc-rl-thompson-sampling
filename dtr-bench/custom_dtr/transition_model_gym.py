import gymnasium as gym
import numpy as np
import pickle
import os
import stable_baselines3 as sb3
import sepsis_gym as sgym
from sepsis_types import ACTIONS, STATES, state_to_index, random_initial_state

n_states = len(STATES)
n_actions = len(ACTIONS)


class TrModelEnv(gym.Env):
    def __init__(self, transition_model):
        super(TrModelEnv, self).__init__()
        self.transition_model = transition_model
        self.max_episode_length = 10
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_space = gym.spaces.Discrete(n_actions)
        self.observation_space = gym.spaces.Discrete(n_states)
        self.state = state_to_index[random_initial_state()]
        self.step_count = 0

    def reset(self, **kwargs):
        # Reset to initial state
        self.state = state_to_index[random_initial_state()]
        self.step_count = 0
        return self.state, {}  # Return initial observation and empty info dic

    def step(self, action):
        prev_state = STATES[self.state]
        transition_probs = self.transition_model[self.state, action]
        next_state = np.random.choice(self.n_states, p=transition_probs)
        reward = sgym.get_reward(STATES[next_state])
        done = reward != 0 or self.step_count >= self.max_episode_length
        self.step_count += 1
        self.state = next_state
        return next_state, reward, done, False, {"prev_state": prev_state, "action": ACTIONS[action], "state": STATES[next_state]}


def transition_model(dirichlet_counts=np.ones((n_states, n_actions, n_states))):
    model = np.zeros((n_states, n_actions, n_states))
    for i in range(n_states):
        for j in range(n_actions):
            counts = dirichlet_counts[i, j, :]
            transition_probs = np.random.dirichlet(counts)
            model[i, j, :] = transition_probs
    return model


def update_state_counts(episodes, state_counts=np.ones((n_states, n_actions, n_states))):
    for episode in episodes:
        for i, state in enumerate(episode.visited[:-1]):
            action = episode.policy[state_to_index[state]]
            next_state = episode.visited[i + 1]
            state_counts[state_to_index[state], action,
                         state_to_index[next_state]] += 1

    return state_counts


REWARDS = np.array([
    [sgym.get_reward(state) for _ in range(n_actions)]
    for state in STATES
])


def matrix_value_iteration(prev_V, transition_model, gamma=0.99, theta=1e-6):
    """
    Optimized Value Iteration using matrix operations.

    Args:
        transition_model (np.ndarray): A 3D array of shape (n_states, n_actions, n_states) representing 
                                       transition probabilities.
        gamma (float): Discount factor.
        theta (float): Convergence threshold.

    Returns:
        policy (np.ndarray): Optimal policy as a 1D array of shape (n_states,).
        V (np.ndarray): Optimal value function as a 1D array of shape (n_states,).
    """
    # Initialize value function
    V = prev_V

    while True:
        # Compute Q-values for all state-action pairs
        # Q[s, a] = R(s, a) + γ * Σ_s' P(s' | s, a) * V(s')
        Q = REWARDS + gamma * np.einsum('ijk,k->ij', transition_model, V)

        # Perform Bellman update
        new_V = np.max(Q, axis=1)  # Take the maximum value over actions
        delta = np.max(np.abs(new_V - V))  # Measure the largest change

        # Update value function
        V = new_V

        # Stop if converged
        if delta < theta:
            break

    # Derive policy: the action that maximizes Q-value for each state
    policy = np.argmax(Q, axis=1)

    return policy, V
