import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
from ..sepsis_env import *

REWARDS = np.array([
    [get_reward(state) for _ in range(n_actions)]
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
    Q = None

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
    pol = np.argmax(Q, axis=1)
    policy = {STATES[i]: ACTIONS[action] for i, action in enumerate(pol)}

    return policy, V
