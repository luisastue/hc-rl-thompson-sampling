from typing import List, Any, Optional
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
from collections import namedtuple
import numpy as np
import gymnasium as gym
import DTRGym
import pickle
from collections import defaultdict


# Define the State structure
State = namedtuple('State', ['hr', 'bp', 'o2', 'glu',
                   'diabetic', 'abx', 'vaso', 'vent'])

# Define constants
ACTIONS = list(range(8))
STATES = [
    State(hr, bp, o2, glu / 2, diabetic, abx, vaso, vent)
    for hr in range(-1, 2)
    for bp in range(-1, 2)
    for o2 in range(-1, 2)
    for glu in range(-2, 3)
    for diabetic in [True, False]
    for abx in [True, False]
    for vaso in [True, False]
    for vent in [True, False]
]

# Define the get_reward function


def get_reward(state_ix):
    state = STATES[state_ix]
    critical_counts = sum(
        1 for c in [state.hr, state.bp, state.o2, state.glu] if c != 0)
    if critical_counts >= 3:
        return -1
    elif critical_counts == 0 and not state.abx and not state.vaso and not state.vent:
        return 1
    return 0

# Convert dictionary to state


def to_state(info):
    return State(
        info["hr_state"],
        info["sysbp_state"],
        info["percoxyg_state"],
        info["glucose_state"],
        info["diabetic_idx"],
        info["antibiotic_state"],
        info["vaso_state"],
        info["vent_state"]
    )

# Generate a random policy


def random_policy():
    return [random.choice(ACTIONS) for state in STATES]

# Define the Episode class


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


n_states = len(STATES)
n_actions = len(ACTIONS)

REWARDS = np.array([
    [get_reward(state) for _ in range(n_actions)]
    for state in range(n_states)
])
state_to_index = {state: i for i, state in enumerate(STATES)}

sepsis_env = gym.make("OberstSepsisEnv-discrete")


def run_episode(policy: List[int], max_length=200):
    state_dict = sepsis_env.reset()[1]
    state = to_state(state_dict["state"])
    visited, rewards = [state], []
    for _ in range(max_length):
        action = policy[state_to_index[state]]
        obs, reward, terminated, truncated, info = sepsis_env.step(action)
        new_state = to_state(info["state"])
        visited.append(new_state)
        rewards.append(reward)
        state = new_state
        if terminated:
            break
    return Episode(policy, rewards, visited)


def transition_model(dirichlet_counts=np.ones((n_states, n_actions, n_states))):
    model = np.zeros((n_states, n_actions, n_states))
    for i in range(n_states):
        for j in range(n_actions):
            counts = dirichlet_counts[i, j, :]
            transition_probs = np.random.dirichlet(counts)
            model[i, j, :] = transition_probs
    return model


def matrix_value_iteration(prev_V, transition_model, rewards, n_states, n_actions, gamma=0.99, theta=1e-6):
    """
    Optimized Value Iteration using matrix operations.

    Args:
        transition_model (np.ndarray): A 3D array of shape (n_states, n_actions, n_states) representing 
                                       transition probabilities.
        rewards (np.ndarray): A 2D array of shape (n_states, n_actions) containing rewards for each state-action pair.
        n_states (int): Number of states in the environment.
        n_actions (int): Number of actions in the environment.
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
        Q = rewards + gamma * np.einsum('ijk,k->ij', transition_model, V)

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


def update_state_counts(episodes, state_counts):
    for episode in episodes:
        for i, state in enumerate(episode.visited[:-1]):
            action = episode.policy[state_to_index[state]]
            next_state = episode.visited[i + 1]
            state_counts[state_to_index[state], action,
                         state_to_index[next_state]] += 1

    return state_counts
