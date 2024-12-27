import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
from ..sepsis_env import *


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
        reward = get_reward(STATES[next_state])
        done = reward != 0 or self.step_count >= self.max_episode_length
        self.step_count += 1
        self.state = next_state
        return next_state, reward, done, False, {"prev_state": prev_state, "action": ACTIONS[action], "state": STATES[next_state]}
