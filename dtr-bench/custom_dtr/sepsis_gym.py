import gymnasium as gym
import numpy as np
from sepsis_types import Action, Episode, State, STATES, ACTIONS, random_initial_state, state_to_index, action_to_index, Level, EnvParameters
from typing import List

TRUE_ENV_PARAMS = EnvParameters(
    0.5,
    0.5,
    0.1,
    0.5,
    0.7,
    0.1,
    0.7,
    0.7,
    0.5,
    0.4,
    0.9,
    0.5,
    0.1,
    0.1,
    0.05,
    0.05,
    0.1,
    0.3
)


def sample_from_uniform():
    return EnvParameters(
        np.random.beta(1, 1),
        np.random.beta(1, 1),
        np.random.beta(1, 1),
        np.random.beta(1, 1),
        np.random.beta(1, 1),
        np.random.beta(1, 1),
        np.random.beta(1, 1),
        np.random.beta(1, 1),
        np.random.beta(1, 1),
        np.random.beta(1, 1),
        np.random.beta(1, 1),
        np.random.beta(1, 1),
        np.random.beta(1, 1),
        np.random.beta(1, 1),
        np.random.beta(1, 1),
        np.random.beta(1, 1),
        np.random.beta(1, 1),
        np.random.beta(1, 1))


def get_next_state(env_params: EnvParameters, state: State, action: Action):
    hr = state.hr
    bp = state.bp
    o2 = state.o2
    glu = state.glu

    # Antibiotics -----------------------------
    if action.abx:
        if state.hr == Level.HIGH and np.random.rand() < 0.5:
            hr = Level.NORMAL
        if state.bp == Level.HIGH and np.random.rand() < 0.5:
            bp = Level.NORMAL
    elif action.abx == False and state.abx:  # withdrawn
        if state.hr == Level.NORMAL and np.random.rand() < 0.1:
            hr = Level.HIGH
        if state.bp == Level.NORMAL and np.random.rand() < 0.5:
            bp = Level.HIGH
    # Ventilation -----------------------------
    if action.vent:
        if state.o2 == Level.LOW and np.random.rand() < 0.7:
            o2 = Level.NORMAL
    elif action.vent == False and state.vent:  # withdrawn
        if state.o2 == Level.NORMAL and np.random.rand() < 0.1:
            o2 = Level.LOW
    # Vasopressors ----------------------------
    if state.diabetic == False:
        if action.vaso:
            # blood pressure ---------------------
            if state.bp == Level.LOW and np.random.rand() < 0.7:
                bp = Level.NORMAL
            if state.bp == Level.NORMAL and np.random.rand() < 0.7:
                bp = Level.HIGH
        elif action.vaso == False and state.vaso:  # withdrawn
            if state.bp == Level.NORMAL and np.random.rand() < 0.1:
                bp = Level.LOW
            if state.bp == Level.HIGH and np.random.rand() < 0.1:
                bp = Level.NORMAL
    elif state.diabetic == True:
        if action.vaso:
            # blood pressure ---------------------
            if state.bp == Level.LOW and np.random.rand() < 0.5:
                bp = Level.NORMAL
            if state.bp == Level.LOW and np.random.rand() < 0.4:
                bp = Level.HIGH
            if state.bp == Level.NORMAL and np.random.rand() < 0.9:
                bp = Level.HIGH
            # glucose -----------------------------
            if state.glu == Level.SUPER_LOW and np.random.rand() < 0.5:
                glu = Level.LOW
            if state.glu == Level.LOW and np.random.rand() < 0.5:
                glu = Level.NORMAL
            if state.glu == Level.NORMAL and np.random.rand() < 0.5:
                glu = Level.HIGH
            if state.glu == Level.HIGH and np.random.rand() < 0.5:
                glu = Level.SUPER_HIGH
        elif action.vaso == False and state.vaso:  # withdrawn
            if state.bp == Level.NORMAL and np.random.rand() < 0.05:
                bp = Level.LOW
            if state.bp == Level.HIGH and np.random.rand() < 0.05:
                bp = Level.NORMAL

    # Fluctuations ----------------------------
    # random fluctuations only if no change in treatment
    if state.abx == action.abx and np.random.rand() < 0.1:
        # heart rate is only affected by antibiotics
        hr = Level(
            max(min(state.hr.value + np.random.choice([-1, 1]), 1), -1))
    if state.abx == action.abx and state.vaso == action.vaso and np.random.rand() < 0.1:
        # heart rate is affected by antibiotics and vasopressors --> both need to stay the same for fluctuation
        bp = Level(
            max(min(state.bp.value + np.random.choice([-1, 1]), 1), -1))
    if state.vent == action.vent and np.random.rand() < 0.1:
        # oxygen is only affected by ventilation
        o2 = Level(
            max(min(state.o2.value + np.random.choice([-1, 1]), 0), -1))
    glu_prob = 0.3 if state.diabetic else 0.1
    if state.vaso == action.vaso and np.random.rand() < glu_prob:
        # glucose is only affected by vasopressors
        glu = Level(
            max(min(state.glu.value + np.random.choice([-1, 1]), 2), -2))

    return State(hr, bp, o2, glu, state.diabetic, action.abx, action.vaso, action.vent)


def get_reward(state: State):
    critical_counts = sum(
        1 for c in [state.hr, state.bp, state.o2, state.glu] if c != Level.NORMAL)
    if critical_counts >= 3:
        return -1
    elif critical_counts == 0 and not state.abx and not state.vaso and not state.vent:
        return 1
    return 0


class SepsisEnv(gym.Env):
    def __init__(self, parameters: EnvParameters):
        super(SepsisEnv, self).__init__()
        self.parameters = parameters
        # hard coded because it needs to be the same for every env, otherwise there will be mismatches
        self.max_episode_length = 10
        self.n_states = len(STATES)
        self.n_actions = len(ACTIONS)
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Discrete(self.n_states)
        # state is the index of the state
        self.state = state_to_index[random_initial_state()]
        self.step_count = 0

    def reset(self, **kwargs):
        # Reset to initial state
        self.state = state_to_index[random_initial_state()]
        self.step_count = 0
        return self.state, {}  # Return initial observation and empty info dic

    def step(self, action: int):
        action = ACTIONS[action]
        state = STATES[self.state]
        next_state = get_next_state(self.parameters, state, action)
        ix = state_to_index[next_state]
        reward = get_reward(next_state)
        self.state = ix
        self.step_count += 1
        done = reward != 0 or self.step_count >= self.max_episode_length
        return self.state, reward, done, False, {"step_count": self.step_count, "previous": state, "action": action, "next_state": next_state}


true_env = SepsisEnv(TRUE_ENV_PARAMS)


def run_episode(policy: List[int]):
    state_ix = true_env.reset()[0]
    state = STATES[state_ix]
    visited, rewards = [state], []
    terminated = False
    while not terminated:
        action = policy[state_ix]
        obs, reward, terminated, truncated, info = true_env.step(action)
        new_state = STATES[obs]
        visited.append(new_state)
        rewards.append(reward)
        state = new_state

    return Episode(policy, rewards, visited)
