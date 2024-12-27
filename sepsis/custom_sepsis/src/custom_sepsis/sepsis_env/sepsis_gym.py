import gymnasium as gym
import numpy as np
from .sepsis_types import *
from typing import List

TRUE_ENV_PARAMS = [
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
]


def random_initial_state():
    # returns a random initial state with all actions set to False
    state = State(
        random.randint(-1, 1),
        random.randint(-1, 1),
        random.randint(-1, 0),
        random.randint(-2, 2),
        random.choice([True, False]),
        False,
        False,
        False
    )
    while get_reward(state) != 0:
        state = State(
            random.randint(-1, 1),
            random.randint(-1, 1),
            random.randint(-1, 0),
            random.randint(-2, 2),
            random.choice([True, False]),
            False,
            False,
            False
        )
    return state


def sample_from_uniform():
    return [np.random.beta(1, 1) for _ in len(TRUE_ENV_PARAMS)]


def get_next_state(state: State, action: Action):
    env_params = TRUE_ENV_PARAMS
    hr = state.hr
    bp = state.bp
    o2 = state.o2
    glu = state.glu

    # Antibiotics -----------------------------
    if action.abx:
        if state.hr == Level.HIGH.value and np.random.rand() < env_params[param_to_index['abx_on_hr_H_N']]:
            hr = Level.NORMAL.value
        if state.bp == Level.HIGH.value and np.random.rand() < env_params[param_to_index['abx_on_bp_H_N']]:
            bp = Level.NORMAL.value
    elif action.abx == False and state.abx:  # withdrawn
        if state.hr == Level.NORMAL.value and np.random.rand() < env_params[param_to_index['abx_withdrawn_hr_N_H']]:
            hr = Level.HIGH.value
        if state.bp == Level.NORMAL.value and np.random.rand() < env_params[param_to_index['abx_withdrawn_bp_N_H']]:
            bp = Level.HIGH.value
    # Ventilation -----------------------------
    if action.vent:
        if state.o2 == Level.LOW.value and np.random.rand() < env_params[param_to_index['vent_on_o2_L_N']]:
            o2 = Level.NORMAL.value
    elif action.vent == False and state.vent:  # withdrawn
        if state.o2 == Level.NORMAL.value and np.random.rand() < env_params[param_to_index['vent_withdrawn_o2_N_L']]:
            o2 = Level.LOW.value
    # Vasopressors ----------------------------
    if state.diabetic == False:
        if action.vaso:
            # blood pressure ---------------------
            if state.bp == Level.LOW.value and np.random.rand() < env_params[param_to_index['nond_vaso_on_bp_L_N']]:
                bp = Level.NORMAL.value
            if state.bp == Level.NORMAL.value and np.random.rand() < env_params[param_to_index['nond_vaso_on_bp_N_H']]:
                bp = Level.HIGH.value
        elif action.vaso == False and state.vaso:  # withdrawn
            if state.bp == Level.NORMAL.value and np.random.rand() < env_params[param_to_index['nond_vaso_withdrawn_bp_N_L']]:
                bp = Level.LOW.value
            if state.bp == Level.HIGH.value and np.random.rand() < env_params[param_to_index['nond_vaso_withdrawn_bp_H_N']]:
                bp = Level.NORMAL.value
    elif state.diabetic == True:
        if action.vaso:
            # blood pressure ---------------------
            if state.bp == Level.LOW.value and np.random.rand() < env_params[param_to_index['diab_vaso_on_bp_L_N']]:
                bp = Level.NORMAL.value
            if state.bp == Level.LOW.value and np.random.rand() < env_params[param_to_index['diab_vaso_on_bp_L_H']]:
                bp = Level.HIGH.value
            if state.bp == Level.NORMAL.value and np.random.rand() < env_params[param_to_index['diab_vaso_on_bp_N_H']]:
                bp = Level.HIGH.value
            # glucose -----------------------------
            if state.glu == Level.SUPER_LOW.value and np.random.rand() < env_params[param_to_index['diab_vaso_on_glu_up']]:
                glu = Level.LOW.value
            if state.glu == Level.LOW.value and np.random.rand() < env_params[param_to_index['diab_vaso_on_glu_up']]:
                glu = Level.NORMAL.value
            if state.glu == Level.NORMAL.value and np.random.rand() < env_params[param_to_index['diab_vaso_on_glu_up']]:
                glu = Level.HIGH.value
            if state.glu == Level.HIGH.value and np.random.rand() < env_params[param_to_index['diab_vaso_on_glu_up']]:
                glu = Level.SUPER_HIGH.value
        elif action.vaso == False and state.vaso:  # withdrawn
            if state.bp == Level.NORMAL.value and np.random.rand() < env_params[param_to_index['diab_vaso_withdrawn_bp_N_L']]:
                bp = Level.LOW.value
            if state.bp == Level.HIGH.value and np.random.rand() < env_params[param_to_index['diab_vaso_withdrawn_bp_H_N']]:
                bp = Level.NORMAL.value

    # Fluctuations ----------------------------
    # random fluctuations only if no change in treatment
    if state.abx == action.abx and np.random.rand() < env_params[param_to_index['fluct']]:
        # heart rate is only affected by antibiotics
        hr = max(min(state.hr + np.random.choice([-1, 1]), 1), -1)
    if state.abx == action.abx and state.vaso == action.vaso and np.random.rand() < env_params[param_to_index['fluct']]:
        # heart rate is affected by antibiotics and vasopressors --> both need to stay the same for fluctuation
        bp = max(min(state.bp + np.random.choice([-1, 1]), 1), -1)
    if state.vent == action.vent and np.random.rand() < env_params[param_to_index['fluct']]:
        # oxygen is only affected by ventilation
        o2 = max(min(state.o2 + np.random.choice([-1, 1]), 0), -1)
    glu_prob = env_params[param_to_index['diab_fluct_glu']
                          ] if state.diabetic else env_params[param_to_index['fluct']]
    if state.vaso == action.vaso and np.random.rand() < glu_prob:
        # glucose is only affected by vasopressors
        glu = max(min(state.glu + np.random.choice([-1, 1]), 2), -2)

    return state_to_index[State(hr, bp, o2, glu, state.diabetic, action.abx, action.vaso, action.vent)]


def get_reward(state: State):
    critical_counts = sum(
        1 for c in [state.hr, state.bp, state.o2, state.glu] if c != Level.NORMAL.value)
    if critical_counts >= 3:
        return -1
    elif critical_counts == 0 and not state.abx and not state.vaso and not state.vent:
        return 1
    return 0


class SepsisEnv(gym.Env):
    def __init__(self, get_next_state=get_next_state):
        super(SepsisEnv, self).__init__()
        # hard coded because it needs to be the same for every env, otherwise there will be mismatches
        self.max_episode_length = 10
        self.get_next_state = get_next_state
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Discrete(self.n_states)
        # state is the index of the state
        self.state = state_to_index[random_initial_state()]
        self.step_count = 0
        self.done = False

    def reset(self, **kwargs):
        # Reset to initial state
        self.state = state_to_index[random_initial_state()]
        self.step_count = 0
        self.done = False
        return self.state, {}  # Return initial observation and empty info dic

    def step(self, action: int):
        action = ACTIONS[action]
        state = STATES[self.state]
        ix = state if self.done else self.get_next_state(state, action)
        reward = 0 if self.done else get_reward(STATES[ix])
        self.state = ix
        self.step_count += 1
        self.done = reward != 0 or (self.step_count >= self.max_episode_length)
        return self.state, reward, self.done, False, {"step_count": self.step_count, "previous": state, "action": action, "next_state": STATES[ix]}


true_env = SepsisEnv()


def run_episode(policy: Policy):
    obs = true_env.reset()[0]
    state = STATES[obs]
    visited, rewards = [state], []
    terminated = False
    while not terminated:
        action = action_to_index[policy[state]]
        obs, reward, terminated, truncated, info = true_env.step(action)
        state = STATES[obs]
        visited.append(state)
        rewards.append(reward)

    return Episode(policy, rewards, visited)


def evaluate_policy(policy: Policy, n_episodes: int):
    test_eps = [run_episode(policy) for _ in range(n_episodes)]
    return np.mean([np.sum(ep.rewards) for ep in test_eps])
