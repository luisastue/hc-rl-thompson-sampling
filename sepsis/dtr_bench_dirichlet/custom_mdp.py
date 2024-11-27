import gymnasium as gym
import matrix_value_iter as vi
import evaluate_algos as ev
import numpy as np
import pickle
import os
import stable_baselines3 as sb3

# Define the CustomMDPEnv


class CustomMDPEnv(gym.Env):
    def __init__(self, n_states, n_actions, transition_model):
        super(CustomMDPEnv, self).__init__()
        self.transition_model = transition_model
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_space = gym.spaces.Discrete(n_actions)
        self.observation_space = gym.spaces.Discrete(n_states)
        self.state = np.random.randint(n_states)

    def reset(self, **kwargs):
        # Reset to initial state
        self.state = np.random.randint(n_states)
        return self.state, {}  # Return initial observation and empty info dic

    def step(self, action):
        transition_probs = self.transition_model[self.state, action]
        next_state = np.random.choice(self.n_states, p=transition_probs)
        reward = vi.get_reward(next_state)
        done = reward != 0
        self.state = next_state
        return next_state, reward, done, False, {}


# Initialize environment with required parameters
n_states = len(vi.STATES)
n_actions = len(vi.ACTIONS)


class MDPOpt:
    def __init__(self, model, nr_iterations: int, tr_mod, episodes: list, name: str, info: dict):
        self.model = model
        self.nr_iterations = nr_iterations
        self.transition_model = tr_mod
        self.episodes = episodes
        self.info = info
        self.info["name"] = name
        self.info["date"] = str(np.datetime64('now'))
        self.policy = None

    def save(self, directory="data/mdp_opts"):
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(
            directory, f"{self.info['name']}_{self.info['date']}_model.zip")
        object_path = os.path.join(
            directory, f"{self.info['name']}_{self.info['date']}_object.pkl")

        self.model.save(model_path)
        self.info['model_path'] = model_path
        mod = self.model
        self.model = None

        with open(object_path, 'wb') as file:
            pickle.dump(self, file)

        self.model = mod
        return object_path, model_path

    def get_policy(self):
        if self.policy is None:
            self.policy = []
            for state_ix in range(n_states):
                action = self.model.predict(state_ix, deterministic=True)[0]
                self.policy.append(action)
        return self.policy

    def evaluate_policy(self, n_episodes: int):
        pol = self.get_policy()
        test_eps = [vi.run_episode(pol) for _ in range(n_episodes)]
        return np.mean([np.sum(ep.rewards) for ep in test_eps])

    @staticmethod
    def load(object_path: str):
        with open(object_path, 'rb') as file:
            mdp_opt = pickle.load(file)

        if 'model_path' in mdp_opt.info:
            mdp_opt.model = sb3.DQN.load(mdp_opt.info['model_path'])
        else:
            raise ValueError(
                "Model path not found in the saved MDPOpt object.")

        return mdp_opt


class CustomLoggingCallback(sb3.common.callbacks.BaseCallback):
    def __init__(self, verbose=0):
        super(CustomLoggingCallback, self).__init__(verbose)
        self.episodes = []
        self.evaluation = None
        self.current_rewards = []

    def _on_step(self) -> bool:
        # Collect rewards at each step
        reward = self.locals["rewards"][0]
        self.current_rewards.append(reward)

        # Check if the episode has ended
        done = self.locals["dones"]
        if done:
            self.episodes.append(ev.Episode(None, self.current_rewards, None))
            self.current_rewards = []  # Reset for the next episode

        return True  # Continue training


def train_new_custom_mdp(tr, nr_iter, name):
    env = CustomMDPEnv(n_states, n_actions, tr)
    model = sb3.DQN("MlpPolicy", env, verbose=0)
    reward_callback = CustomLoggingCallback()
    model.learn(nr_iter, callback=reward_callback)
    mdp_opt = MDPOpt(model, nr_iter, tr,
                     reward_callback.episodes, f"custom-dqn-{name}", {})
    mdp_opt.save()
    return mdp_opt
