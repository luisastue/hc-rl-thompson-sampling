from ..inference import Policy, DirModel, FullModel, compress_array, decompress_array, n_states, n_actions, compress_policy, decompress_policy
import os
import dill as pickle
import stable_baselines3 as sb3
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from .evaluate import Evaluation, evaluate_policy
from custom_sepsis import Episode, Policy
import gzip


class ThompsonSampling(Evaluation):
    def __init__(self, nr_iterations: int, episodes: List[Episode], name: str, info: dict):
        super().__init__(nr_iterations, episodes, name, info)
        self.policy = None

    def save(self, directory="data/thompson_sampling"):
        os.makedirs(directory, exist_ok=True)
        object_path = os.path.join(
            directory, f"{self.name}.pkl.gz"
        )

        with gzip.open(object_path, 'wb') as file:
            pickle.dump(self, file)

        return object_path

    @staticmethod
    def load(object_path: str):
        with gzip.open("data.pkl.gz", "rb") as f:
            data = pickle.load(f)

    @staticmethod
    def load_pickle(object_path: str):
        with open(object_path, 'rb') as file:
            return pickle.load(file)

    def get_policy(self):
        if self.policy is None:
            self.policy = self.episodes[len(self.episodes) - 1].policy
        return self.policy


class DirThompsonSampling():
    def __init__(self, model: DirModel, rewards: dict[int, float], state_counts: dict[int, any], policies: dict[int, Policy], name: str, info: dict):
        self.name = name
        self.info = info
        self.info["name"] = name
        self.info["date"] = str(np.datetime64('now'))
        self.model = model
        self.policies = {i: compress_policy(policy) for i,
                         policy in policies.items()}
        self.state_counts = state_counts
        self.mean_rewards = {}
        self.rewards = rewards

    def save(self, directory="data/dir_ts"):
        os.makedirs(directory, exist_ok=True)
        object_path = os.path.join(
            directory, f"{self.name}.pkl.gz"
        )

        with gzip.open(object_path, 'wb') as file:
            pickle.dump(self, file)

        return object_path

    @staticmethod
    def load(object_path: str):
        with gzip.open(object_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_pickle(object_path: str):
        with open(object_path, 'rb') as file:
            return pickle.load(file)

    def get_mean_rewards(self, nr_eval_episodes: int):
        if nr_eval_episodes not in self.mean_rewards:
            self.mean_rewards[nr_eval_episodes] = [evaluate_policy(decompress_policy(policy), nr_eval_episodes)
                                                   for policy in list(self.policies.values())]
        if len(self.mean_rewards[nr_eval_episodes]) != len(list(self.policies.keys())):
            start = len(self.mean_rewards[nr_eval_episodes])
            self.mean_rewards[nr_eval_episodes].extend([evaluate_policy(decompress_policy(policy), nr_eval_episodes)
                                                        for policy in list(self.policies.values())[start:]])

        return self.mean_rewards[nr_eval_episodes]

    def add_data(self, index: int, rewards: dict[int, float], policy: Policy, state_counts: tuple[list]):
        self.rewards.update(rewards)
        self.policies[index] = compress_policy(policy)
        self.state_counts[index] = state_counts

    def get_state_counts(self, index: int):
        return self.state_counts[index]


class FullThompsonSampling(DirThompsonSampling):
    def __init__(self, model: FullModel, rewards: dict[int, float], state_counts: dict[int, list], policies: dict[int, Policy], name: str, info: dict):
        super().__init__(model, rewards, state_counts, policies, name, info)

    def add_data(self, index: int, rewards: dict[int, float], policy: Policy, state_counts: list):
        self.rewards.update(rewards)
        self.policies[index] = compress_policy(policy)
        self.state_counts[index] = compress_array(state_counts)

    def get_state_counts(self, index: int):
        return decompress_array(*self.state_counts[index], (n_states, n_actions, n_states))


def plot_mean_rewards_ts(ts: List[DirThompsonSampling], policy_range: int, nr_eval_episodes: int):
    plt.figure(figsize=(10, 6))

    for t in ts:
        mean_rewards = t.get_mean_rewards(policy_range, nr_eval_episodes)
        plt.plot(list(t.policies.keys())[:policy_range],
                 mean_rewards, label=t.info["name"])

    plt.xlabel(f"Policies")
    plt.ylabel(f"Mean reward across {nr_eval_episodes} episodes")
    plt.title("Mean Reward")
    plt.legend()  # Add a legend to distinguish evaluations
    plt.show()
