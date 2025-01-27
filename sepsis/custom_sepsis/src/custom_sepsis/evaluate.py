import numpy as np
from typing import List
import matplotlib.pyplot as plt
from custom_sepsis import run_episode, Policy

from abc import ABC, abstractmethod


class Evaluation(ABC):
    def __init__(self, nr_iterations: int, episodes: list, name: str, info: dict):
        self.nr_iterations = nr_iterations
        self.episodes = episodes
        self.name = name
        self.info = info
        self.info["name"] = name
        self.info["date"] = str(np.datetime64('now'))
        self.policy = None

    @abstractmethod
    def save(self, directory: str):
        pass

    @abstractmethod
    def load(cls, object_path: str):
        pass

    @abstractmethod
    def get_policy(self):
        pass

    def evaluate_policy(self, n_episodes: int):
        pol = self.get_policy()
        test_eps = [run_episode(pol) for _ in range(n_episodes)]
        return np.mean([np.sum(ep.rewards) for ep in test_eps])


def plot_rewards(evaluations: List[Evaluation], episode_range: int, window_size: int = 100):
    plt.figure(figsize=(10, 6))

    for evaluation in evaluations:
        rewards = [np.sum(ep.rewards)
                   for ep in evaluation.episodes[1:episode_range]]
        avg_rewards = [np.mean(rewards[i:i+window_size])
                       for i in range(len(rewards) - window_size + 1)]
        plt.plot(avg_rewards, label=evaluation.info["name"])

    plt.xlabel(f"{window_size} episodes")
    plt.ylabel(f"Mean reward across {window_size} episodes")
    plt.title("Sliding Window Average Reward")
    plt.legend()  # Add a legend to distinguish evaluations
    plt.show()


def plot_mean_rewards(evaluations: List[Evaluation], episode_range: int, nr_eval_episodes: int):
    plt.figure(figsize=(10, 6))

    for evaluation in evaluations:
        mean_rewards = [evaluate_policy(ep.policy, nr_eval_episodes)
                        for ep in evaluation.episodes[1:episode_range]]
        plt.plot(mean_rewards, label=evaluation.info["name"])

    plt.xlabel(f"Policies")
    plt.ylabel(f"Mean reward across {nr_eval_episodes} episodes")
    plt.title("Mean Reward")
    plt.legend()  # Add a legend to distinguish evaluations
    plt.show()


def plot_cumulative_rewards(evaluations: List[Evaluation], episode_range: int):
    plt.figure(figsize=(10, 6))

    for evaluation in evaluations:
        rewards = [np.sum(ep.rewards)
                   for ep in evaluation.episodes[1:episode_range]]
        plt.plot(np.cumsum(rewards), label=evaluation.info["name"])

    plt.xlabel(f"Episodes")
    plt.ylabel(f"Cumulative reward across episodes")
    plt.title("Cumulative Reward")
    plt.legend()  # Add a legend to distinguish evaluations
    plt.show()


def evaluate_policy(policy: Policy, n_episodes: int):
    test_eps = [run_episode(policy) for _ in range(n_episodes)]
    return np.mean([np.sum(ep.rewards) for ep in test_eps])
