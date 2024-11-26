import matrix_value_iter as vi
from matrix_value_iter import Episode
from typing import List, Any, Optional
import pickle
import stable_baselines3 as sb3
import numpy as np
import matplotlib.pyplot as plt


class Evaluation:
    def __init__(self, nr_iterations: int, total_nr_episodes: int, episodes: List[vi.Episode], name: str, info: dict):
        self.nr_iterations = nr_iterations
        self.total_nr_episodes = total_nr_episodes
        self.episodes = episodes
        self.info = info
        self.info["name"] = name
        self.info["date"] = str(np.datetime64('now'))

    # Save the object as a pickle file
    def save_pickle(self):
        with open(f"data/evaluations/{self.info['name']}-{self.info['date']}.pkl", 'wb') as file:
            pickle.dump(self, file)

    # Load an object from a pickle file
    @staticmethod
    def load_pickle(file_path: str):
        with open(file_path, 'rb') as file:
            return pickle.load(file, fix_imports=True, encoding='bytes')


class RewardLoggingCallback(sb3.common.callbacks.BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.episodes = []
        self.evaluation = None
        self.current_rewards = []
        self.current_states = []

    def _on_step(self) -> bool:
        # Collect rewards at each step
        reward = self.locals["rewards"][0]
        state = vi.to_state(self.locals["infos"][0]["state"])
        self.current_rewards.append(reward)
        self.current_states.append(state)

        # Check if the episode has ended
        done = self.locals["dones"]
        if done:
            self.episodes.append(
                vi.Episode(None, self.current_rewards, self.current_states))
            self.current_states = []
            self.current_rewards = []  # Reset for the next episode

        return True  # Continue training

    # def _on_training_end(self) -> None:
        # Log final results at the end of training
        # if self.verbose > 0:
        # print(f"Training completed. Total episodes: {len(self.episodes)}")
        # print(f"Episodes: {(self.episodes)}")


def evaluate_model(model, total_timesteps: int, name: str):
    reward_callback = RewardLoggingCallback(verbose=1)
    model.learn(total_timesteps, callback=reward_callback)
    date = str(np.datetime64('now'))
    model_name = f"{name}-{date}"
    evaluation = Evaluation(total_timesteps, len(reward_callback.episodes), reward_callback.episodes, name, {
        "generated_by": "evaluate_model", "model_name": model_name})
    model.save(f"models/{model_name}")
    evaluation.save_pickle()
    return evaluation


def plot_evals(evaluations, episode_range: int, window_size: int = 100):
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


def plot_cumsums(evaluations: List[Evaluation], episode_range: int):
    plt.figure(figsize=(10, 6))

    for evaluation in evaluations:
        rewards = [np.sum(ep.rewards)
                   for ep in evaluation.episodes[1:episode_range]]
        plt.plot(np.cumsum(rewards), label=evaluation.info["name"])

    plt.xlabel(f"Episodes")
    plt.ylabel(f"Cumulative reward across episodes")
    plt.title("Average Reward")
    plt.legend()  # Add a legend to distinguish evaluations
    plt.show()
