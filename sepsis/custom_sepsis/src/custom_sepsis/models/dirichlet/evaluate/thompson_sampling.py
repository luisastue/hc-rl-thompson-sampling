from ..model import DirModel, FullModel, Simplification, MediumModel, SimpleModel
import os
import numpy as np
import json
from ....sepsis_env import Policy, evaluate_policy,  compress_policy, decompress_policy
import gzip
import dill as pickle


class DirThompsonSampling():
    def __init__(self, model: DirModel, rewards: dict[int, float], models: dict[int, any], policies: dict[int, Policy], mean_rewards: dict[int, list[float]], name: str, info: dict):
        self.name = name
        self.info = info
        self.info["name"] = name
        self.info["date"] = str(np.datetime64('now'))
        self.model = model
        self.policies = policies
        self.models = models
        self.mean_rewards = mean_rewards
        self.rewards = rewards

    def save_json(self, directory="json/dirichlet/ts"):
        os.makedirs(directory, exist_ok=True)
        object_path = os.path.join(
            directory, f"{self.name}.json"
        )

        with open(object_path, 'w') as file:
            json_file = {
                "info": self.info,
                "model": self.model.to_dict(),
                "policies": {k: compress_policy(p) for k, p in self.policies.items()},
                "models": {k: self.model.to_dict_counts(v) for k, v in self.models.items()},
                "mean_rewards": {k: v.tolist() for k, v in self.mean_rewards.items()},
                "rewards": {k: rew.tolist() for k, rew in self.rewards.items()}
            }
            json.dump(json_file, file)

        return object_path

    @staticmethod
    def load(object_path: str):
        with gzip.open(object_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_json(object_path: str):
        with open(object_path, 'r') as file:
            json_file = json.load(file)
            model = None
            if json_file["model"]["type"] == Simplification.NONE.value:
                model = FullModel(
                    FullModel.from_dict_counts(json_file["model"]["state_counts"]))
            elif json_file["model"]["type"] == Simplification.MEDIUM.value:
                model = MediumModel(
                    MediumModel.from_dict_counts(json_file["model"]["state_counts"]))
            elif json_file["model"]["type"] == Simplification.SIMPLE.value:
                model = SimpleModel(
                    SimpleModel.from_dict_counts(json_file["model"]["state_counts"]))
            policies = {k: decompress_policy(p)
                        for k, p in json_file["policies"].items()}
            models = {k: model.from_dict_counts(v)
                      for k, v in json_file["models"].items()}
            return DirThompsonSampling(model, json_file["rewards"], models, policies, json_file["mean_rewards"], json_file["info"]["name"], json_file["info"])

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
