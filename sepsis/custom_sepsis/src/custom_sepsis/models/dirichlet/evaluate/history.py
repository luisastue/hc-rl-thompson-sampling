import json
import os
import numpy as np
from typing import List
from ..model import DirModel, Simplification, FullModel, MediumModel, SimpleModel
from ....sepsis_env.sepsis_types import Policy, compress_policy, decompress_policy


class DirHistory():
    def __init__(self, model: DirModel, policies: dict[int, List[Policy]], models: dict[int, any], mean_rewards: dict[int, list[float]], name: str, info: dict):
        self.name = name
        self.info = info
        self.info["name"] = name
        self.info["date"] = str(np.datetime64('now'))
        self.model = model
        self.policies = policies
        self.models = models
        self.mean_rewards = mean_rewards

    def save_json(self, directory="json/dirichlet/history"):
        os.makedirs(directory, exist_ok=True)
        object_path = os.path.join(
            directory, f"{self.name}.json"
        )

        with open(object_path, 'w') as file:
            json_file = {
                "info": self.info,
                "model": self.model.to_dict(),
                "policies": {k: [compress_policy(p) for p in v] for k, v in self.policies.items()},
                "models": {k: self.model.to_dict_counts(v) for k, v in self.models.items()},
                "mean_rewards": self.mean_rewards
            }
            json.dump(json_file, file)

        return object_path

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
            policies = {k: [decompress_policy(p) for p in v]
                        for k, v in json_file["policies"].items()}
            models = {k: model.from_dict_counts(v)
                      for k, v in json_file["models"].items()}
            return DirHistory(model, policies, models, json_file["mean_rewards"], json_file["info"]["name"], json_file["info"])

    def add_point(self, index: int, policies: List[Policy], rewards: List[float], models: any):
        self.policies[index] = policies
        self.mean_rewards[index] = rewards
        self.models[index] = models
