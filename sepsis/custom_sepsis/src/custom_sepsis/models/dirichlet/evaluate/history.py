import os
import dill as pickle
import numpy as np
from typing import List
from ..model import DirModel
from ....sepsis_env.sepsis_types import Policy


class DirHistory():
    def __init__(self, model: DirModel, policies: dict[int, List[Policy]], state_counts: dict[int, any], mean_rewards: dict[int, list[float]], name: str, info: dict):
        self.name = name
        self.info = info
        self.info["name"] = name
        self.info["date"] = str(np.datetime64('now'))
        self.model = model
        self.policies = policies
        self.state_counts = state_counts
        self.mean_rewards = mean_rewards

    def save(self, directory="data/history"):
        os.makedirs(directory, exist_ok=True)
        object_path = os.path.join(
            directory, f"{self.name}.pkl"
        )

        with open(object_path, 'wb') as file:
            pickle.dump(self, file)

        return object_path

    @staticmethod
    def load(object_path: str):
        with open(object_path, 'rb') as file:
            return pickle.load(file)

    def add_point(self, index: int, policies: List[Policy], rewards: List[float], state_counts: any):
        self.policies[index] = policies
        self.mean_rewards[index] = rewards
        self.state_counts[index] = state_counts
