import os
import pickle
import stable_baselines3 as sb3
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from .evaluate import Evaluation
from custom_sepsis import Episode


class ThompsonSampling(Evaluation):
    def __init__(self, nr_iterations: int, episodes: List[Episode], name: str, info: dict):
        super().__init__(nr_iterations, episodes, name, info)
        self.policy = None

    def save(self, directory="data/thompson_sampling"):
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

    def get_policy(self):
        # Implement Thompson Sampling policy logic here
        # Placeholder logic for generating a policy
        if self.policy is None:
            self.policy = self.episodes[len(self.episodes) - 1].policy
        return self.policy
