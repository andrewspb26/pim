import numpy as np
from pim.transformers import generate_pseudo_observations, generate_predictors


class ProbabilisticIndexModel:

    def __init__(self):
        self.target = None
        self.predictors = None

    def fit(self, target: np.ndarray, predictors: np.ndarray, estimator='GLM', is_transformed=False):
        if not is_transformed:
            self.target = generate_pseudo_observations(target)
            self.predictors = generate_predictors(predictors)
        else:
            self.target, self.predictors = target, predictors
        if self.target.shape[0] != self.predictors.shape[0]:
            raise ValueError('target and predictors have different first dimensions')


