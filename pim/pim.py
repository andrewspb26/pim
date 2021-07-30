import numpy as np
from pim.transformers import generate_pseudo_observations, generate_pseudo_observations_with_ties, generate_predictors


class ProbabilisticIndexModel:

    def __init__(self, target: np.ndarray, predictors: np.ndarray, with_ties=True):
        self.target = generate_pseudo_observations_with_ties(target) if with_ties else generate_pseudo_observations(
            target)
        self.predictors = generate_predictors(predictors)
        if self.target.shape[0] != self.predictors.shape[0]:
            raise ValueError('target and predictors have different first dimensions')

    def fit(self, estimator='GLM'):
       pass



