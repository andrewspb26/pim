import numpy as np


class ProbabilisticIndexModel:

    def __init__(self):
        self.target = None
        self.predictors = None

    def __generate_pseudo_observations(self, target):
        if isinstance(target, np.ndarray):
            mask = target[:, None] > target
            self.target = mask[~np.eye(target.size, dtype=bool)]
        else:
            raise ValueError('target should be np.ndarray type')

    def __generate_predictors(self, predictors):
        pass

    def fit(self, target, predictors, is_transformed=False):
        if not is_transformed:
            self.__generate_pseudo_observations(target)
            self.__generate_predictors(predictors)
