import numpy as np


class ProbabilisticIndexModel:

    def __init__(self):
        self.target = None
        self.predictors = None

    def __generate_pseudo_observations(self, target: np.ndarray):
        if isinstance(target, np.ndarray):
            mask = target[:, None] > target
            self.target = mask[~np.eye(target.size, dtype=bool)]
        else:
            raise ValueError('target should be np.ndarray type')

    def __generate_predictors(self, predictors: np.ndarray):
        if isinstance(predictors, np.ndarray):
            predictors = predictors if len(predictors.shape) > 1 else predictors.reshape(predictors.shape[0], 1)
            features = []
            for column_idx in range(predictors.shape[1]):
                predictor = predictors[:, column_idx]
                mask = predictor[:, None] - predictor
                features.append(mask[~np.eye(predictor.size, dtype=bool)])
            self.predictors = np.vstack(features).T
        else:
            raise ValueError('predictors should be np.ndarray type')

    def fit(self, target, predictors, is_transformed=False):
        if not is_transformed:
            self.__generate_pseudo_observations(target)
            self.__generate_predictors(predictors)
        else:
            self.target, self.predictors = target, predictors
        if self.target.shape[0] != self.predictors.shape[0]:
            raise ValueError('target and predictors have different first dimensions')

