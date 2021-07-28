import numpy as np


def generate_pseudo_observations(target: np.ndarray):
    if isinstance(target, np.ndarray):
        mask = target[:, None] > target
        return mask[~np.eye(target.size, dtype=bool)].astype(int)
    else:
        raise ValueError('target should be np.ndarray type')


def generate_pseudo_observations_with_ties(target: np.ndarray):
    if isinstance(target, np.ndarray):
        difference = target[:, None] - target
        mask = np.heaviside(difference, 0.5)
        mask = mask[~np.eye(target.size, dtype=bool)]
        return mask
    else:
        raise ValueError('target should be np.ndarray type')


def generate_predictors(predictors: np.ndarray):
    if isinstance(predictors, np.ndarray):
        predictors = predictors if len(predictors.shape) > 1 else predictors.reshape(predictors.shape[0], 1)
        features = []
        for column_idx in range(predictors.shape[1]):
            predictor = predictors[:, column_idx]
            mask = predictor[:, None] - predictor
            features.append(mask[~np.eye(predictor.size, dtype=bool)])
        return np.vstack(features).T
    else:
        raise ValueError('predictors should be np.ndarray type')
