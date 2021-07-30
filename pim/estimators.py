import numpy as np
import statsmodels.api as sm


def glm(target: np.ndarray, predictors: np.ndarray, link='logit'):
    link = sm.families.links.logit() if link == 'logit' else sm.families.links.probit()
    model = sm.GLM(target, predictors, family=sm.families.Binomial(link))  # link function check ?
    results = model.fit()
    return results


def gee(target: np.ndarray, predictors: np.ndarray, link='logit', covariance='independence'):
    link = sm.families.links.logit() if link == 'logit' else sm.families.links.probit()
    model = sm.GEE(target, predictors, family=sm.families.Binomial(link), cov_struct=covariance)  # link function check ?
    results = model.fit()
    return results


def ols(target: np.ndarray, predictors: np.ndarray):
    pass
