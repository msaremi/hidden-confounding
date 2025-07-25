import numpy as np
from scipy.stats import norm, bernoulli
from scipy.special import expit
import pandas as pd

def generate_data(num_env: int=10, num_obs: int=100, coef: float=5.):
    """
    Generates data based on the distributions of the article
    :param num_env: Number of environments
    :param num_obs: Number of observations per environment
    :param coef: λ
    :return: A dataframe containing the data
    """
    sigma_theta_t, sigma_theta_u, sigma_theta_y = 1., 1., 1.

    rvs = norm.rvs(loc=[0, 0, 0], scale=[sigma_theta_t, sigma_theta_u, sigma_theta_y], size=(num_env, 3))
    theta_t = rvs[:, 0]
    theta_u = rvs[:, 1]
    theta_y = rvs[:, 2]

    t = np.empty((num_env, num_obs))
    u = np.empty_like(t)
    y = np.empty_like(t)

    for env in np.arange(num_env):
        u[env, :] = norm.rvs(loc=theta_u[env], scale=1)
        t[env, :] = bernoulli.rvs(p=expit(u[env, :] + theta_t[env]))
        y[env, :] = bernoulli.rvs(p=expit(coef * u[env, :] + t[env, :] + theta_y[env]))

    columns = pd.MultiIndex.from_product([['u', 't', 'y'], np.arange(num_obs)])
    return pd.DataFrame(np.hstack((u, t, y)), columns=columns)