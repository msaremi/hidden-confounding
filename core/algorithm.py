import numpy as np
from scipy.stats import chi2
from pgmpy.estimators.CITests import g_sq

def test_confounder(data, num_env=None, num_obs=None):
    """
    Algorithm for detecting the existence of confounders
    :param data: The dataframe
    :param num_env: Number of environments, must be equal to or smaller than the number of environments in data
    :param num_obs: Number of observations per environment, must be equal to or smaller than the number of observations in data
    :return: aggregated p-value
    """
    if num_env is None:
        num_env = data.shape[0]

    if num_obs is None:
        num_obs = data['u'].shape[1]

    num_cmp = num_obs // 2
    data = data.loc[:num_env]
    p_vals = np.empty((num_cmp,))

    for i in np.arange(num_cmp):
        # We know the direction of causation, so a single test suffices
        _, p_vals[i], _ = g_sq(('t', 2 * i), ('y', 2 * i + 1), (('t', 2 * i + 1),), data, boolean=False)

    z = -2 * np.sum(np.log(p_vals + 1e-4))
    return 1 - chi2.cdf(z, df=2 * num_cmp)