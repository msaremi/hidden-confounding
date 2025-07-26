import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import chi2
from pgmpy.estimators.CITests import g_sq


def test_confounder(data, num_env=None, num_obs=None, concat=False):
    """
    Algorithm for detecting the existence of confounders
    :param data: The dataframe
    :param num_env: Number of environments, must be equal to or smaller than the number of environments in data
    :param num_obs: Number of observations per environment, must be equal to or smaller than the number of observations in data
    :param concat: If true, concatenate observations and then do a single independence test
    :return: aggregated p-value
    """
    if num_env is None:
        num_env = data.shape[0]

    if num_obs is None:
        num_obs = data['u'].shape[1]

    num_cmp = num_obs // 2

    if concat:
        # Concat multiple observations and do a single conditional independence test
        columns = pd.MultiIndex.from_product([['u', 't', 'y'], [0, 1]])
        concat_data = pd.DataFrame(np.empty((num_env * num_cmp, len(columns))), columns=columns)

        for i in np.arange(num_cmp):
            concat_data.iloc[num_env * i:num_env * (i + 1)] \
                = data[list(product(['u', 't', 'y'], [2 * i, 2 * i + 1]))].iloc[:num_env]

        _, p_val, _ = g_sq(('t', 0), ('y', 1), (('t', 1),), data, boolean=False)
        return p_val
    else:
        data = data.iloc[:num_env]
        p_vals = np.empty((num_cmp,))

        for i in np.arange(num_cmp):
            # We know the direction of causation, so a single test suffices
            _, p_vals[i], _ = g_sq(('t', 2 * i), ('y', 2 * i + 1), (('t', 2 * i + 1),), data, boolean=False)

        z = -2 * np.sum(np.log(p_vals + 1e-4))
        return 1 - chi2.cdf(z, df=2 * num_cmp)