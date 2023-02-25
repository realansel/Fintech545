import numpy as np
import pandas as pd


def return_calculate(prices, method='DISCRETE', date_column='Date'):
    vars = prices.columns
    n_vars = len(vars)
    vars = [v for v in vars if v != date_column]
    if n_vars == len(vars):
        raise ValueError(f'{date_column} not in DataFrame {vars}')
    n_vars = len(vars)
    p = prices[vars].to_numpy()
    n, m = p.shape
    p2 = np.empty((n - 1, m))
    for i in range(n - 1):
        p2[i, :] = p[i + 1, :] / p[i, :]
    if method.upper() == 'DISCRETE':
        p2 -= 1.0
    elif method.upper() == 'LOG':
        p2 = np.log(p2)
    else:
        raise ValueError(f'method {method} must be in ("LOG", "DISCRETE")')
    dates = prices.iloc[1:, prices.columns.get_loc(date_column)]
    out = pd.DataFrame({date_column: dates})
    for i in range(n_vars):
        out[vars[i]] = p2[:, i]
    return out