import numpy as np
import pandas as pd
from dask import dataframe as dd
from dask_ml.datasets import make_regression


def make_fake_regression(ncols=10, nrows=100):
    ddf = make_regression(
        n_samples=nrows,
        n_features=ncols,
        n_informative=10,
        n_targets=1,
        bias=0.0,
        effective_rank=None,
        tail_strength=0.5,
        noise=0.0,
        shuffle=True,
        coef=False,
        random_state=None,
        chunks=None,
    )
    return ddf


def make_fake_impute_data(to_pandas=False):
    nrows = 10
    row = np.ones(nrows)
    nan_row = np.array([1, 1, np.nan, np.nan, 1, 1, 1, 1, np.nan, 1])
    df = pd.DataFrame(dict(a=nan_row, b=row, c=row))
    if to_pandas:
        return df
    else:
        return dd.from_pandas(df, npartitions=2)


def make_fake_data(to_pandas=False):
    df = pd.DataFrame(
        dict(
            a=list("abaabcaaab"),
            b=[1, 1, 0, 0, 1, 1, 0, 0, np.nan, 0],
            c=[0.8011, 0.2202, 0.777, 0.736, 0.44, 0.1398, 0.593, 0.029, np.nan, 0.949],
        )
    )
    if to_pandas:
        return df
    else:
        return dd.from_pandas(df, npartitions=2)
