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


def get_types_classes_for_fake_data():
    feat_type_dct = dict(a="string", b="numeric", c="numeric")
    feat_class_dct = dict(a="categorical", b="binary", c="continuous")
    return feat_type_dct, feat_class_dct


def make_various_type_data():
    df = pd.DataFrame(
        dict(
            a=list("abcdefgabc"),  # more than 5 unique vals, but string
            b=[1, 2, 3, 4, 5, 4, 3, 2, 3, 1],
            c=[
                1.2,
                1.3,
                1.5,
                1.2,
                1.3,
                1.5,
                1.2,
                1.3,
                1.2,
                1.3,
            ],  # float with fewer than 5 unique vals
            d=list("abcabcabca"),
            e=[1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
            f=[1, 0, 1, 1, 1, 0, 1, 0, 1, np.nan],  # binary with nan
            g=[1, 0, 1, 1, 1, 0, 1, 2, 1, np.nan],  # categborical int with nan
            h=[
                1.2,
                1.3,
                1.5,
                1.21,
                1.32,
                1.53,
                1.24,
                1.35,
                1.26,
                1.37,
            ],  # float with more than 5 unique vals
            i=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # int with more than 5 unique vals
        )
    )
    return df
