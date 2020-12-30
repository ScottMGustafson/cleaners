import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask_ml.datasets import make_regression

from cleaners.impute import ImputeByValue


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


def test_impute_by_value_dd():
    fake_ddf = make_fake_impute_data(to_pandas=False)
    imputer = ImputeByValue(
        missing_values=np.nan,
        strategy="constant",
        add_indicator=False,
        copy=False,
        cols=["a", "b"],
        fill_value=1,
    )

    ones = np.ones(10)

    pre_a = fake_ddf["a"].values.compute()
    pre_b = fake_ddf["b"].values.compute()
    res = imputer.fit(fake_ddf).transform(fake_ddf)

    res_arr = res["a"].values.compute()
    np.testing.assert_array_equal(res_arr, ones)
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, res_arr, pre_a)

    res_arr = res["b"].values.compute()
    np.testing.assert_array_equal(res_arr, pre_b)


def test_impute_by_value_pd():
    fake_df = make_fake_impute_data(to_pandas=True)
    imputer = ImputeByValue(
        missing_values=np.nan,
        strategy="constant",
        add_indicator=False,
        copy=False,
        cols=["a", "b"],
        fill_value=1,
    )

    ones = np.ones(10)

    pre_a = fake_df["a"].values
    pre_b = fake_df["b"].values
    res = imputer.fit(fake_df).transform(fake_df)

    res_arr = res["a"].values
    np.testing.assert_array_equal(res_arr, ones)
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, res_arr, pre_a)

    res_arr = res["b"].values
    np.testing.assert_array_equal(res_arr, pre_b)
