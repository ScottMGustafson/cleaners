import numpy as np
import pytest

from cleaners.impute import ImputeByValue

from .make_data import make_fake_impute_data


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
