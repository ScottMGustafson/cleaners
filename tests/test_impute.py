import dask.dataframe as dd
import numpy as np

from cleaners.impute import ImputeByValue

from .make_data import make_fake_impute_data


def impute_dask_smoke_test(make_pd_data):
    X = make_pd_data
    cat_cols = [c for c in X.columns if c.startswith("cat")]
    num_cols = [c for c in X.columns if c.startswith("var")]
    bin_cols = [c for c in X.columns if c.startswith("bin")]

    df = dd.from_pandas(X, npartitions=4)
    imputer = ImputeByValue(
        allow_passthrough=True,
        sample_rate=0.1,
        cols=cat_cols + bin_cols,
        imputer_kwargs=dict(
            strategy="most_frequent",
            fill_value="-999",
            add_indicator=True,
            copy=True,
        ),
    ).fit(df[cat_cols + bin_cols])

    expected_cols_out = [
        "cat_0",
        "cat_1",
        "cat_2",
        "cat_3",
        "cat_4",
        "cat_0_dupe",
        "cat_3_dupe",
        "bin_0",
        "bin_1",
        "bin_2",
        "bin_3",
        "bin_4",
        "bin_3_dupe",
        "bin_1_dupe",
        "missingindicator_bin_0",
        "missingindicator_bin_1",
        "missingindicator_bin_2",
        "missingindicator_bin_3",
        "missingindicator_bin_4",
        "missingindicator_bin_3_dupe",
        "missingindicator_bin_1_dupe",
    ]
    assert imputer.get_feature_names_out() == expected_cols_out

    # test with extra passthrough cols
    res = imputer.transform(df[cat_cols + bin_cols + num_cols]).head()
    assert all(x in res.columns for x in num_cols + expected_cols_out)


def test_impute_by_value_dd():
    fake_ddf = make_fake_impute_data(to_pandas=False)
    imputer = ImputeByValue(
        cols=["a", "b"],
        imputer_kwargs=dict(
            strategy="constant",
            fill_value=1,
            add_indicator=False,
            copy=False,
        ),
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
        cols=["a", "b"],
        imputer_kwargs=dict(
            strategy="constant",
            fill_value=1,
            add_indicator=False,
            copy=False,
        ),
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
