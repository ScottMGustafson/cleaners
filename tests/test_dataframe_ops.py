import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from cleaners import dataframe_ops
from tests.make_data import make_fake_date_data


def test_multi_index_pd_resets():
    ix_list = ["date", "a"]
    df = make_fake_date_data(to_pandas=True).set_index(ix_list)
    obj = dataframe_ops.CompositeIndex(ix_list=ix_list, drop=False, new_ix_name="index")
    _ = obj.transform(df)


def test_multi_index_pd():
    ix_list = ["date", "a"]
    df = make_fake_date_data(to_pandas=True)
    obj = dataframe_ops.CompositeIndex(ix_list=ix_list, drop=False, new_ix_name="index")
    res = obj.transform(df)
    res_index = list(res.index.values)
    expected = [
        "2021-01-01-a",
        "2021-01-02-b",
        "2021-01-03-a",
        "2021-01-16-a",
        "2021-01-11-b",
        "2021-01-20-c",
        "2021-01-21-a",
        "2021-01-21-a",
        "2021-01-23-a",
        "2021-01-24-b",
    ]
    assert sorted(res_index) == sorted(expected), "{}".format(res_index)


def test_multi_index_dd():
    ix_list = ["date", "a"]
    df = make_fake_date_data(to_pandas=False)
    obj = dataframe_ops.CompositeIndex(ix_list=ix_list, drop=False, new_ix_name="index")
    res = obj.transform(df).compute()
    res_index = list(res.index.values)
    expected = [
        "2021-01-01-a",
        "2021-01-02-b",
        "2021-01-03-a",
        "2021-01-16-a",
        "2021-01-11-b",
        "2021-01-20-c",
        "2021-01-21-a",
        "2021-01-21-a",
        "2021-01-23-a",
        "2021-01-24-b",
    ]
    assert sorted(res_index) == sorted(expected), "{}".format(res_index)


def make_merge_data():
    df1 = pd.DataFrame(
        dict(
            date=["2021-01-01", "2021-01-02", "2021-01-03", "2020-01-04"],
            a=[1, 2, 3, 4],
            b=[4, 5, 6, 7],
        )
    )

    df2 = pd.DataFrame(dict(date=["2021-01-03", "2021-01-04", "2021-01-05"], c=list("abc")))
    return df1, df2


def test_join_df_pd_raises():
    df1, df2 = make_merge_data()
    df1 = df1.set_index("date")
    df2["b"] = np.ones(3)  # overlapping column
    df2 = df2.set_index("date")

    obj = dataframe_ops.JoinDFs(df2, how="inner", join=True)
    with pytest.raises(AssertionError):
        _ = obj.transform(df1)


def test_join_df_pd():
    df1, df2 = make_merge_data()
    df1 = df1.set_index("date")
    df2 = df2.set_index("date")

    obj = dataframe_ops.JoinDFs(df2, how="inner", join=True)
    res = obj.transform(df1)
    exp = {"a": {"2021-01-03": 3}, "b": {"2021-01-03": 6}, "c": {"2021-01-03": "a"}}
    assert res.to_dict() == exp


def test_join_df_dd():
    df1, df2 = make_merge_data()
    df1 = dd.from_pandas(df1.set_index("date"), npartitions=2)
    df2 = dd.from_pandas(df2.set_index("date"), npartitions=1)

    obj = dataframe_ops.JoinDFs(df2, how="inner", join=True)
    res = obj.transform(df1).compute()
    exp = {"a": {"2021-01-03": 3}, "b": {"2021-01-03": 6}, "c": {"2021-01-03": "a"}}
    assert res.to_dict() == exp


def test_merge_df_dd():
    df1, df2 = make_merge_data()
    df1 = dd.from_pandas(df1, npartitions=2)
    df2 = dd.from_pandas(df2, npartitions=1)
    obj = dataframe_ops.JoinDFs(df2, how="inner", join=False, ix_col="date")
    res = obj.transform(df1).compute().set_index("date")
    exp = {"a": {"2021-01-03": 3}, "b": {"2021-01-03": 6}, "c": {"2021-01-03": "a"}}
    assert res.to_dict() == exp


def test_ffill_pd():
    obj = dataframe_ops.IndexForwardFillna(ix_col="date", method="ffill")
    df = make_fake_date_data(to_pandas=True).set_index("date")
    res = obj.transform(df)
    filled_nans = res[["b", "c"]].values[8]
    pre_arr = res[["b", "c"]].values[7]
    np.testing.assert_array_equal(filled_nans, pre_arr)


def test_ffill_dd():
    obj = dataframe_ops.IndexForwardFillna(ix_col="date", method="ffill")
    df = make_fake_date_data(to_pandas=False).set_index("date")
    res = obj.transform(df).compute()
    filled_nans = res[["b", "c"]].values[8]
    pre_arr = res[["b", "c"]].values[7]
    np.testing.assert_array_equal(filled_nans, pre_arr)
