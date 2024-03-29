from unittest import mock

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

import cleaners.util
from cleaners import dataframe_ops

from .make_data import make_fake_date_data


def test_multi_index_pd_resets():
    ix_list = ["date", "a"]
    df = make_fake_date_data(to_pandas=True)
    obj = dataframe_ops.CompositeIndex(ix_list=ix_list, drop=False, new_ix_name="index")
    _ = obj.fit_transform(df)


def test_multi_index_pd():
    ix_list = ["date", "a"]
    df = make_fake_date_data(to_pandas=True)
    df["date"] = pd.to_datetime(df["date"])
    obj = dataframe_ops.CompositeIndex(ix_list=ix_list, drop=False, new_ix_name="index")
    res = obj.fit_transform(df)
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
    df["date"] = df["date"].astype("M8[us]")
    obj = dataframe_ops.CompositeIndex(ix_list=ix_list, drop=False, new_ix_name="index")
    res = obj.fit_transform(df).compute()
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
        _ = obj.fit_transform(df1)


def test_join_df_pd():
    df1, df2 = make_merge_data()
    df1 = df1.set_index("date")
    df2 = df2.set_index("date")

    obj = dataframe_ops.JoinDFs(df2, how="inner", join=True)
    res = obj.fit_transform(df1)
    exp = {"a": {"2021-01-03": 3}, "b": {"2021-01-03": 6}, "c": {"2021-01-03": "a"}}
    assert res.to_dict() == exp


def test_join_df_dd():
    df1, df2 = make_merge_data()
    df1 = dd.from_pandas(df1.set_index("date"), npartitions=2)
    df2 = dd.from_pandas(df2.set_index("date"), npartitions=1)

    obj = dataframe_ops.JoinDFs(df2, how="inner", join=True)
    res = obj.fit_transform(df1).compute()
    exp = {"a": {"2021-01-03": 3}, "b": {"2021-01-03": 6}, "c": {"2021-01-03": "a"}}
    assert res.to_dict() == exp


def test_merge_df_dd():
    df1, df2 = make_merge_data()
    df1 = dd.from_pandas(df1, npartitions=2)
    df2 = dd.from_pandas(df2, npartitions=1)
    obj = dataframe_ops.JoinDFs(df2, how="inner", join=False, ix_col="date")
    res = obj.fit_transform(df1).compute().set_index("date")
    exp = {"a": {"2021-01-03": 3}, "b": {"2021-01-03": 6}, "c": {"2021-01-03": "a"}}
    assert res.to_dict() == exp


def test_ffill_pd():
    obj = dataframe_ops.IndexForwardFillna(ix_col="date", method="ffill")
    df = make_fake_date_data(to_pandas=True).set_index("date")
    res = obj.fit_transform(df)
    filled_nans = res[["b", "c"]].values[8]
    pre_arr = res[["b", "c"]].values[7]
    np.testing.assert_array_equal(filled_nans, pre_arr)


def test_ffill_dd():
    obj = dataframe_ops.IndexForwardFillna(ix_col="date", method="ffill")
    df = make_fake_date_data(to_pandas=False).set_index("date")
    res = obj.fit_transform(df).compute()
    filled_nans = res[["b", "c"]].values[8]
    pre_arr = res[["b", "c"]].values[7]
    np.testing.assert_array_equal(filled_nans, pre_arr)


def test_ffill_transform_ix():
    obj = dataframe_ops.IndexForwardFillna(ix_col="date", method="ffill")
    df = make_fake_date_data(to_pandas=True).reset_index()

    out = obj.fit_transform(df)
    assert out.index.name == "date"

    df["cumsum"] = pd.Series(np.ones(len(df))).cumsum() + 1

    ddf = dd.from_pandas(df, npartitions=2)
    ddf = ddf.set_index("cumsum", sorted=True)
    out = obj.fit_transform(ddf).compute()
    assert out.index.name == "date"


def test_reset_index():
    df = make_fake_date_data(to_pandas=True)
    pd.testing.assert_frame_equal(df.reset_index(), dataframe_ops.ResetIndex().fit_transform(df))


@mock.patch("cleaners.dataframe_ops.sort_index")
def test_ffillna_sort_called(mock_sort):
    df = make_fake_date_data(to_pandas=True)
    mock_sort.return_value = df
    _ = dataframe_ops.IndexForwardFillna(is_sorted=False).transform(df)
    assert mock_sort.called
