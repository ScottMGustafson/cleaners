import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from cleaners import eda

from .make_data import make_fake_data, make_various_type_data


def test_process_feats_fails_on_dask():
    df = make_fake_data(to_pandas=False)

    with pytest.raises(AssertionError):
        _, _ = eda.process_feats(df, unique_thresh=1, feats=["a", "b", "c"])


def test_process_feats():
    df = make_fake_data(to_pandas=True)

    feat_type_dct, feat_class_dct = eda.process_feats(df, unique_thresh=1, feats=["a", "b", "c"])

    assert feat_class_dct == {"a": "categorical", "b": "binary", "c": "continuous"}, "{}".format(
        feat_class_dct
    )
    assert feat_type_dct == {"a": "object", "b": "numeric", "c": "numeric"}, "{}".format(
        feat_type_dct
    )


def test_get_unary_columns():
    df = make_fake_data(to_pandas=True)
    df["d"] = np.ones(10)
    res = eda.get_unary_columns(df)
    assert res == ["d"], f"{res}"


def test_mostly_nan():
    df = make_fake_data(to_pandas=True)
    df["d"] = np.ones(10)
    for i in [0, 1, 3, 5, 6]:
        df.loc[i, "d"] = np.nan

    # nan rate == 80%
    res = eda.get_mostly_nan(df, thresh=0.2)
    assert res == []

    # nan rate == 20%
    res = eda.get_mostly_nan(df, thresh=0.8)
    assert res == ["d"]


def test_get_binary():
    df = make_fake_data(to_pandas=True)
    res = eda.get_binary(df)
    assert res == ["b"]


def test_get_mostly():
    df = make_fake_data(to_pandas=True)
    df["d"] = np.ones(10)
    df.loc[0, "d"] = 0
    df.loc[3, "d"] = 0
    df.loc[4, "d"] = 0
    # test that 80% of d is 1
    res = eda.get_mostly_value(df, 1, thresh=0.8)
    assert res == []

    # test that at least 20% of d is 1
    res = eda.get_mostly_value(df, 1, thresh=0.2)
    assert res == ["b", "d"]


def test_find_relevant():
    res = eda.find_relevant_columns(list("abcdefg"), model_feats=list("abc"), mandatory=list("de"))
    assert res == list("abcde"), str(res)


def test_get_uninformative():
    feat_class_dct = dict(
        a="uninformative", b="null", c="binary", d="binary", e="null", f="uninformative"
    )
    res = eda.get_uninformative(feat_class_dct, mandatory=("a", "b", "c"))
    assert res == ["e", "f"]


def test_get_type_list():
    exclude_lst = list("xyz")
    feat_type = "categorical"
    feat_type_dct = dict(a="numeric", b="categorical", x="categorical", y="binary")
    res = eda.get_type_lst(feat_type_dct, feat_type, exclude_lst)
    assert res == ["b"]


def test_get_categorical():
    df = make_various_type_data()
    res = eda.get_categorical(df, thresh=5)
    assert res == ["a", "b", "c", "d", "g"]


def test_describe_df_fails_on_dask():
    ddf = dd.from_pandas(make_various_type_data(), npartitions=2)
    with pytest.raises(AssertionError):
        _ = eda.get_describe_df(ddf, unique_thresh=6, percentiles=None)


def test_describe_df():
    df = make_various_type_data()
    df = eda.get_describe_df(df, unique_thresh=6, percentiles=None)

    exp_df = pd.DataFrame(
        {
            "inferred_kind": {
                "a": "categorical",
                "b": "categorical",
                "c": "categorical",
                "d": "categorical",
                "e": "binary",
                "f": "binary",
                "g": "categorical",
                "h": "continuous",
                "i": "continuous",
            },
            "inferred_type": {
                "a": "object",
                "b": "numeric",
                "c": "numeric",
                "d": "object",
                "e": "numeric",
                "f": "numeric",
                "g": "numeric",
                "h": "numeric",
                "i": "numeric",
            },
            "max": {
                "a": "",
                "b": 5.0,
                "c": 1.5,
                "d": "",
                "e": 1.0,
                "f": 1.0,
                "g": 2.0,
                "h": 1.53,
                "i": 10.0,
            },
            "min": {
                "a": "",
                "b": 1.0,
                "c": 1.2,
                "d": "",
                "e": 0.0,
                "f": 0.0,
                "g": 0.0,
                "h": 1.2,
                "i": 1.0,
            },
            "n_unique": {
                "a": 7.0,
                "b": 5.0,
                "c": 3.0,
                "d": 3.0,
                "e": 2.0,
                "f": 2.0,
                "g": 3.0,
                "h": 10.0,
                "i": 10.0,
            },
            "nan_rate": {
                "a": 0.0,
                "b": 0.0,
                "c": 0.0,
                "d": 0.0,
                "e": 0.0,
                "f": 0.1,
                "g": 0.1,
                "h": 0.0,
                "i": 0.0,
            },
            "zero_rate": {
                "a": 0.0,
                "b": 0.0,
                "c": 0.0,
                "d": 0.0,
                "e": 0.4,
                "f": 0.3,
                "g": 0.2,
                "h": 0.0,
                "i": 0.0,
            },
        }
    )
    assert isinstance(df, pd.DataFrame)
    allowed_cols = exp_df.columns
    df = df[allowed_cols].sort_index()
    exp_df = exp_df.sort_index()
    pd.testing.assert_frame_equal(df, exp_df)
