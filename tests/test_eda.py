import numpy as np
import pytest

from cleaners import eda
from tests.make_data import make_fake_data


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
