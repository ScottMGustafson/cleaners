import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from cleaners import drop_replace

from .make_data import make_fake_data


def test_drop_named():
    ddf = make_fake_data(to_pandas=False)
    obj = drop_replace.DropNamedCol(drop_cols=["b", "c"])
    res = obj.fit_transform(ddf).compute()
    assert "b" not in res.columns
    assert "c" not in res.columns

    feats_out = obj.get_feature_names_out()
    assert not any(x in feats_out for x in ["b", "c"])

    res2 = obj.transform(ddf).compute()
    pd.testing.assert_frame_equal(res2, res)


def test_replace_name():
    obj = drop_replace.ReplaceBadColnameChars()
    df = make_fake_data(to_pandas=True)
    df["b[, ]<>>><<ad"] = np.ones(10)
    ddf = dd.from_pandas(df, npartitions=1)
    res = obj.fit_transform(ddf).compute()
    assert "bad" in res.columns

    feats_out = obj.get_feature_names_out()
    assert "b[, ]<>>><<ad" not in feats_out
    assert "bad" in feats_out


def test_dropna():
    df = make_fake_data(to_pandas=True)
    df.loc[9, "c"] = np.inf
    df.loc[9, "a"] = "NaN"
    ddf = dd.from_pandas(df, npartitions=2)
    obj = drop_replace.DropNa(["b", "c", "a"], replace_infinities=True)
    res = obj.transform(ddf).compute()
    assert res.index.size == 8, str(res)


def test_dropna_dd():
    df = make_fake_data(to_pandas=True)
    df.loc[9, "c"] = np.inf
    df.loc[9, "a"] = "NaN"
    ddf = dd.from_pandas(df, npartitions=2)
    obj = drop_replace.DropNa(["b", "c", "a"], replace_infinities=True)
    res = obj.transform(ddf).compute()
    assert res.index.size == 8, str(res)


def test_drop_dupes_raises():
    obj = drop_replace.DropDuplicates(silently_fix=False, df_identifier="")
    df = make_fake_data(to_pandas=True)
    df["d"] = df[["b"]].copy()
    df.columns = ["a", "b", "c", "b"]
    ddf = dd.from_pandas(df, npartitions=2)
    with pytest.raises(AssertionError):
        _ = obj.transform(ddf)


def test_drop_dupes():
    obj = drop_replace.DropDuplicates(silently_fix=True, df_identifier="")
    df = make_fake_data(to_pandas=True)
    df["d"] = df[["b"]].copy()
    df.columns = ["a", "b", "c", "b"]
    ddf = dd.from_pandas(df, npartitions=2)
    res = obj.transform(ddf).compute()
    assert len(res.columns) == 3, str(res)
