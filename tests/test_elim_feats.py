import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from cleaners import eliminate_feats


def test_mostly_nan():
    df = pd.DataFrame(
        dict(
            a=[1, 2, np.nan, 3, 4, 5],
            b=[np.nan, 2, np.nan, 3, np.nan, np.nan],
            c=[1, 2, 0, 3, 0, 0],
            d=[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            e=[1, 2, 0, 3, 0, 0],
            f=[1, 2, np.nan, np.nan, np.nan, np.nan],
        )
    )
    obj = eliminate_feats.DropMostlyNaN(mandatory=["f"], nan_frac_thresh=0.3)
    res = obj.fit_transform(df)

    assert res.columns.tolist() == ["a", "c", "e", "f"]


def test_mostly_nan_dd():
    df = pd.DataFrame(
        dict(
            a=[1, 2, np.nan, 3, 4, 5],
            b=[1, 2, np.nan, 3, np.nan, np.nan],
            c=[1, 2, 0, 3, 0, 0],
            d=[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            e=[1, 2, 0, 3, 0, 0],
            f=[1, 2, np.nan, 3, np.nan, np.nan],
        )
    )
    df = dd.from_pandas(df, npartitions=2)
    obj = eliminate_feats.DropMostlyNaN(mandatory=["f"], nan_frac_thresh=0.3, sample_rate=None)
    res = obj.fit_transform(df).compute()
    assert not any([x in res.columns for x in ["b", "d"]])
    assert all([x in res.columns for x in ["a", "c", "e", "f"]]), str(res)


def test_mostly_nan_score():
    df = pd.DataFrame(
        dict(
            a=[1, 2, np.nan, 3, 4, 5],
            b=[1, 2, np.nan, np.nan, np.nan, np.nan],
            c=[1, 2, 0, 3, 0, 0],
            d=[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            e=[1, 2, 0, 3, 0, 0],
            f=[np.nan, 2, np.nan, 3, np.nan, np.nan],
        )
    )
    df = dd.from_pandas(df, npartitions=2)
    obj = eliminate_feats.DropMostlyNaN(mandatory=["f"], nan_frac_thresh=0.5, sample_rate=None)
    res = obj.fit_transform(df).compute()
    assert res.columns.tolist() == ["a", "c", "e", "f"]


def test_base_drop_cols_transf(make_pd_data):
    df = make_pd_data
    obj = eliminate_feats.BaseDropColsMixin()
    obj.feature_names_in_ = df.columns.tolist()
    pd.testing.assert_frame_equal(obj.transform(df), df)
    obj.drop_cols_ = _to_drop = [x for x in df.columns if x.startswith("var")]
    df_ = obj.transform(df)
    assert not any(x in df_.columns for x in _to_drop)
    assert obj.get_feature_names_out() == [x for x in df.columns if not x.startswith("var")]


def test_drop_mostly_nan():
    rs = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "a": rs.choice([1, np.nan], p=[0.7, 0.3], size=1000),
            "b": rs.choice([1, np.nan], p=[0.4, 0.6], size=1000),
            "c": rs.choice([1, np.nan], p=[0.2, 0.8], size=1000),
            "d": rs.choice([1, np.nan], p=[0.8, 0.2], size=1000),
        }
    )

    res = eliminate_feats.DropMostlyNaN.find_mostly_nan(df, nan_frac_thresh=0.5)
    assert res == ["b", "c"]


def test_validate_missing():
    obj = eliminate_feats.DropMostlyNaN()

    X = pd.DataFrame({k: [1, 2, 3] for k in "abcde"})
    obj.mandatory = ["a", "b", "c"]
    obj.drop_cols_ = ["a", "d", "e"]

    with pytest.raises(AssertionError):
        obj._validate_missing(X)

    obj.skip_if_missing = False
    obj.drop_cols_ = ["d", "e", "f", "g"]
    with pytest.raises(KeyError):
        obj._validate_missing(X)

    obj.skip_if_missing = True
    obj.drop_cols_ = ["d", "e", "f", "g"]
    obj._validate_missing(X)
    assert obj.drop_cols_ == ["d", "e"]
