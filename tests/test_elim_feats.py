import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from cleaners import eliminate_feats


def test_mostly_nan():
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
    obj = eliminate_feats.DropMostlyNaN(mandatory=["f"], nan_frac_thresh=0.3)
    res = obj.transform(df)

    assert not any([x in res.columns for x in ["b", "d"]])
    assert all([x in res.columns for x in ["a", "c", "e", "f"]]), str(res)


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
    res = obj.transform(df).compute()
    assert not any([x in res.columns for x in ["b", "d"]])
    assert all([x in res.columns for x in ["a", "c", "e", "f"]]), str(res)


@pytest.mark.skip
def test_mostly_nan_score():
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
    obj = eliminate_feats.DropMostlyNaN(
        drop_cols=("a", "b", "e"), mandatory=["f"], nan_frac_thresh=0.3, sample_rate=None
    )
    res = obj.transform(df).compute()
    assert not any([x in res.columns for x in ["a", "b", "e"]]), str(res)
    assert all([x in res.columns for x in ["c", "d", "f"]]), str(res)
