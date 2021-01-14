import pandas as pd
import pytest

from cleaners.cleaner_base import CleanerBase, DaskDataFrameNotSampled, DataTooSmallForEDA
from tests.make_data import make_fake_data


def test_get_sample_df():
    obj = CleanerBase(logger_name="test", sample_rate=None, verbose=False, fail_on_warning=True)
    df = make_fake_data(to_pandas=True)
    obj.get_sample_df(df, min_rows=10)
    assert obj.sample_df is not df
    pd.testing.assert_frame_equal(obj.sample_df, df)


def test_get_sample_ddf_fails_on_small_data():
    sample_rate = 0.5
    obj = CleanerBase(
        logger_name="test", sample_rate=sample_rate, verbose=False, fail_on_warning=True
    )
    df = make_fake_data(to_pandas=False)
    min_rows = (df.index.size.compute() * sample_rate) + 1  # ensure min rows is actually bigger
    with pytest.raises(DataTooSmallForEDA):
        obj.get_sample_df(df, min_rows=min_rows)


def test_get_sample_ddf_fails_on_no_sample_rate():
    obj = CleanerBase(logger_name="test", sample_rate=None, verbose=False, fail_on_warning=True)
    df = make_fake_data(to_pandas=False)
    with pytest.raises(DaskDataFrameNotSampled):
        obj.get_sample_df(df, min_rows=9)


def test_get_sample_ddf():
    obj = CleanerBase(logger_name="test", sample_rate=0.6, verbose=False, fail_on_warning=True)
    df = make_fake_data(to_pandas=False)
    obj.get_sample_df(df, min_rows=5)
    assert obj.sample_df.index.size == 6


@pytest.mark.parametrize(
    "fail, exp_exc, exc",
    [
        (True, KeyError, KeyError),
        (True, Exception, KeyError),  # KeyError derives from Exception
        (True, Exception, Exception),
        (False, None, None),
    ],
)
def test_fail_on_warning(fail, exp_exc, exc):
    obj = CleanerBase(logger_name="test", sample_rate=None, verbose=False, fail_on_warning=fail)

    if fail:
        with pytest.raises(exp_exc):
            obj.fail_on_warning("test", exception=exc)
    else:
        obj.fail_on_warning("test", exception=exc)
