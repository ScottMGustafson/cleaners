import dask.dataframe as dd
import pandas as pd
import pytest

from cleaners.cleaner_base import CleanerBase, DaskDataFrameNotSampled, DataTooSmallForEDA


class ConcreteClass(CleanerBase):
    def __init(self, **kwargs):
        super(ConcreteClass, self).__init__(**kwargs)

    def fit(self, X, y=None, **kwargs):
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        return X


def test_get_sample_df(make_pd_data):
    obj = ConcreteClass(logger_name="test", sample_rate=None, verbose=False, fail_on_warning=True)
    df = make_pd_data
    obj.get_sample_df(df, min_rows=10)
    assert obj.sample_df is not df
    pd.testing.assert_frame_equal(obj.sample_df, df)


def test_get_sample_ddf_fails_on_small_data(make_pd_data):
    sample_rate = 0.5
    obj = ConcreteClass(
        logger_name="test", sample_rate=sample_rate, verbose=False, fail_on_warning=True
    )
    df = make_pd_data
    df = dd.from_pandas(df, npartitions=4)
    min_rows = (df.index.size.compute() * sample_rate) + 1  # ensure min rows is actually bigger
    with pytest.raises(DataTooSmallForEDA):
        obj.get_sample_df(df, min_rows=min_rows)


def test_get_sample_ddf_fails_on_no_sample_rate(make_pd_data):
    obj = ConcreteClass(logger_name="test", sample_rate=None, verbose=False, fail_on_warning=True)
    df = make_pd_data
    df = dd.from_pandas(df, npartitions=4)
    with pytest.raises(DaskDataFrameNotSampled):
        obj.get_sample_df(df, min_rows=9)


def test_get_sample_ddf(make_pd_data):
    obj = ConcreteClass(logger_name="test", sample_rate=0.6, verbose=False, fail_on_warning=True)
    df = make_pd_data
    df = dd.from_pandas(df, npartitions=4)
    obj.get_sample_df(df, min_rows=5)
    assert obj.sample_df.index.size <= df.index.size.compute()


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
    obj = ConcreteClass(logger_name="test", sample_rate=None, verbose=False, fail_on_warning=fail)

    if fail:
        with pytest.raises(exp_exc):
            obj._fail_on_warning("test", exception=exc)
    else:
        obj._fail_on_warning("test", exception=exc)
