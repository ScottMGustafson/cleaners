import dask.dataframe as dd
import pytest

from cleaners import util
from cleaners.util import set_index, sort_index
from tests.make_data import make_date_data


def test_sort_index():
    pd_df = make_date_data(n_samples=10).sample(frac=1).set_index("date")
    sorted_pd_df = sort_index(pd_df)
    assert sorted_pd_df["cumsum"].values.tolist() == list(map(float, range(1, 11)))


def test_sort_index_dd():
    df = make_date_data(n_samples=10, to_dask=True).sample(frac=1).set_index("date")
    sorted_df = sort_index(df)
    assert sorted_df["cumsum"].compute().values.tolist() == list(map(float, range(1, 11)))


def test_set_index():
    pd_df = make_date_data(n_samples=10)
    sorted_pd_df = set_index(pd_df, ix_name="date", sort=True)
    assert sorted_pd_df["cumsum"].values.tolist() == list(map(float, range(1, 11)))


def test_set_index_dd():
    df = make_date_data(n_samples=10, to_dask=True)
    sorted_df = set_index(df, ix_name="date", sort=True)
    assert sorted_df["cumsum"].compute().values.tolist() == list(map(float, range(1, 11)))


@pytest.mark.parametrize("to_dask", [True, False])
def test_sort_index_raises(to_dask):
    df = make_date_data(n_samples=10, to_dask=to_dask).sample(frac=1).reset_index()
    with pytest.raises(IndexError):
        _ = sort_index(df)


def test_cumsum_index(make_pd_data):
    df = make_date_data()
    df_ = util.cum_sum_index(df)
    assert df_.index.name == "cum_sum"
    assert df_.index.values.min() == 1
    assert df_.index.values.max() == len(df)

    df_ = util.cum_sum_index(dd.from_pandas(df, npartitions=3))
    df_ = df_.compute()
    assert df_.index.name == "cum_sum"
    assert df_.index.values.min() == 1
    assert df_.index.values.max() == len(df)
