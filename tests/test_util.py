import pytest

from cleaners.util import set_index, sort_index
from tests.make_data import make_date_data


def test_sort_index():
    pd_df = make_date_data(num_periods=10).sample(frac=1).set_index("date")
    sorted_pd_df = sort_index(pd_df)
    assert sorted_pd_df["cumsum"].values.tolist() == list(map(float, range(1, 11)))


def test_sort_index_dd():
    df = make_date_data(num_periods=10, to_dask=True).sample(frac=1).set_index("date")
    sorted_df = sort_index(df)
    assert sorted_df["cumsum"].compute().values.tolist() == list(map(float, range(1, 11)))


def test_set_index():
    pd_df = make_date_data(num_periods=10)
    sorted_pd_df = set_index(pd_df, ix_name="date", sort=True)
    assert sorted_pd_df["cumsum"].values.tolist() == list(map(float, range(1, 11)))


def test_set_index_dd():
    df = make_date_data(num_periods=10, to_dask=True)
    sorted_df = set_index(df, ix_name="date", sort=True)
    assert sorted_df["cumsum"].compute().values.tolist() == list(map(float, range(1, 11)))


@pytest.mark.parametrize("to_dask", [True, False])
def test_sort_index_raises(to_dask):
    df = make_date_data(num_periods=10, to_dask=to_dask).sample(frac=1).reset_index()
    with pytest.raises(AssertionError):
        _ = sort_index(df)
