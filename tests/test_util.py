import dask.dataframe as dd
import numpy as np
import pandas as pd
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


def test_assert_no_dupes():
    df = pd.DataFrame(
        {
            'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
            'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
            'rating': [4, 4, 3.5, 15, 5],
        }
    )
    df = pd.concat([df, df], axis=1)
    with pytest.raises(IndexError):
        util.assert_no_duplicate_columns(df)


def test_validate_feats():
    df = pd.DataFrame(
        {
            'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
            'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
            'rating': [4, 4, 3.5, 15, 5],
        }
    )
    with pytest.raises(KeyError):
        util.validate_feats(df, feats=["brand", "style", "rating", "coolness_coefficient"])


@pytest.mark.parametrize(
    "input, output",
    [
        (3, 3),
        (3.3, 3.3),
        ("3", "3"),
        (np.float64(6), float(6)),
        (np.int64(6), float(6)),
        (True, float(1)),
    ],
)
def test_to_float_or_str(input, output):
    """Test that _to_float_or_str converts types as expected."""
    assert util._to_float_or_str(input) == output


@pytest.mark.parametrize(
    "inp, exp",
    [
        ("A", "A"),
        (("A", "B"), ["A", "B"]),
        (np.int64(12), int(12)),
        (np.float64(0.1), float(0.1)),
        (np.inf, float(np.inf)),
        (list(np.array([[2], [1], [0]])), [[2], [1], [0]]),
        ({"A": list(np.array([[2], [1], [0]]))}, {"A": [[2], [1], [0]]}),
        ({"A": {"A": 1}}, {"A": {"A": 1}}),
    ],
)
def test_to_json_serializable(inp, exp):
    """Test json serialization."""
    res = util.to_json_serializable(inp)
    assert res == exp


@pytest.mark.parametrize(
    "inp, exp",
    [
        ({"A": list(np.array([[2], [1], [0]]))}, {"A": [[2], [1], [0]]}),
        ({"A": {"A": 1}}, {"A": {"A": 1}}),
    ],
)
def test_serializable_dict(inp, exp):
    """Test json serialization."""
    res = util.serializable_dict(inp)
    assert res == exp


@pytest.mark.parametrize(
    "bogus_object",
    [
        type('MyBogusClass', (object,), {'propertyName': 'propertyValue'}),
        pd.Series(dtype=float),
        np.array([1, 2, 3]),
    ],
)
def test_to_float_or_str_raises(bogus_object):
    """Test that _to_float_or_str raises an exception on a bad type."""
    with pytest.raises(ValueError):
        util._to_float_or_str(bogus_object)
