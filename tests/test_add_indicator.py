import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.api.types import CategoricalDtype

from cleaners import indicators

from .make_data import get_types_classes_for_fake_data, make_fake_data


def test_one_hot_encoding_str():
    ddf = make_fake_data(to_pandas=False)
    res = indicators._one_hot_encode_dd(ddf, cols=["a", "b"])
    assert set(res.columns) == {"c", "a_b", "a_a", "a_c", "b_1.0", "b_0.0"}

    a_a = res["a_a"].values.compute()
    a_b = res["a_b"].values.compute()
    a_c = res["a_c"].values.compute()

    expected_a = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 0])
    expected_b = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 1])
    expected_c = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(a_a, expected_a)
    np.testing.assert_array_equal(a_b, expected_b)
    np.testing.assert_array_equal(a_c, expected_c)


def test_one_hot_encoding_drop_first():
    ddf = make_fake_data(to_pandas=False)
    res = indicators._one_hot_encode_dd(ddf, cols=["a", "b"], drop_first=True)
    assert set(res.columns) == {"c", "a_b", "a_c", "b_0.0"}

    a_b = res["a_b"].values.compute()
    a_c = res["a_c"].values.compute()

    expected_b = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 1])
    expected_c = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(a_b, expected_b)
    np.testing.assert_array_equal(a_c, expected_c)


def test_one_hot_encoding_str_with_nan():
    ddf = make_fake_data(to_pandas=True)
    ddf.loc[0, "a"] = np.nan
    ddf = dd.from_pandas(ddf, npartitions=2)
    res = indicators._one_hot_encode_dd(ddf, cols=["a", "b"])
    assert set(res.columns) == {"c", "a_b", "a_a", "a_c", "b_1.0", "b_0.0"}

    a_a = res["a_a"].values.compute()
    a_b = res["a_b"].values.compute()
    a_c = res["a_c"].values.compute()

    expected_a = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
    expected_b = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 1])
    expected_c = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(a_a, expected_a)
    np.testing.assert_array_equal(a_b, expected_b)
    np.testing.assert_array_equal(a_c, expected_c)


def test_one_hot_encoding_int():
    ddf = make_fake_data(to_pandas=False)
    res = indicators._one_hot_encode_dd(ddf, cols=["b"])
    # [1, 1, 0, 0, 1, 1, 0, 0, np.nan, 0]
    assert "b_1.0" in res.columns
    assert "b_0.0" in res.columns
    assert "b" not in res.columns

    b_0 = res["b_0.0"].values.compute()
    b_1 = res["b_1.0"].values.compute()

    expected_0 = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])
    expected_1 = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(b_0, expected_0)
    np.testing.assert_array_equal(b_1, expected_1)


def test_one_hot_encoding_int_limit_categories():
    cats = {"b": CategoricalDtype([1.0])}
    ddf = make_fake_data(to_pandas=False)
    res = indicators._one_hot_encode_dd(ddf, cols=["b"], categories=cats)
    # [1, 1, 0, 0, 1, 1, 0, 0, np.nan, 0]
    assert "b_1.0" in res.columns, str(res.columns)
    assert "b_0.0" not in res.columns
    assert "b_nan" not in res.columns, str(res.columns)
    b_1 = res["b_1.0"].values.compute()

    expected_1 = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(b_1, expected_1)


@pytest.mark.parametrize(
    "cats, drop",
    [
        (None, False),
        (None, True),
        ({"b": CategoricalDtype([1.0])}, False),
        ({"b": CategoricalDtype([1.0, 0.0]), "a": CategoricalDtype(["a", "b"])}, False),
        ({"b": CategoricalDtype([1.0, 0.0]), "a": CategoricalDtype(["a", "b"])}, True),
        ({"c": CategoricalDtype([0.8011, 0.2202, 0.777])}, False),
        ({"c": CategoricalDtype([0.8011, 0.2202, 0.777])}, True),
    ],
)
def test_ohe_consistency(cats, drop):
    df = make_fake_data(to_pandas=True)
    ddf = make_fake_data(to_pandas=False)
    cols = list(cats.keys()) if cats else list("abc")
    res_dd = indicators.one_hot_encode(ddf, cols=cols, categories=cats, drop_first=drop).compute()
    res_pd = indicators.one_hot_encode(df, cols=cols, categories=cats, drop_first=drop)
    assert len(res_pd.columns) == len(
        res_dd.columns
    ), f"{cats}, drop={drop}\n\nlen({res_pd.columns}) != len({res_dd.columns})\n\n"
    if not drop:
        # can't predict which col will be dropped, so
        # best we can do is check columns length
        cols = res_pd.columns
        pd.testing.assert_frame_equal(res_dd[cols], res_pd[cols])


@pytest.mark.parametrize(
    "cats",
    [
        {"b": CategoricalDtype([1.0])},
        {"b": CategoricalDtype([1.0]), "a": CategoricalDtype(["a", "b"])},
    ],
)
def test_ohe_fails_drop(cats):
    df = make_fake_data(to_pandas=True)
    ddf = make_fake_data(to_pandas=False)
    cols = list(cats.keys()) if cats else list("abc")
    drop = True
    with pytest.raises(Exception):
        _ = indicators.one_hot_encode(ddf, cols=cols, categories=cats, drop_first=drop).compute()
    with pytest.raises(Exception):
        _ = indicators.one_hot_encode(df, cols=cols, categories=cats, drop_first=drop)


def test_ohe_consistency_():
    df = make_fake_data(to_pandas=True)
    ddf = make_fake_data(to_pandas=False)
    cats = {"b": CategoricalDtype([1.0, 0.0]), "a": CategoricalDtype(["a", "b"])}
    drop = True
    cols = list(cats.keys())
    res_dd = indicators.one_hot_encode(ddf, cols=cols, categories=cats, drop_first=drop).compute()
    res_pd = indicators.one_hot_encode(df, cols=list("ab"), categories=cats, drop_first=drop)
    assert len(res_pd.columns) == len(res_dd.columns)
    if not drop:
        # can't predict which col will be dropped, so
        # best we can do is check columns length
        cols = res_pd.columns
        pd.testing.assert_frame_equal(res_dd[cols], res_pd[cols])


def test_one_hot_encoding_int_limit_categories_drop_first():
    cats = {"b": CategoricalDtype([1.0, 0.0]), "a": CategoricalDtype(["a", "b"])}
    ddf = make_fake_data(to_pandas=False)
    res = indicators._one_hot_encode_dd(
        ddf, cols=["a", "b"], categories=cats, drop_first=True
    ).compute()
    # [1, 1, 0, 0, 1, 1, 0, 0, np.nan, 0]
    assert set(res.columns) == {"c", "a_b", "b_0.0"}


def test_one_hot_encoding_pd():
    ddf = make_fake_data(to_pandas=True)
    res = indicators._one_hot_encode_pd(ddf, cols=["a", "b"])
    assert "a_a" in res.columns
    assert "a_b" in res.columns
    assert "a_c" in res.columns

    a_a = res["a_a"].values
    a_b = res["a_b"].values
    a_c = res["a_c"].values

    expected_a = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 0])
    expected_b = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 1])
    expected_c = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(a_a, expected_a)
    np.testing.assert_array_equal(a_b, expected_b)
    np.testing.assert_array_equal(a_c, expected_c)


@pytest.fixture
def get_indicator_pd_setup():
    obj = indicators.AddIndicators(feats=["a", "b", "c"])
    X = make_fake_data(to_pandas=True)
    obj.get_sample_df(X, random_state=0, min_rows=1)
    obj.feat_type_dct, obj.feat_class_dct = get_types_classes_for_fake_data()
    return obj, X


@pytest.fixture
def get_indicator_dd_setup():
    obj = indicators.AddIndicators(feats=["a", "b", "c"], fail_on_warning=False, sample_rate=None)
    X = make_fake_data(to_pandas=False)
    obj.get_sample_df(X, random_state=0, min_rows=1)
    obj.feat_type_dct, obj.feat_class_dct = get_types_classes_for_fake_data()
    return obj, X


def test_set_defaults_pd(get_indicator_pd_setup):
    obj, X = get_indicator_pd_setup
    obj._set_defaults(X)


def test_set_defaults_dd():
    obj = indicators.AddIndicators(feats=["a", "b", "c"])
    X = make_fake_data(to_pandas=False)
    obj._set_defaults(X)


def test_process_feats():
    from cleaners import eda

    obj = indicators.AddIndicators(
        feats=["a", "b", "c"], unique_thresh=1, sample_rate=None, fail_on_warning=False
    )
    X = make_fake_data(to_pandas=False)
    obj.get_sample_df(X, random_state=0)

    feat_type_dct, feat_class_dct = eda.process_feats(
        obj.sample_df, unique_thresh=obj.unique_thresh, feats=obj.feats
    )

    assert feat_class_dct == {"a": "categorical", "b": "binary", "c": "continuous"}, "{}".format(
        feat_class_dct
    )
    assert feat_type_dct == {"a": "object", "b": "numeric", "c": "numeric"}, "{}".format(
        feat_type_dct
    )


def test_get_ohe_cols_pd(get_indicator_pd_setup):
    obj, X = get_indicator_pd_setup
    obj.get_ohe_cols(X)
    assert obj.ohe_cols == ["a", "b"], "{}".format(obj.ohe_cols)


def test_get_ohe_cols_dd(get_indicator_dd_setup):
    obj, X = get_indicator_dd_setup
    obj.get_ohe_cols(X)
    assert obj.ohe_cols == ["a", "b"], "{}".format(obj.ohe_cols)


def test_get_ohe_assertionerror(get_indicator_dd_setup):
    obj, X = get_indicator_dd_setup
    obj.ohe_cols = ["c", "not", "in", "data"]
    with pytest.raises(KeyError):
        obj.get_ohe_cols(X)


def test_get_cont_na_feats_pd(get_indicator_pd_setup):
    obj, X = get_indicator_pd_setup
    obj.get_cont_na_feats(X)
    assert obj.cont_na_feats == ["c"]


def test_get_cont_na_feats_dd(get_indicator_dd_setup):
    obj, X = get_indicator_dd_setup
    obj.get_cont_na_feats(X)
    assert obj.cont_na_feats == ["c"]


def test_get_cont_na_feats_assertionerror(get_indicator_dd_setup):
    obj, X = get_indicator_dd_setup
    obj.cont_na_feats = ["c", "not", "in", "data"]
    with pytest.raises(KeyError):
        obj.get_cont_na_feats(X)


@pytest.mark.parametrize("copy_it", [True, False])
def test_encode_nan_copy_pd(get_indicator_pd_setup, copy_it):
    obj, X = get_indicator_pd_setup
    col = "c"
    exp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    res = indicators.encode_nans(X, col, new_col="c_nan", copy_data=copy_it)
    np.testing.assert_array_equal(res["c_nan"].values, exp)


@pytest.mark.parametrize("copy_it", [True, False])
def test_encode_nan_copy_dd(get_indicator_dd_setup, copy_it):
    obj, X = get_indicator_dd_setup
    col = "c"
    exp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    res = indicators.encode_nans(X, col, new_col="c_nan", copy_data=copy_it).compute()
    np.testing.assert_array_equal(res["c_nan"].values, exp)


def test_encode_nan_pd_fails_on_dupe_column_name(get_indicator_pd_setup):
    obj, X = get_indicator_pd_setup
    col = "c"
    X["c_nan"] = np.ones(10)
    with pytest.raises(AssertionError):
        _ = indicators.encode_nans(X, col, new_col="c_nan")


@pytest.mark.parametrize(
    "col, exp", [("b", [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), ("c", [0, 0, 0, 0, 0, 0, 0, 0, 1, 0])]
)
def test_encode_nan_pd(col, exp, get_indicator_pd_setup):
    obj, X = get_indicator_pd_setup
    new_col = f"{col}_nan"

    res = indicators.encode_nans(X, col, new_col=new_col)
    np.testing.assert_array_equal(res[new_col].values, np.array(exp))


@pytest.mark.parametrize(
    "col, exp", [("b", [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), ("c", [0, 0, 0, 0, 0, 0, 0, 0, 1, 0])]
)
def test_encode_nan_dd(col, exp, get_indicator_dd_setup):
    obj, X = get_indicator_dd_setup
    new_col = f"{col}_nan"

    res = indicators.encode_nans(X, col, new_col=new_col).compute()
    np.testing.assert_array_equal(res[new_col].values, np.array(exp))


def test_encode_nan_dd_copy(get_indicator_dd_setup):
    obj, X = get_indicator_dd_setup
    new_col = "b_nan"
    col = "b"

    res = indicators.encode_nans(X, col, new_col=new_col, copy_data=True).compute()
    assert new_col not in X.columns
    assert new_col in res.columns

    res = indicators.encode_nans(X, col, new_col=new_col, copy_data=False).compute()
    assert new_col in X.columns
    assert new_col in res.columns


def test_filter_categories():
    col = "a"
    dummy_cols = ["a_1", "a_2", "a_3", "a_4", "b_3"]
    cat = CategoricalDtype([1, 2, 3]).categories.tolist()
    res = indicators._filter_categories(dummy_cols, col, cat)
    assert res == ["a_1", "a_2", "a_3"]


def test_make_ind_pd():
    X = make_fake_data(to_pandas=True)
    res = indicators._make_dummy_cols(
        X,
        expected_dummies=("x_y", "y_z", "x_z"),
        added_indicators=None,
        cols=None,
        category_dct=None,
        drop_first=False,
    )

    for col in ["x_y", "y_z", "x_z"]:
        assert col in res.columns
        np.testing.assert_array_equal(res[col].values, np.zeros(res.index.size))


def test_make_ind_dd():
    X = make_fake_data(to_pandas=False)
    res = indicators._make_dummy_cols(
        X,
        expected_dummies=("x_y", "y_z", "x_z"),
        added_indicators=None,
        cols=None,
        category_dct=None,
        drop_first=False,
    ).compute()

    for col in ["x_y", "y_z", "x_z"]:
        assert col in res.columns
        np.testing.assert_array_equal(res[col].values, np.zeros(res.index.size))


def test_make_ind_added_cols():
    added_cols = ["test_col"]
    X = make_fake_data(to_pandas=True)
    res = indicators._make_dummy_cols(X, added_indicators=added_cols, cols=["a"])
    assert set(added_cols) == {"test_col", "a_a", "a_b", "a_c"}
