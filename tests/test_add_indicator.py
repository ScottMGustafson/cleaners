import pytest

from cleaners.indicators import *
from tests.make_data import get_types_classes_for_fake_data, make_fake_data


def test_one_hot_encoding():
    ddf = make_fake_data(to_pandas=False)
    res = one_hot_encoding(ddf, cols=["a", "b"])
    assert "a_a" in res.columns
    assert "a_b" in res.columns
    assert "a_c" in res.columns

    a_a = res["a_a"].values.compute()
    a_b = res["a_b"].values.compute()
    a_c = res["a_c"].values.compute()

    expected_a = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 0])
    expected_b = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 1])
    expected_c = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(a_a, expected_a)
    np.testing.assert_array_equal(a_b, expected_b)
    np.testing.assert_array_equal(a_c, expected_c)


def test_one_hot_encoding_pd():
    ddf = make_fake_data(to_pandas=True)
    res = one_hot_encode_pd(ddf, cols=["a", "b"])
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
    obj = AddIndicators(feats=["a", "b", "c"])
    X = make_fake_data(to_pandas=True)
    obj.get_sample_df(X, random_state=0, min_rows=1)
    obj.feat_type_dct, obj.feat_class_dct = get_types_classes_for_fake_data()
    return obj, X


@pytest.fixture
def get_indicator_dd_setup():
    obj = AddIndicators(feats=["a", "b", "c"], fail_on_warning=False, sample_rate=None)
    X = make_fake_data(to_pandas=False)
    obj.get_sample_df(X, random_state=0, min_rows=1)
    obj.feat_type_dct, obj.feat_class_dct = get_types_classes_for_fake_data()
    return obj, X


def test_set_defaults_pd(get_indicator_pd_setup):
    obj, X = get_indicator_pd_setup
    obj._set_defaults(X)


def test_set_defaults_dd():
    obj = AddIndicators(feats=["a", "b", "c"])
    X = make_fake_data(to_pandas=False)
    obj._set_defaults(X)


def test_process_feats():
    from cleaners import eda

    obj = AddIndicators(
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
    with pytest.raises(AssertionError):
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
    with pytest.raises(AssertionError):
        obj.get_cont_na_feats(X)


def test_encode_nan_pd_fails_on_dask(get_indicator_dd_setup):
    obj, X = get_indicator_dd_setup
    col = "c"
    with pytest.raises(TypeError):
        _ = encode_nan_columns_pd(X, col, new_col=None)


def test_encode_nan_pd_fails_on_dupe_column_name(get_indicator_pd_setup):
    obj, X = get_indicator_pd_setup
    col = "c"
    X["c_nan"] = np.ones(10)
    with pytest.raises(AssertionError):
        _ = encode_nan_columns_pd(X, col, new_col="c_nan")


@pytest.mark.parametrize(
    "col, exp", [("b", [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), ("c", [0, 0, 0, 0, 0, 0, 0, 0, 1, 0])]
)
def test_encode_nan_pd(col, exp, get_indicator_pd_setup):
    obj, X = get_indicator_pd_setup
    new_col = f"{col}_nan"

    res = encode_nan_columns_pd(X, col, new_col=new_col)
    np.testing.assert_array_equal(res[new_col].values, np.array(exp))


@pytest.mark.parametrize(
    "col, exp", [("b", [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), ("c", [0, 0, 0, 0, 0, 0, 0, 0, 1, 0])]
)
def test_encode_nan_dd(col, exp, get_indicator_dd_setup):
    obj, X = get_indicator_dd_setup
    new_col = f"{col}_nan"

    res = encode_nan_columns_dd(X, col, new_col=new_col).compute()
    np.testing.assert_array_equal(res[new_col].values, np.array(exp))


def test_encode_nan_dd_copy(get_indicator_dd_setup):
    obj, X = get_indicator_dd_setup
    new_col = "b_nan"
    col = "b"

    res = encode_nan_columns_dd(X, col, new_col=new_col, copy=True).compute()
    assert new_col not in X.columns
    assert new_col in res.columns

    res = encode_nan_columns_dd(X, col, new_col=new_col, copy=False).compute()
    assert new_col in X.columns
    assert new_col in res.columns


def test_make_nan_ind_columns(get_indicator_pd_setup):
    obj, X = get_indicator_pd_setup
    obj.added_indicator_columns = ["z", "y"]
    new_data = obj.make_nan_indicator_columns(X, "b", "b_nan")
    assert obj.added_indicator_columns == ["z", "y", "b_nan"]
    assert "b_nan" in new_data.columns
