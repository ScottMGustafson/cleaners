import numpy as np
import pytest

from cleaners.indicators import (AddIndicators, one_hot_encode_pd,
                                 one_hot_encoding)
from tests.make_data import make_fake_data


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
    return obj, X


@pytest.fixture
def get_indicator_dd_setup():
    obj = AddIndicators(feats=["a", "b", "c"], fail_on_warning=False)
    X = make_fake_data(to_pandas=False)
    return obj, X


def test_set_defaults_pd(get_indicator_pd_setup):
    obj, X = get_indicator_pd_setup
    obj._set_defaults(X)


def test_set_defaults_dd():
    obj = AddIndicators(feats=["a", "b", "c"])
    X = make_fake_data(to_pandas=False)
    obj._set_defaults(X)


def test_it():
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


# def test_get_ohe_cols_pd(get_indicator_pd_setup):
#     obj, X = get_indicator_pd_setup
#
#
#     obj.feat_type_dct, obj.feat_class_dct = {}, {}
#     obj.get_ohe_cols(X)
#     assert obj.ohe_cols == ["a", "b"], "{}".format(obj.ohe_cols)

# def test_get_ohe_cols(get_indicator_dd_setup):
#     obj, X = get_indicator_dd_setup
#     obj.feat_type_dct, obj.feat_class_dct = {}, {}
#     obj.get_ohe_cols(X)
#     assert obj.ohe_cols == ["a", "b"], "{}".format(obj.ohe_cols)
#
#
# def test_get_cont_na_feats(get_indicator_pd_setup):
#     obj, X = get_indicator_pd_setup
#     obj.feat_type_dct, obj.feat_class_dct = {}, {}
#     obj.get_cont_na_feats(X)
