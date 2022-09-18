import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from cleaners import indicators

from .make_data import get_types_classes_for_fake_data, make_fake_data


def test_one_hot_encoding_str():
    ddf = make_fake_data(to_pandas=False)
    with pytest.raises(TypeError):
        obj = indicators.AddIndicators(allow_nan=False).fit(ddf)
    obj = indicators.AddIndicators(raise_exception_on_unseen=False).fit(ddf.dropna(how="any"))
    expected_cols = sorted(["a", "b", "c", "a_b", "a_a", "a_c", "b_1.0", "b_0.0"])

    assert sorted(obj.feature_names_out_) == expected_cols

    res = obj.transform(ddf.dropna(how="any"))

    a_a = res["a_a"].values.compute()
    a_b = res["a_b"].values.compute()
    a_c = res["a_c"].values.compute()

    expected_a = np.array([1, 0, 1, 1, 0, 0, 1, 1, 0])
    expected_b = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1])
    expected_c = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(a_a, expected_a)
    np.testing.assert_array_equal(a_b, expected_b)
    np.testing.assert_array_equal(a_c, expected_c)


def test_one_hot_encoding_str_with_nan():
    ddf = make_fake_data(to_pandas=True)
    ddf.loc[0, "a"] = np.nan
    ddf = dd.from_pandas(ddf, npartitions=2)

    obj = indicators.AddIndicators(raise_exception_on_unseen=False, feats=["a", "b"]).fit(ddf)
    assert all(x in obj.ohe_categories["a"] for x in ["a", "b", "c", np.nan])

    res = obj.transform(ddf)
    assert all(x in res.columns for x in ["a", "a_b", "a_a", "a_c", "a_nan"])

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
    res = indicators.AddIndicators(raise_exception_on_unseen=False, feats=["b"]).fit_transform(ddf)
    # [1, 1, 0, 0, 1, 1, 0, 0, np.nan, 0]
    assert "b_1.0" in res.columns
    assert "b_0.0" in res.columns

    b_0 = res["b_0.0"].values.compute()
    b_1 = res["b_1.0"].values.compute()

    expected_0 = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])
    expected_1 = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(b_0, expected_0)
    np.testing.assert_array_equal(b_1, expected_1)


def test_ohe_consistency():
    df = make_fake_data(to_pandas=True)
    ddf = make_fake_data(to_pandas=False)

    res_dd = indicators.AddIndicators().fit_transform(ddf).compute()
    res_pd = indicators.AddIndicators().fit_transform(df)
    cols = res_pd.columns
    pd.testing.assert_frame_equal(res_dd[cols], res_pd[cols])


@pytest.fixture
def get_indicator_pd_setup():
    X = make_fake_data(to_pandas=True)
    obj = indicators.AddIndicators(feats=["a", "b", "c"], unique_thresh=1, sample_rate=None).fit(X)
    obj.feat_type_dct, obj.feat_class_dict = get_types_classes_for_fake_data()
    return obj, X


@pytest.fixture
def get_indicator_dd_setup():
    X = make_fake_data(to_pandas=False)
    obj = indicators.AddIndicators(feats=["a", "b", "c"], unique_thresh=1, sample_rate=None).fit(X)
    obj.feat_type_dct, obj.feat_class_dict = get_types_classes_for_fake_data()
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

    X = make_fake_data(to_pandas=False)
    obj = indicators.AddIndicators(feats=["a", "b", "c"], unique_thresh=1, sample_rate=None).fit(X)

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
    assert obj.ohe_cols == ["a", "b"], "{}".format(obj.ohe_cols)


def test_get_ohe_cols_dd(get_indicator_dd_setup):
    obj, X = get_indicator_dd_setup
    assert obj.ohe_cols == ["a", "b"], "{}".format(obj.ohe_cols)


def test_get_ohe_assertionerror(get_indicator_dd_setup):
    obj, X = get_indicator_dd_setup
    obj.ohe_cols = ["c", "not", "in", "data"]
    with pytest.raises(KeyError):
        obj.get_ohe_cols(X)
