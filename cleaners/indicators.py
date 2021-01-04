"""Add Indicators to either build or scoring data without imputation."""

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask_ml.preprocessing import Categorizer, DummyEncoder

from cleaners import eda
from cleaners.cleaner_base import CleanerBase
from cleaners.util import assert_no_duplicate_columns


class AddIndicators(CleanerBase):
    """
    Add indicators to data without imputing missings.

    Attributes
    ----------
    added_indicator_columns : list
        indicators added during build
    expected_indicator_columns : list
        expected indicators during scoring
    """

    def __init__(self, unique_thresh=6, ignore=("target", "date", "symbol"), **kwargs):
        """
        Add indicators to data without imputing missings.

        Parameters
        ----------
        unique_thresh : int, default=6
        ignore : iterable, default=``("target", "date", "symbol")``
        """
        super(AddIndicators, self).__init__(**kwargs)
        self.unique_thresh = unique_thresh
        self.ignore = ignore
        self.feats = kwargs.get("feats", [])
        self.feat_type_dct = kwargs.get("feat_type_dct")
        self.feat_class_dct = kwargs.get("feat_class_dct")
        self.ohe_cols = kwargs.get("ohe_cols")
        self.cont_na_feats = kwargs.get("cont_na_feats")
        self.expected_indicator_columns = kwargs.get("expected_indicator_columns", [])
        self.scoring = bool(self.expected_indicator_columns)
        self.added_indicator_columns = []
        self.impute_value = kwargs.get("impute_value", -999)
        self.category_dict = kwargs.get("category_dict")
        self.drop_first = kwargs.get("drop_first")

    def _set_defaults(self, X):
        if not self.feats:
            self.feats = [x for x in X.columns if x not in self.ignore]

        self.get_sample_df(X, random_state=0)

        if not self.feat_class_dct:
            self.feat_type_dct, self.feat_class_dct = eda.process_feats(
                self.sample_df, unique_thresh=self.unique_thresh, feats=self.feats
            )
        self.get_ohe_cols(self.sample_df)
        self.get_cont_na_feats(self.sample_df)

        msg = f"ohe_cols and continuous cols overlap: {self.cont_na_feats}, {self.ohe_cols}"
        assert len(set(self.ohe_cols + self.cont_na_feats)) == len(self.ohe_cols) + len(
            self.cont_na_feats
        ), msg

    def get_ohe_cols(self, X):
        """Determine which columns should be one-hot-encoded."""
        if not self.ohe_cols:
            self.ohe_cols = list(
                sorted(
                    set(
                        eda.get_type_lst(self.feat_class_dct, "categorical", self.ignore)
                        + eda.get_type_lst(self.feat_class_dct, "binary", self.ignore)
                        + eda.get_type_lst(self.feat_type_dct, "object", self.ignore)
                    )
                )
            )
        assert all([col in X.columns for col in self.ohe_cols]), "not all cols in data: {}".format(
            self.ohe_cols
        )

    def get_cont_na_feats(self, X):
        """Get continuous features, in which we might care about NaNs."""
        if not self.cont_na_feats:
            num_cols = eda.get_type_lst(self.feat_type_dct, "numeric", self.ignore)
            self.cont_na_feats = [x for x in num_cols if self.feat_class_dct[x] == "continuous"]
        assert all(
            [col in X.columns for col in self.cont_na_feats]
        ), "not all cols in data: {}".format(self.cont_na_feats)

    def make_nan_indicator_columns(self, X, col, new_col):
        """Make NaN indicator columns without imputing."""
        X = encode_nans(X, col=col, new_col=new_col, copy=False)
        self.added_indicator_columns.append(new_col)
        return X

    def make_dummy_cols(self, X, cols, expected_dummies=()):
        """
        Make Dummy columns.

        Parameters
        ----------
        X : dataframe
        cols : list
            list of subset columns
        expected_dummies: list, optional
            if present, will make columns, even if associated value wasn't present
            in the data.  In this case, it will pass the new columns as all zeros.

        Returns
        -------
        Dataframe
        """
        X = _make_dummy_cols(
            X,
            expected_dummies=expected_dummies,
            added_indicators=self.added_indicator_columns,
            cols=cols,
            category_dct=self.category_dict,
            drop_first=self.drop_first,
        )
        return X

    def scoring_transform(self, X):
        """Transform to be used when scoring new data on an existing model."""
        for col in self.cont_na_feats:
            new_col = f"{col}_nan"
            if new_col in self.expected_indicator_columns:
                X = self.make_nan_indicator_columns(X, col, new_col)

        expected_dummies = []
        for col in self.ohe_cols:
            assert col not in self.cont_na_feats, f"{col} already in cont_na_feats"
            expected_dummies.extend(
                [x for x in self.expected_indicator_columns if x.startswith(col + "_")]
            )
        X = self.make_dummy_cols(X, self.ohe_cols, expected_dummies=expected_dummies)
        return X

    def build_transform(self, X):
        """Transform to be run on model build/train."""
        for col in self.cont_na_feats:
            new_col = f"{col}_nan"
            X = self.make_nan_indicator_columns(X, col, new_col)

        for col in self.ohe_cols:
            assert col not in self.cont_na_feats, f"{col} already in cont_na_feats"
        X = self.make_dummy_cols(X, self.ohe_cols, expected_dummies=[])
        return X

    def transform(self, X):  # noqa: D102
        assert_no_duplicate_columns(X)
        if self.scoring:
            X = self.scoring_transform(X)
        else:
            self._set_defaults(X)
            X = self.build_transform(X)
        assert_no_duplicate_columns(X)
        return X


def _check_dummies(X, dummies, col):
    try:
        assert_no_duplicate_columns(dummies)
    except AssertionError:
        print("\n\ncolumn name: \n-----------------------------------", col)
        print("dummies: {}\n".format(sorted(dummies.columns.tolist())))
        print(
            "unique values already in data: {}\n-----------------------------------\n".format(
                X[col].unique()
            )
        )
        raise


def get_occur_frac(ddf, col, val=np.nan, sample_rate=0.1):
    """
    Estimates occurrence rates for data frame from sample_rate.

    Parameters
    ----------
    ddf : dataframe (dask or pd)
    col : str
        column anem
    val : Any
        value of which to check occurences
    sample_rate : float
        random sample_rate rate (0-1)

    Returns
    -------
    float
        occurrence rate
    """
    ser = ddf[col].sample(frac=sample_rate, random_state=0)
    vc = ser.value_counts()

    if hasattr(vc, "compute"):
        vc = vc.compute()
    if val not in vc.keys():
        return 0
    return vc[val] / vc.size


def encode_val(ddf, col, val, indicator_name=None, min_frac=None, sample_rate=0.1):
    """
    Encode special values in dataframe.

    Parameters
    ----------
    ddf : dataframe
    col : str
        column indicator_name
    val : Any
        value to encode
    indicator_name : str
        new indicator column indicator_name
    min_frac : float
        min occurrence rate.  if rate less than this value, do nothing.
    sample_rate : float
        sample_rate rate on data frame (random, row-wise) used to determine occurrence rate

    Returns
    -------
    None
    """
    if min_frac:
        frac = get_occur_frac(ddf, col, val=val, sample_rate=sample_rate)
        if frac < min_frac:
            return
    if not indicator_name:
        indicator_name = "{}_{}".format(col, val)
    ddf[indicator_name] = 0
    ddf[ddf[col] == val][indicator_name] = 1


def encode_lt_val(ddf, col, val, indicator_name=None, min_frac=None, sample_rate=0.1):
    """
    Encode if less than value.

    Parameters
    ----------
    ddf : dataframe
    col : str
        column indicator_name
    val : Any
        value to encode
    indicator_name : str
        new indicator column indicator_name
    min_frac : float
        min occurrence rate.  if rate less than this value, do nothing.
    sample_rate : float
        sample_rate rate on data frame (random, row-wise) used to determine occurrence rate

    Returns
    -------
    None
    """
    if min_frac:
        frac = get_occur_frac(ddf, col, val=val, sample_rate=sample_rate)
        if frac < min_frac:
            return
    if not indicator_name:
        indicator_name = "{}_lt_{}".format(col, val)
    ddf[indicator_name] = 0
    ddf[ddf[col] < val][indicator_name] = 1


def _validate_categories(categories):
    for k, v in categories.items():
        if len(v.categories) == 1:
            raise Exception(
                f"drop_first=True, {k}:{v}\n"
                + "``drop_first=True`` on a column with only one category specified. "
                + "This results in your only dummy column being dropped. Set to ``drop_first=False``."
            )


def _one_hot_encode_dd(ddf, cols, categories=None, drop_first=False):
    """
    One hot encode dask dataframes.

    Parameters
    ----------
    ddf : dd.DataFrame
    cols = list[str]
        list of column names
    categories : dict[pandas.api.CategoricalDtype], default=none
        mapping of cat_list to limit dummies.
    drop_first : bool, default=False
        drop first seen category

    Returns
    -------
    dd.DataFrame

    """
    ddf = Categorizer(columns=cols, categories=categories).fit_transform(ddf)
    ddf = DummyEncoder(columns=cols, drop_first=drop_first).fit_transform(ddf)

    return ddf


def _filter_categories(dummy_cols, col, cat_list):
    return [
        x for x in dummy_cols if x.split(f"{col}_")[-1] in map(str, cat_list) and x.startswith(col)
    ]


def _one_hot_encode_pd(df, cols, categories=None, drop_first=False):
    """
    One hot encode pandas dataframes.

    Parameters
    ----------
    df : pd.DataFrame
    cols = list[str]
        list of column names
    categories : dict[pandas.api.CategoricalDtype], default=none
        mapping of cat_list to limit dummies.
    drop_first : bool, default=False
        drop first seen category

    Returns
    -------
    pd.DataFrame

    """
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False, drop_first=False)
        dummy_cols = list(dummies.columns)
        if categories:
            try:
                dummy_cols = _filter_categories(
                    dummy_cols, col, categories[col].categories.tolist()
                )
            except KeyError:
                pass
        if drop_first:
            dummy_cols = dummy_cols[1:]
        dummies = dummies[dummy_cols]
        df = df.drop(columns=[col])
        df = pd.concat([df, dummies], axis=1)
    return df


def one_hot_encode(df, cols, categories=None, drop_first=False):
    """Apply One-hot-encoding to either dask for pandas dataframes."""
    if categories and drop_first:
        _validate_categories(categories)
    func = _one_hot_encode_dd if isinstance(df, dd.DataFrame) else _one_hot_encode_pd
    res = func(df, cols, drop_first=drop_first, categories=categories)
    return res


def encode_nans(X, col, new_col=None, copy=True):
    """Encode NaNs in a dataframe."""
    if not new_col:
        new_col = col + "_nan"
    assert new_col not in X.columns, f"AddIndicators::nan ind : {new_col} already exists in data"
    if isinstance(X, dd.DataFrame):
        ser = dd.map_partitions(pd.isna, X[col], meta=float)
    else:
        ser = pd.isna(X[col]).astype(float)

    ret_data = X.copy() if copy else X
    ret_data[new_col] = ser
    assert_no_duplicate_columns(ret_data)
    return ret_data


def _validate_category_dict(category_dct, cols):
    assert isinstance(category_dct, dict)
    for k, v in category_dct.items():
        assert k in cols, f"make_dummy_cols: {k} not in allowed columns: \n{cols}\n\n"
        if not isinstance(v, pd.CategoricalDtype):
            assert isinstance(
                v, list
            ), f"categories must be list or pd.CategoricalDtype. Got {type(v)}"
            category_dct[k] = pd.CategoricalDtype(list(set(v)))


def _make_dummy_cols(
    X, expected_dummies=(), added_indicators=None, cols=None, category_dct=None, drop_first=False
):
    """Make dummy columns for OHE without imputing nans."""
    if not added_indicators:
        added_indicators = []

    if not cols:
        cols = X.columns

    if category_dct:
        _validate_category_dict(category_dct, cols)

    assert all(
        [x in X.columns for x in cols]
    ), f"Not all requested columns in data: {[x for x in cols if x not in X.columns]}"
    old_cols = X.columns  # all columns currently in X
    X = one_hot_encode(X, cols, categories=category_dct, drop_first=drop_first)
    added_indicators.extend([x for x in X.columns if x not in old_cols])  # columns just added
    assert_no_duplicate_columns(X)

    for k in expected_dummies:
        if k not in X.columns:
            X[k] = 0.0
    return X
