"""Add Indicators to either build or scoring data without imputation."""

import copy

import dask.dataframe as dd
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

    def __init__(self, unique_thresh=6, **kwargs):
        """
        Add indicators to data without imputing missings.

        Parameters
        ----------
        unique_thresh : int, default=6

        Other Parameters
        ----------------
        feats : list
        feat_type_dict : dict
        feat_class_dict : dict
        ohe_cols : list
            columns that should get one-hot encoded.
        cont_na_feats : list
            continuous feats which will get nan indicators.
        expected_indicators : list
        impute_value : numeric
        ohe_categories : dict
        drop_first : bool
        """
        super(AddIndicators, self).__init__(**kwargs)
        self.unique_thresh = unique_thresh
        self.ignore = kwargs.get("ignore", [])
        self.feats = kwargs.get("feats", [])
        self.feat_type_dct = kwargs.get("feat_type_dct")
        self.feat_class_dct = kwargs.get("feat_class_dct")
        self.ohe_cols = kwargs.get("ohe_cols")
        self.cont_na_feats = kwargs.get("cont_na_feats")
        self.expected_indicator_columns = kwargs.get("expected_indicator_columns", [])
        self.scoring = bool(self.expected_indicator_columns)
        self.added_indicator_columns = []
        self.impute_value = kwargs.get("impute_value", -999)
        self.ohe_categories = kwargs.get("ohe_categories", dict())
        self.drop_first = kwargs.get("drop_first")

    def _set_defaults(self, X):
        if len(self.feats) == 0:
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
            self.ohe_cols = sorted(
                set(
                    eda.get_type_lst(self.feat_class_dct, "categorical", self.ignore)
                    + eda.get_type_lst(self.feat_class_dct, "binary", self.ignore)
                    + eda.get_type_lst(self.feat_type_dct, "object", self.ignore)
                )
            )
        if any(col not in X.columns for col in self.ohe_cols):
            raise KeyError(f"not all cols in data: {self.ohe_cols}")

        # self._set_ohe_categories()

    def get_cont_na_feats(self, X):
        """Get continuous features, in which we might care about NaNs."""
        if not self.cont_na_feats:
            num_cols = eda.get_type_lst(self.feat_type_dct, "numeric", self.ignore)
            self.cont_na_feats = [x for x in num_cols if self.feat_class_dct[x] == "continuous"]
        if any(col not in X.columns for col in self.cont_na_feats):
            raise KeyError("not all cols in data: {}".format(self.cont_na_feats))

    def scoring_transform(self, X):
        """Transform to be used when scoring new data on an existing model.

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
        for col in self.cont_na_feats:
            new_col = f"{col}_nan"
            if new_col in self.expected_indicator_columns:
                X = encode_nans(X, col=col, new_col=new_col, copy_data=False)

        expected_dummies = []
        for col in self.ohe_cols:
            assert col not in self.cont_na_feats, f"{col} already in cont_na_feats"
            expected_dummies.extend(
                [x for x in self.expected_indicator_columns if x.startswith(col + "_")]
            )
        res = _make_dummy_cols(
            X,
            expected_dummies=expected_dummies,
            added_indicators=self.added_indicator_columns,
            cols=self.ohe_cols,
            category_dct=self.ohe_categories,
            drop_first=self.drop_first,
        )
        return res

    def _set_ohe_categories(self):
        for col in self.ohe_cols:
            if col in self.ohe_categories.keys():
                continue  # to not override user supplied categories
            unique_vals = self.sample_df[col].dropna().unique()
            try:
                self.ohe_categories[col] = pd.CategoricalDtype(unique_vals)
            except ValueError:
                pass

    def build_transform(self, X):
        """Transform to be run on model build/train."""
        for col in self.cont_na_feats:
            new_col = f"{col}_nan"
            X = encode_nans(X, col=col, new_col=new_col, copy_data=False)
            self.added_indicator_columns.append(new_col)

        for col in self.ohe_cols:
            assert col not in self.cont_na_feats, f"{col} already in cont_na_feats"

        X = _make_dummy_cols(
            X,
            expected_dummies=[],
            added_indicators=self.added_indicator_columns,
            cols=self.ohe_cols,
            category_dct=self.ohe_categories,
            drop_first=self.drop_first,
        )

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
    except IndexError:
        raise IndexError(
            f"\n\ncolumn name: {col}\n-----------------------------------"
            + f"dummies: {sorted(dummies.columns.tolist())}\n"
            + f"unique values already in data: {X[col].unique()}\n-----------------------------------\n"
        )


def _validate_categories(categories):
    for k, v in categories.items():
        if len(v.categories) == 1:
            raise Exception(
                f"drop_first=True, {k}:{v}\n"
                + "``drop_first=True`` on a column with only one category specified. "
                + "This results in your only dummy column being dropped. Set to ``drop_first=False``."
            )


def _one_hot_encode_dd_(ddf, cols, categories=None, drop_first=False):
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


def _get_dummies(cat_ser, col, drop_first, categories):
    dummies = dd.get_dummies(cat_ser, prefix=col, dummy_na=False, drop_first=False)
    dummy_cols = list(dummies.columns)
    if categories:
        try:
            dummy_cols = _filter_categories(dummy_cols, col, categories[col].categories.tolist())
        except KeyError:
            pass
    if drop_first:  # to match dask behavior
        dummy_cols = dummy_cols[1:]
    return dummies[dummy_cols]


def _one_hot_encode_dd(df, cols, categories=None, drop_first=False):
    """
    One hot encode dask dataframes.

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
    dd.DataFrame

    """
    # this is causing a keyerror: remove for now
    subdf = df[cols].categorize(columns=cols)
    # for col in cols:
    #     dummies = dd.get_dummies(subdf[col], prefix=col, dummy_na=False, drop_first=False)
    for col in cols:
        cat_ser = subdf[col]
        # if not categories:
        #     cat_ser = cat_ser.cat.as_known()  # this is slow.
        dummies = _get_dummies(cat_ser, col, drop_first, categories)
        df = df.drop(columns=[col]).merge(dummies, left_index=True, right_index=True)
    return df


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
        dummies = _get_dummies(df[col], col, drop_first, categories)
        df = df.drop(columns=[col]).merge(dummies, left_index=True, right_index=True)
    return df


def one_hot_encode(df, cols, categories=None, drop_first=False):
    """Apply One-hot-encoding to either dask for pandas dataframes."""
    if categories and drop_first:
        _validate_categories(categories)
    func = _one_hot_encode_dd if isinstance(df, dd.DataFrame) else _one_hot_encode_pd
    res = func(df, cols, drop_first=drop_first, categories=categories)
    return res


def encode_nans(X, col, new_col=None, copy_data=True):
    """Encode NaNs in a dataframe."""
    if not new_col:
        new_col = col + "_nan"
    assert new_col not in X.columns, f"AddIndicators::nan ind : {new_col} already exists in data"
    if isinstance(X, dd.DataFrame):
        ser = dd.map_partitions(pd.isna, X[col], meta=float)
    else:
        ser = pd.isna(X[col]).astype(float)

    if copy_data:
        ret_data = copy.copy(X)
    else:
        ret_data = X
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
    """
    Make dummy columns for OHE without imputing nans.

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
    dataframe
    """
    if not added_indicators:
        added_indicators = []

    if not cols:
        cols = X.columns

    if category_dct:
        _validate_category_dict(category_dct, cols)
    if not all([x in X.columns for x in cols]):
        raise ValueError(
            f"Not all requested columns in data: {[x for x in cols if x not in X.columns]}"
        )
    old_cols = X.columns  # all columns currently in X
    X = one_hot_encode(X, cols, categories=category_dct, drop_first=drop_first)
    added_indicators.extend([x for x in X.columns if x not in old_cols])  # columns just added
    assert_no_duplicate_columns(X)

    for k in expected_dummies:
        if k not in X.columns:
            X[k] = 0.0
    return X
