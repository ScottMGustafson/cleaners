import numpy as np
import pandas as pd
from dask_ml.preprocessing import Categorizer, DummyEncoder
import dask.dataframe as dd
from cleaners import eda
from cleaners.cleaner_base import CleanerBase
from cleaners.util import assert_no_duplicate_columns


class AddIndicators(CleanerBase):
    """
    Attributes
    ----------
    added_indicator_columns : list
        indicators added during build
    expected_indicator_columns : list
        expected indicators during scoring
    """

    def __init__(self, unique_thresh=6, ignore=("target", "date", "symbol"), **kwargs):
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

    def fit(self, X, y=None):
        return self

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
        """Determine which columns should be one-hot-encoded"""
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
        num_cols = eda.get_type_lst(self.feat_type_dct, "numeric", self.ignore)
        if not self.cont_na_feats:
            self.cont_na_feats = [x for x in num_cols if self.feat_class_dct[x] == "continuous"]
        assert all(
            [col in X.columns for col in self.cont_na_feats]
        ), "not all cols in data: {}".format(self.cont_na_feats)

    def make_nan_indicator_columns(self, X, col, new_col):
        if new_col in X.columns:
            raise Exception(f"AddIndicators::nan ind : {new_col} already exists in data")

        if hasattr(X, "compute"):
            X[new_col] = pd.isna(X[col]).astype(float)
            X[col] = X[col].fillna(self.impute_value)
        else:
            X[new_col] = pd.isna(X[col]).astype(float)
            X[col] = X[col].fillna(self.impute_value)
        self.added_indicator_columns.append(new_col)
        assert_no_duplicate_columns(X)
        return X

    def make_dummy_cols(self, X, col, expected_dummies=()):
        try:
            if X[col].dtype in ["str", "object"]:
                X[col] = X[col].replace({"nan": np.nan})
            dummies = pd.get_dummies(X[col], prefix=col, dummy_na=True, drop_first=True)
        except Exception as e:
            msg = f"Problem with get_dummies on {col} with dtype={X[col].dtype}:\n\n"
            msg += str(e)
            raise Exception(msg)
        _check_dummies(X, dummies, col)
        for x in dummies.columns:
            assert x not in X.columns, f"AddIndicators::one hot : {x} already exists in data"
        for x in expected_dummies:
            # create a dummy column if it doesn't already exists.
            # needed to ensure that dataframes have all expected columns
            # between build and scoring data
            if x not in dummies.columns.tolist() + X.columns.tolist():
                dummies[x] = 0.0
        self.added_indicator_columns.extend(dummies.columns.tolist())
        X = X.drop(columns=[col])
        X = pd.concat([X, dummies], axis=1)
        assert_no_duplicate_columns(X)
        return X

    def scoring_transform(self, X):
        for col in self.cont_na_feats:
            new_col = f"{col}_nan"
            if new_col in self.expected_indicator_columns:
                X = self.make_nan_indicator_columns(X, col, new_col)

        for col in self.ohe_cols:
            assert col not in self.cont_na_feats, f"{col} already in cont_na_feats"
            expected_dummies = [
                x for x in self.expected_indicator_columns if x.startswith(col + "_")
            ]
            X = self.make_dummy_cols(X, col, expected_dummies=expected_dummies)

        return X

    def build_transform(self, X):
        for col in self.cont_na_feats:
            new_col = f"{col}_nan"
            X = self.make_nan_indicator_columns(X, col, new_col)

        for col in self.ohe_cols:
            assert col not in self.cont_na_feats, f"{col} already in cont_na_feats"
            X = self.make_dummy_cols(X, col, expected_dummies=[])

        return X

    def transform(self, X):
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
    encode special values in dataframe

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
    Encode if less than value

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

    """
    if min_frac:
        frac = get_occur_frac(ddf, col, val=val, sample_rate=sample_rate)
        if frac < min_frac:
            return
    if not indicator_name:
        indicator_name = "{}_lt_{}".format(col, val)
    ddf[indicator_name] = 0
    ddf[ddf[col] < val][indicator_name] = 1


def one_hot_encoding(ddf, cols):
    """apply one-hot-encoding on dask dfs"""
    ddf = Categorizer(columns=cols).fit_transform(ddf)
    ddf = DummyEncoder(columns=cols).fit_transform(ddf)
    return ddf


def one_hot_encode_pd(df, cols):
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
        df = df.drop(columns=[col])
        df = pd.concat([df, dummies], axis=1)
    return df
