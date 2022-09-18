"""Add Indicators to either build or scoring data without imputation."""

import pandas as pd

from cleaners import eda
from cleaners.cleaner_base import CleanerBase
from cleaners.util import assert_no_duplicate_columns


class AddIndicators(CleanerBase):
    """
    Add indicators to data without imputing missings.

    Parameters
    ----------
    unique_thresh : int, default=6
    allow_nan : bool
    raise_exception_on_unseen : bool
    values_to_indicate : dict
    ignore : list
    feat_type_dict : dict
    feat_class_dict : dict
    feats : list
    ohe_cols : list
    ohe_categories : dict

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
    ohe_categories : dict

    Attributes
    ----------
    added_indicator_columns : list
        indicators added during build
    expected_indicator_columns : list
        expected indicators during scoring
    """

    def __init__(
        self,
        unique_thresh=6,
        allow_nan=True,
        raise_exception_on_unseen=False,
        values_to_indicate=None,
        ignore=None,
        feat_type_dict=None,
        feat_class_dict=None,
        feats=None,
        ohe_cols=None,
        ohe_categories=None,
        **kwargs,
    ):
        super(AddIndicators, self).__init__(**kwargs)
        self.unique_thresh = unique_thresh
        self.ignore = ignore or []
        self.feats = feats or []
        self.values_to_indicate = values_to_indicate or {}
        self.feat_type_dict = feat_type_dict
        self.feat_class_dict = feat_class_dict
        self.ohe_cols = ohe_cols
        self.added_indicators_ = []
        self.expected_dummies_ = []
        self.allow_nan = allow_nan
        self.ohe_categories = ohe_categories or {}
        self.feature_names_out_ = []
        self.raise_exception_on_unseen = raise_exception_on_unseen

    def get_ohe_cols(self, X):
        """Determine which columns should be one-hot-encoded."""
        if not self.ohe_cols:
            self.ohe_cols = sorted(
                set(
                    eda.get_type_lst(self.feat_class_dict, "categorical", self.ignore)
                    + eda.get_type_lst(self.feat_class_dict, "binary", self.ignore)
                    + eda.get_type_lst(self.feat_type_dict, "object", self.ignore)
                )
            ) + list(self.values_to_indicate.keys())
        if any(col not in X.columns for col in self.ohe_cols):
            raise KeyError(f"not all cols in data: {self.ohe_cols}")

        self._set_ohe_categories()

    def _set_ohe_categories(self):
        for col in self.ohe_cols:
            if col in self.ohe_categories.keys():
                continue  # to not override user supplied categories
            if col in self.values_to_indicate.keys():
                self.ohe_categories[col] = list(self.values_to_indicate[col])
            else:
                unique_vals = self.sample_df[col].unique().tolist()
                if not self.allow_nan:
                    if pd.Series(unique_vals).isnull().any():
                        raise TypeError("NaNs detected. Please run imputer first")
                self.ohe_categories[col] = list(unique_vals)
            self.added_indicators_.extend([f"{col}_{val}" for val in self.ohe_categories[col]])

    def fit(self, X, y=None, **kwargs):  # noqa: D102
        self.feature_names_in_ = X.columns.tolist()
        if len(self.added_indicators_) != 0:
            raise ValueError("Added indicators has been set already.  Is this right?")
        if len(self.feats) == 0:
            self.feats = [x for x in X.columns if x not in self.ignore]
        else:
            self.feats = X.columns.tolist()

        self.get_sample_df(X, random_state=0)

        if not self.feat_class_dict or not self.feat_type_dict:
            self.feat_type_dict, self.feat_class_dict = eda.process_feats(
                self.sample_df, unique_thresh=self.unique_thresh, feats=self.feats
            )
        self.get_ohe_cols(self.sample_df[self.feats])
        self.feature_names_out_ = self.feature_names_in_ + self.added_indicators_
        return self

    def _make_dummy_cols(self, X):
        """
        Make dummy columns for OHE without imputing nans.

        Parameters
        ----------
        X : dataframe
        Returns
        -------
        dataframe
        """
        if not self.added_indicators_:
            self.added_indicators_ = []

        if not self.ohe_cols:
            self.ohe_cols = X.columns

        if not all([x in X.columns for x in self.ohe_cols]):
            raise ValueError(
                f"Not all requested columns in data: {[x for x in self.ohe_cols if x not in X.columns]}"
            )

        for k, values in self.ohe_categories.items():
            for value in list(values):
                _name = f"{k}_{values}"
                if self.raise_exception_on_unseen and _name not in self.added_indicators_:
                    raise Exception(f"Unseen value {value} in {k}")
                assert _name not in X.columns, f"Existing column {_name}"
                X[f"{k}_{value}"] = (X[k] == value).astype(int)
        return X

    def transform(self, X):  # noqa: D102
        assert_no_duplicate_columns(X)
        return self._make_dummy_cols(X)

    def get_feature_names_out(self, input_features=None):
        """
        Get feature names.

        Parameters
        ----------
        input_features : list

        Returns
        -------
        list
        """
        if input_features:
            if input_features != self.feature_names_in_:
                raise NotImplementedError(
                    "feature names out not implemented for partial feature lists."
                )
        if self.feature_names_out_ is None:
            raise ValueError("Feature names have not yet been computed.")
        else:
            return self.feature_names_out_
