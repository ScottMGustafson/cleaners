"""
Imputation utilities.

Really the only value add here is an ever-so-slightly easier way to
manage diff behavior for diff columns.  If this isn't a big deal,
just use dask_ml.impute.SimpleImputer directly.
"""

import dask.dataframe as dd
import pandas as pd
from sklearn.impute import SimpleImputer

from cleaners.cleaner_base import CleanerBase


class ImputeByValue(CleanerBase):
    """
    Impute using dask_ml / sklearn SimpleImputer.

    Attributes
    ----------
    imputer : dask_ml.impute.SimpleImputer
        Simple Imputer instance
    cols : list
        list of columns to impute by imputer's strategy

    See Also
    --------
    https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
    """

    def __init__(self, cols=None, allow_passthrough=True, imputer_kwargs=None, **kwargs):
        """
        Impute with dask_ml simple imputer.

        Parameters
        ----------
        cols : list, default=None
            columns to operate on.  If none, then impute on all available columns
        """
        super().__init__(**kwargs)
        self.cols = cols
        self.allow_passthrough = allow_passthrough
        self.imputer_kwargs = imputer_kwargs or dict(
            copy=True, add_indicator=True, strategy="constant", fill_value=-999999
        )

        self.imputer_ = SimpleImputer(**self.imputer_kwargs)
        self.cols = cols

    def fit(self, X, y=None, **kwargs):  # noqa: D102
        if not self.cols:
            self.cols = X.columns.tolist()

        if isinstance(X, dd.DataFrame):
            # sample to get pd info
            self.get_sample_df(X)
            self.imputer_ = self.imputer_.fit(self.sample_df[self.cols])
        elif isinstance(X, pd.DataFrame):
            self.imputer_ = self.imputer_.fit(X[self.cols], y)
        else:
            raise TypeError(f"not supported for type: {type(X)}")

        self.feature_names_in_ = self.cols
        self.feature_names_out_ = self.imputer_.get_feature_names_out(
            input_features=self.feature_names_in_
        )

        return self

    def _transform_pd(self, X):
        """Transform method."""
        X_extra = X[[c for c in X.columns if c not in self.feature_names_in_]]
        X_tr = pd.DataFrame(
            self.imputer_.transform(X[self.feature_names_in_]),
            columns=self.feature_names_out_,
            index=X.index,
        )
        if self.allow_passthrough:
            return pd.concat([X_tr, X_extra], axis=1)
        else:
            return X_tr

    def _transform_dd(self, X):
        return X.map_partitions(self._transform_pd)

    def transform(self, X):  # noqa: D102
        self._check_input_features(X)
        if isinstance(X, pd.DataFrame):
            return self._transform_pd(X)
        elif isinstance(X, dd.DataFrame):
            return self._transform_dd(X)
        else:
            raise TypeError(f"not supported for type: {type(X)}")

    def get_feature_names_out(self, input_features=None):  # noqa: D102
        return self.imputer_.get_feature_names_out(input_features)
