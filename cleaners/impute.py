"""
Imputation utilities.

Really the only value add here is an ever-so-slightly easier way to
manage diff behavior for diff columns.  If this isn't a big deal,
just use dask_ml.impute.SimpleImputer directly.
"""

from dask_ml.impute import SimpleImputer
from numpy import nan

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

    def __init__(
        self,
        missing_values=nan,
        strategy="mean",
        add_indicator=False,
        copy=True,
        cols=None,
        fill_value=None,
        **kwargs,
    ):
        """
        Impute with dask_ml simple imputer.

        Parameters
        ----------
        cols : list, default=None
            columns to operate on.  If none, then impute on all available columns
        missing_values : str or numeric, default=nan
        strategy: str, default="mean"
            strategy for imputation. Allowed values are:
            ``["mean", "median", "most_frequent", "constant"]``
        fill_value : str or numeric, default=none
        copy: boolean, default=True
        add_indicator: boolean, default=True
        """
        super(ImputeByValue, self).__init__(**kwargs)
        self.imputer = SimpleImputer(
            missing_values=missing_values,
            strategy=strategy,
            fill_value=fill_value,
            verbose=bool(kwargs.get("verbose")),
            copy=copy,
            add_indicator=add_indicator,
        )
        self.cols = cols

    def fit(self, X, y=None):
        """Fit method."""
        if self.cols:
            self.imputer = self.imputer.fit(X[self.cols], y)
        else:
            self.imputer = self.imputer.fit(X, y)
        return self

    def transform(self, X):
        """Transform method."""
        if self.cols:
            return self.imputer.transform(X[self.cols])
        else:
            return self.imputer.transform(X)
