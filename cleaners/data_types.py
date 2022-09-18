"""Utilities for data types."""

import pandas as pd

from cleaners.cleaner_base import CleanerBase


def _infer_type(ser, type_list=None):
    if not type_list:
        type_list = ["float64", "M8[us]"]
    for _type in type_list:
        try:
            _ = ser.astype(_type)
            return _type
        except (TypeError, ValueError):
            pass
    return "str"


def infer_data_types(df, type_list=None):
    """Infer data types by trial and error."""
    type_dct = {}
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Only pd dataframes supported here")
    for k in df.columns:
        type_dct[k] = _infer_type(df[k], type_list=type_list)
    return type_dct


class FixDTypes(CleanerBase):
    """Fix bad datatypes."""

    def __init__(self, type_lst=None, sample_frac=0.05, random_state=0, **kwargs):
        """
        Init FixDTypes.

        Parameters
        ----------
        type_lst : list
        sample_frac : float
        random_state : int
        """
        super(FixDTypes, self).__init__(sample_rate=sample_frac, **kwargs)
        self.type_lst = type_lst
        self.dtypes = None
        self.random_state = random_state

    def fit(self, X, y=None, **kwargs):  # noqa : D102
        self.get_sample_df(X)
        self.dtypes = infer_data_types(self.sample_df, type_list=self.type_lst)
        return self

    def transform(self, X):  # noqa : D102
        return X.astype(self.dtypes)
