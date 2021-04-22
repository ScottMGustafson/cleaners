"""
Utilities for EDA.

Some clarification on data types versus classes I use here:
I use ``classes`` as a descriptor on functionality of the data
whereas ``types`` refers to the actual type of the data

Examples
---------
 - data like [1,0,0,1,nan] would be classified as binary class, numeric type
 - data like [a,b,b,a,nan] would be classified as binary class, object type
 - data like [1.2, 1.0, 1.2, 1.3, 1.23] would be continuous class, numeric type
 - data like [a,b,c,d,a,b] would be categorical class, object type
 - data like [1.2, 1.0, 1.2, 1.3, 1.0] would be categorical class, numeric type **IF**
 ``unique_thresh > 3/5`` , since there were three unique values
  **but** it would be a continuous class, numeric type if ``unique_thresh <= 3/5``

See Also
--------
https://github.com/ScottMGustafson/feets/blob/master/feets/explore.py
"""

import numpy as np
import pandas as pd
from feets import explore


def fail_on_dask(func):
    """Check the pandas-ness of data."""

    def _fail_on_dask(*args, **kwargs):
        """Assumes that first arg is data and asserts that it is pandas."""
        df = args[0]
        assert isinstance(
            df, pd.DataFrame
        ), "Only pandas dataframes supported here. Got type {} instead.".format(type(df))
        return func(*args, **kwargs)

    return _fail_on_dask


@fail_on_dask
def get_describe_df(df, unique_thresh=6, percentiles=None):
    """Get a descriptive dataframe on each feature."""
    if not percentiles:
        percentiles = [0.01, 0.25, 0.5, 0.75, 0.99]
    feat_type_dct, feat_class_dct = process_feats(df, unique_thresh=unique_thresh)
    descr = df.describe(percentiles=percentiles).T
    pd.options.display.float_format = "{:,.2f}".format
    for col in df.columns:
        descr.loc[col, "dtype"] = df[col].dtype
        descr.loc[col, "nan_rate"] = df[col].isna().sum() / df[col].index.size
        vc = df[col].dropna().value_counts()
        descr.loc[col, "inferred_kind"] = feat_class_dct[col]
        descr.loc[col, "inferred_type"] = feat_type_dct[col]
        descr.loc[col, "n_unique"] = vc.size
        descr.loc[col, "zero_rate"] = (df[col] == 0.0).sum() / df[col].index.size
        try:
            descr.loc[col, "most_frequent"] = vc.index.values[0]
        except (IndexError, KeyError):
            descr.loc[col, "most_frequent"] = "NaN"

    descr = descr.drop(columns="count").replace(np.nan, "")

    front = [
        "dtype",
        "inferred_kind",
        "inferred_type",
        "nan_rate",
        "zero_rate",
        "n_unique",
        "most_frequent",
    ]
    cols = [x for x in descr.columns if x not in front]
    return descr[front + cols]


@fail_on_dask
def get_high_corr_cols(df, rho_thresh, method="spearman"):
    """Get high-correlation columns."""
    corr_matrix = df.corr(method=method).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    return [column for column in upper.columns if any(upper[column] > rho_thresh)]


@fail_on_dask
def get_unary_columns(df):
    """Get columns with single value."""
    return [x for x in df.columns if len(df[x].unique()) == 1]


@fail_on_dask
def get_mostly_nan(df, thresh=0.05):
    """Get columns with nan-rate greater than ``1.0 - thresh``."""
    sz = df.index.size
    return [x for x in df.columns if df[x].dropna().size / sz < thresh]


@fail_on_dask
def get_binary(df):
    """Get binary columns, excluding nans."""
    return [x for x in df.columns if len(df[x].dropna().value_counts()) == 2]


@fail_on_dask
def get_mostly_value(df, value, thresh=0.05):
    """Get columns with occurrences of ``value`` occurring more than ``thresh``."""
    sz = df.index.size
    return [x for x in df.columns if df[df[x] == value][x].size / sz > thresh]


@fail_on_dask
def get_categorical(df, thresh=20):
    """
    Get columns with fewer than ``thresh`` unique values.

    Parameters
    ----------
    df : pd.Dataframe
        data to test
    thresh : int
        max number of unique occurrences for numeric data to be considered categorical

    Returns
    -------
    list
        list of categorical columns
    """
    # exclude binary values...
    binary_lst = get_binary(df)
    cat_list = []
    for x in df.columns:
        if x in binary_lst:
            continue
        if df[x].dtype in ["object", "str"]:
            cat_list.append(x)
        else:
            if df[x].value_counts().size <= thresh:
                cat_list.append(x)

    return cat_list


def get_type_lst(feat_type_dct, feat_type, exclude_lst):
    """Get all columns of type ``feat_type``."""
    return [k for k, v in feat_type_dct.items() if v == feat_type and k not in exclude_lst]


def find_relevant_columns(lst, model_feats, mandatory):
    """Find items where item is in model_feats or mandatory_feats cols."""
    return [x for x in lst if any(x in y for y in model_feats + mandatory)]


@fail_on_dask
def process_feats(df, unique_thresh=0.01, feats=None):
    """
    Analyze features in dataframe and return inferred information.

    Parameters
    ----------
    df : pd.DataFrame
    unique_thresh : int
    feats : list, default=None

    Returns
    -------
    (dict, dict)
        tuple of (feat types, feat classes)
    """
    if any(df.columns.duplicated()):
        lst = df.loc[:, df.columns.duplicated()].columns.tolist()
        raise Exception(f"The following columns are duplicated: {sorted(set(lst))}")

    if not feats:
        feats = df.columns.tolist()
    feat_type_dct = explore.classify_feature_types(df, feats=feats)

    feat_class_dct = {
        k: explore.classify_value_counts(df, k, unique_thresh=unique_thresh, type_dct=feat_type_dct)
        for k in feats
    }
    return feat_type_dct, feat_class_dct


def get_uninformative(feat_class_dct, mandatory=()):
    """Get uninformative columns, where column us all one value or none."""
    return [
        x
        for x, v in feat_class_dct.items()
        if v in ["null", "uninformative"] and x not in mandatory
    ]
