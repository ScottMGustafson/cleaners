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

TYPE_MAPPING = [("numeric", "float64"), ("datetime", "M8[us]"), ("object", "object")]


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


def _validate_mappings(lst):
    allowed_types = set([x[0] for x in TYPE_MAPPING])
    diff = set(lst).difference(allowed_types)
    assert not diff, "{} not recognized type mappings: {}".format(diff, allowed_types)


def classify_feature_types(df, feats=None):
    """
    Get inferred feature types by trying df.astype.

    This will try to cast each type as either
        - numeric: encompassing, int, float and bool types or anything else that
            can be successfully cast as float
        - datetime: anything that pandas can successfully cast to datetime
        - object: anything else, typically treated as a string

    Parameters
    ----------
    df : dataframe
    feats : list (optional)
        list of features

    Returns
    -------
    dict
        column name : inferred type (numeric, datetime or object)
    """

    def test_type(ser, _type):
        try:
            _ = ser.astype(_type)
            return True
        except (ValueError, TypeError):
            return False

    if not feats:
        feats = df.columns.tolist()

    _types = {}

    for col in feats:
        for k, _type in TYPE_MAPPING:
            if test_type(df[col], _type):
                _types[col] = k
                break
    return _types


def classify_value_counts(df, col, unique_thresh=0.05, type_dct=None):
    """
    Infer whether a feature is continuous, categorical, uninformative or binary.

    Parameters
    ----------
    df : dataframe
    col : str
    unique_thresh : int or float
        threshold of unique values to determine whether a numeric column
        is categorical or continuous.
        If ``unique_thresh > 1``, then this is assumed to be a raw number of
         unique value counts:
            - if length of ``df[col].value_counts() > unique_thresh``, then
            ``col`` is inferred to be continuous, otherwise categorical.
        if ``unique_thresh < 1``, this is assumed to be a percentage, i.e.
            - if ``df[col].value_counts() > unique_thresh * len(df[col])`` is
            inferred to be continuous, else categorical
    type_dct : dict
        this is a mapping of columns to allowed types.

    Returns
    -------
    str
        classification of the feature type.  This will be one of
        ``{'null', 'uninformative', 'binary', 'continuous', 'categorical'}``

    """
    val_counts = df[col].dropna().value_counts()
    if val_counts.empty:
        return "null"
    elif val_counts.size == 1:
        return "uninformative"
    elif val_counts.size == 2:
        return "binary"
    else:
        if not type_dct:
            type_dct = classify_feature_types(df[[col]])
        _validate_mappings(list(type_dct.values()))
        if type_dct[col] == "numeric":
            assert unique_thresh > 0
            if unique_thresh < 1.0:
                unique_thresh = int(unique_thresh * df.index.size)
            if len(val_counts) > unique_thresh:
                return "continuous"
    return "categorical"


def get_correlates(df, thresh=0.9, feats=None, **corr_kwargs):
    """
    Get correlate pairs with a correlation coeff greater that ``thresh``.

    Parameters
    ----------
    df : dataframe
    thresh : float (0 -> 1)
    feats : list
        list of column names

    Other Parameters
    ----------------
    See parameters for ``pd.DataFrame.corr``

    Returns
    -------
    pd.Series
    """
    if not feats:
        # remove object and datetime types (not comprehensive).
        feats = [f for f in df.columns.tolist() if df[f].dtype not in ["object", "<M8[ns]"]]

    corr_matrix = df[feats].corr(**corr_kwargs).abs()
    corr_pairs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        .stack()
        .sort_values(ascending=False)
    )
    return corr_pairs[corr_pairs > thresh]


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
    """
    Get particularly strong correlates.

    Parameters
    ----------
    df
    rho_thresh
    method

    Returns
    -------
    list
    """
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
    feat_type_dct = classify_feature_types(df, feats=feats)

    feat_class_dct = {
        k: classify_value_counts(df, k, unique_thresh=unique_thresh, type_dct=feat_type_dct)
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
