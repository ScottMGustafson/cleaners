import numpy as np
import pandas as pd
from feets import explore


def get_describe_df(df, unique_thresh=6, percentiles=None):
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


def get_high_corr_cols(df, rho_thresh, method="spearman"):
    corr_matrix = df.corr(method=method).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    return [column for column in upper.columns if any(upper[column] > rho_thresh)]


def get_unary_columns(df):
    return [x for x in df.columns if len(df[x].unique()) == 1]


def get_mostly_nan(df, thresh=0.05):
    sz = df.index.size
    return [x for x in df.columns if df[x].dropna().size / sz < thresh]


def get_binary(df):
    return [x for x in df.columns if len(df[x].dropna().value_counts()) == 2]


def get_mostly_value(df, value, thresh=0.05):
    sz = df.index.size
    return [x for x in df.columns if df[df[x] == value][x].size / sz > thresh]


def get_categorical(df, thresh=20):
    return [x for x in df.columns if df[x].value_counts().size <= thresh]


def get_type_lst(feat_type_dct, feat_type, exclude_lst):
    return [k for k, v in feat_type_dct.items() if v == feat_type and k not in exclude_lst]


def find_relevant_columns(lst, model_feats, mandatory):
    """return items where item is in model_feats or mandatory_feats cols"""
    return [x for x in lst if any([x in y for y in model_feats + mandatory])]


def process_feats(df, unique_thresh=0.01, feats=None):
    if any(df.columns.duplicated()):
        lst = df.loc[:, df.columns.duplicated()].columns.tolist()
        raise Exception(f"The following columns are duplicated: {sorted(list(set(lst)))}")

    if not feats:
        feats = df.columns.tolist()
    feat_type_dct = explore.classify_feature_types(df, feats=feats)

    feat_class_dct = {
        k: explore.classify_value_counts(df, k, unique_thresh=unique_thresh, type_dct=feat_type_dct)
        for k in feats
    }
    return feat_type_dct, feat_class_dct


def get_uninformative(feat_class_dct, mandatory=()):
    return [
        x
        for x, v in feat_class_dct.items()
        if v in ["null", "uninformative"] and v not in mandatory
    ]
