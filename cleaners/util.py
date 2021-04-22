"""Module for various things."""
import numpy as np


def assert_no_duplicate_indices(df):
    """Assert a df has no duped indexes."""
    if any(df.set_index(["symbol", "date"]).index.duplicated()):
        raise IndexError("duplicate indices present")


def assert_no_duplicate_columns(df):
    """Assert a df has no duped columns."""
    if any(df.columns.duplicated()):
        raise IndexError("duplicate columns present")


def cum_sum_index(df):
    """Use cumulative sum of ones to create simple int index."""
    df = df.reset_index(drop=True)
    df["temp_ix"] = 1
    df["cum_sum"] = df["temp_ix"].cumsum()
    df = df.set_index("cum_sum", drop=True)
    df = df.drop(columns=["temp_ix"])
    return df.persist()


def sort_index(X):
    """Sort an index for either dask or pandas dataframes."""
    if hasattr(X, "compute"):
        if not X.known_divisions:
            raise IndexError("known_divisions must be true here.")
        ix_name = X.index.name
        X = X.reset_index().set_index(ix_name, sorted=True)
    else:
        if not X.index.name:
            raise IndexError("Index not set")
        X = X.sort_index()
    return X  # noqa: R504


def set_index(X, ix_name, sort=True):
    """Set an index for either dask or pandas dataframes."""
    if hasattr(X, "compute"):
        if not X.known_divisions:
            return X.set_index(ix_name, sorted=sort)
        else:
            return X.reset_index().set_index(ix_name, sorted=sort)
    else:
        if not X.index.name:
            _X = X.set_index(ix_name)
        else:
            _X = X.reset_index().set_index(ix_name)
        if sort:
            _X = X.sort_index()
        return _X  # noqa: R504


def validate_feats(X, feats):
    """
    Validate that columns are in dataframe.

    Parameters
    ----------
    df : DataFrame
    feats : list

    Raises
    ------
    KeyError

    """
    missing = [x for x in feats if x not in X.columns]
    if len(missing) > 0:
        raise KeyError("Data is missing columns: {}".format(missing))


def serializable_dict(dct, ignore=None):
    """Replace numpy-specific types to types that can be properly serialized in yaml."""
    type_map = {np.str_: str, np.float64: float, np.int64: int, np.bool_: bool}
    if not ignore:
        ignore = []
    new_dct = {}
    for k, v in dct.items():
        if k in ignore:
            new_dct[k] = v
            continue
        if isinstance(v, dict):
            new_dct[k] = serializable_dict(v)
        else:
            try:
                new_dct[k] = type_map[type(v)](v)
            except KeyError:  # silently pass if not one of these types
                new_dct[k] = v
    return new_dct
