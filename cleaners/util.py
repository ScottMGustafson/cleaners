"""Module for various things."""
import numpy as np


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
    return df


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


def _to_float_or_str(x):
    """Try to convert and object to a python native numeric or string type."""
    for type in [int, float, str]:
        if isinstance(x, type):
            return x
    # otherwise, if it can be converted to a number, do it, otherwise raise.
    try:
        return float(x)
    except (ValueError, TypeError) as _:  # noqa: F841
        raise ValueError(f"Unrecognized type: ``{type(x)}``")


def to_json_serializable(obj):
    """Recursively convert an object to json-serializable types."""
    if isinstance(obj, dict):
        for k in obj.keys():
            obj[k] = to_json_serializable(obj[k])
        return obj
    if any(isinstance(obj, x) for x in [list, tuple, np.ndarray]):
        return list(map(to_json_serializable, obj))
    else:
        return _to_float_or_str(obj)
