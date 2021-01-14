"""Module for various things."""


def assert_no_duplicate_indices(df):
    """Assert a df has no duped indexes."""
    assert not all(df.set_index(["symbol", "date"]).index.duplicated()), "duplicate indices present"


def assert_no_duplicate_columns(df):
    """Assert a df has no duped columns."""
    assert not any(df.columns.duplicated())


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
        assert X.known_divisions
        ix_name = X.index.name
        X = X.reset_index().set_index(ix_name, sorted=True)
    else:
        assert X.index.name, "Index not set"
        X = X.sort_index()
    return X


def set_index(X, ix_name, sort=True):
    """Set an index for either dask or pandas dataframes."""
    if hasattr(X, "compute"):
        if not X.known_divisions:
            X = X.set_index(ix_name, sorted=sort)
        else:
            X = X.reset_index().set_index(ix_name, sorted=sort)
    else:
        if not X.index.name:
            X = X.set_index(ix_name)
        else:
            X = X.reset_index().set_index(ix_name)
        if sort:
            X = X.sort_index()
    return X
