"""Module for various things."""


def assert_no_duplicate_indices(df):
    """Assert a df has no duped indexes."""
    assert not all(df.set_index(["symbol", "date"]).index.duplicated()), "duplicate indices present"


def assert_no_duplicate_columns(df):
    """Assert a df has no duped columns."""
    assert not any(df.columns.duplicated())


def sort_index(X):
    if hasattr(X, "compute"):
        assert X.known_divisions
        ix_name = X.index.name
        X = X.reset_index().set_index(ix_name, sorted=True)
    else:
        assert X.index.name, "Index not set"
        X = X.sort_index()
    return X


def set_index(X, ix_name, sort=True):
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
