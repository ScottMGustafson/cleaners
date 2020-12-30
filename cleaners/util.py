"""Module for various things."""


def assert_no_duplicate_indices(df):
    """Assert a df has no duped indexes."""
    assert not all(df.set_index(["symbol", "date"]).index.duplicated()), "duplicate indices present"


def assert_no_duplicate_columns(df):
    """Assert a df has no duped columns."""
    assert not any(df.columns.duplicated())
