def assert_no_duplicate_indices(df):
    assert not all(df.set_index(["symbol", "date"]).index.duplicated()), "duplicate indices present"


def assert_no_duplicate_columns(df):
    assert not any(df.columns.duplicated())