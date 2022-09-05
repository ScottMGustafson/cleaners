"""Dataframe operations utilities."""

import dask.dataframe as dd

from cleaners.cleaner_base import CleanerBase
from cleaners.util import sort_index


class IndexForwardFillna(CleanerBase):
    """Fill missing data."""

    def __init__(self, ix_col="date", method="ffill", is_sorted=True, **kwargs):
        """
        Init method.

        Parameters
        ----------
        ix_col : str, default=``date``
            column to use to determine fill ordering, typically a datetime
        method : str, default=``ffill``
            fill method.
        is_sorted : bool, default=True
            if not sorted, sort.
        """
        super().__init__(**kwargs)
        self.ix_col = ix_col
        self.method = method
        self.is_sorted = is_sorted

    def transform(self, X):  # noqa: D102
        self.log("fillna...")
        if X.index.name != self.ix_col:
            if isinstance(X, dd.DataFrame):
                X = X.set_index(self.ix_col, sorted=True)
            else:
                X = X.reset_index().set_index(self.ix_col).sort_index()
        if not self.is_sorted:
            X = sort_index(X)
        X = X.fillna(method=self.method)
        return X


class JoinDFs(CleanerBase):
    """Join two dataframes, compatible with either dask or pandas."""

    def __init__(self, right_df, how="left", join=True, ix_col=None, **kwargs):
        """
        Init method.

        Parameters
        ----------
        right_df
        how
        join : bool
            if true, use join on index, otherwise, merge on ix_col
        ix_col
        """
        super().__init__(**kwargs)
        self.right_df = right_df
        self.join = join
        if not self.join:
            assert ix_col
        self.ix_col = ix_col
        self.how = how

    def fit(self, X, y=None, **kwargs):
        self.feature_names_in_ = X.columns.tolist()
        self.check_intersect(X)
        return self

    def check_intersect(self, X):
        _intersect = set(self.right_df.columns).intersection(X.columns)
        if not self.join:
            assert not _intersect, "join: L and R dataframes have overlapping columns: {}".format(
                _intersect
            )
        else:
            assert len(_intersect) == 1
            assert list(_intersect)[0] == self.ix_col

    def transform(self, X):  # noqa: D102
        self._check_input_features(X)
        self.log("joining...")
        self.check_intersect(X)
        if self.join:
            return X.join(self.right_df, how=self.how)
        else:
            return X.merge(self.right_df, on=self.ix_col, how=self.how)

    def get_feature_names_out(self, input_features=None):
        """
        Get feature names.

        Parameters
        ----------
        input_features : list

        Returns
        -------
        list
        """
        input_features = input_features or self.feature_names_in_
        out = input_features + self.right_df.columns.tolist()
        if not self.join:
            del out[out.index(self.ix_col)]
        return out


class ResetIndex(CleanerBase):
    """Reset dataframe index compatible with sklearn pipelines."""

    def __init__(self, **kwargs):  # noqa: D107
        super(ResetIndex, self).__init__(**kwargs)

    def transform(self, X):  # noqa: D102

        return X.reset_index()


class CompositeIndex(CleanerBase):
    """
    Dask does not support multi-indexing so create a composite index from list of index columns.

    Notes
    -----
     - order in ``ix_list`` matters: most important first.
     - if order matters, then elements should be correctly sortable as a string,
     e.g.: MM/DD/YYYY formats for dates won't work, use YYYY-MM-DD instead.
     e.g.: numbers may not sort appropriately without first zero-padding them.
    """

    def __init__(self, ix_list, join_char="-", new_ix_name="index", drop=False, *args, **kwargs):
        """
        Init method.

        Parameters
        ----------
        ix_list : list
        join_char : str, default=``-``
        new_ix_name : str, default=``index``
        drop : boolean, default=False
        """
        super(CompositeIndex, self).__init__(*args, **kwargs)
        self.ix_list = ix_list
        self.join_char = join_char
        self.drop = drop
        self.new_ix_name = new_ix_name
        assert len(ix_list) > 1, "ix_list is size {}: must be > 1".format(len(ix_list))

    def fit(self, X, y=None, **kwargs):
        self.feature_names_in_ = list(X.columns)
        assert all(
            x in self.feature_names_in_ for x in self.ix_list
        ), f"{self.ix_list} not present in data."

    def transform(self, X):  # noqa: D102
        self._check_input_features(X)
        X = X.reset_index()
        X[self.new_ix_name] = X[self.ix_list[0]].astype(str)
        for col in self.ix_list[1:]:
            X[self.new_ix_name] += self.join_char + X[col].astype(str)
        if self.drop:
            X = X.drop(self.ix_list, axis=1)
        self.old_ix_name_ = X.index.name
        X = X.set_index(self.new_ix_name, drop=self.drop)
        return X

    def get_feature_names_out(self, input_features=None):
        input_features = input_features or self.feature_names_in_
