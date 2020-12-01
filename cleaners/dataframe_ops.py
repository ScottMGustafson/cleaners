from cleaners.cleaner_base import CleanerBase


class IndexForwardFillna(CleanerBase):
    def __init__(self, date_col="date", method="ffill", **kwargs):
        super().__init__(**kwargs)
        self.date_col = date_col
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.log("fillna...")
        if X.index.name != self.date_col:
            X = X.set_index(self.date_col)
        X = X.sort_index().fillna(method=self.method)
        return X


class JoinDFs(CleanerBase):
    def __init__(self, right_df, how="left", join=True, ix_col=None, **kwargs):
        """

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

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.log("joining...")
        if self.join:
            X = X.join(self.right_df, how=self.how)
        else:
            X = X.merge(self.right_df, on=self.ix_col, how=self.how)
        return X.loc[:, ~X.columns.duplicated()]


class ResetIndex(CleanerBase):
    def __init__(self, **kwargs):
        super(ResetIndex, self).__init__(**kwargs)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reset_index()