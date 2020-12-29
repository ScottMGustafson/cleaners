import numpy as np

from cleaners.cleaner_base import CleanerBase


class DropNamedCol(CleanerBase):
    def __init__(
        self, drop_cols, mandatory=("target", "date", "symbol"), skip_on_fail=True, **kwargs
    ):
        super().__init__(**kwargs)
        self.drop_cols = drop_cols
        self.mandatory = mandatory
        self.skip_on_fail = skip_on_fail

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        assert not any(
            [x in self.mandatory for x in self.drop_cols]
        ), "cannot drop mandatory_feats columns: {}".format(self.mandatory)
        self.log("Dropping {}".format(self.drop_cols))
        for col in self.drop_cols:
            try:
                X = X.drop(columns=[col])
            except KeyError:
                if not self.skip_on_fail:
                    raise
        return X


class ReplaceBadColnameChars(CleanerBase):
    def __init__(self, bad_chars="[, ]<>", repl_dct=None, **kwargs):
        super().__init__(**kwargs)
        self.bad_chars = bad_chars
        self.repl_dct = repl_dct

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.log("ReplaceBadColnameChars...")
        if not self.repl_dct:
            self.repl_dct = {}
        for col in X.columns:
            _col = col
            for sym in self.bad_chars:
                _col = _col.replace(sym, "")
            self.repl_dct[col] = _col
        return X.rename(columns=self.repl_dct)


class DropNa(CleanerBase):
    def __init__(self, subset, replace_infinities=True, **kwargs):
        super().__init__(**kwargs)
        self.subset = subset
        self.replace_inf = replace_infinities

    def fit(self, X, y=None):
        return self

    def _repl_inf(self, X):
        if hasattr(X, "compute"):
            X[self.subset] = X[self.subset].map_partitions(
                lambda x: x.replace({np.inf: np.nan, -np.inf: np.nan})
            )
        else:
            X[self.subset] = X[self.subset].replace({np.inf: np.nan, -np.inf: np.nan})
        return X

    def transform(self, X):
        self.log("DropNa...")
        assert all([x in X.columns for x in self.subset]), "columns not in data: {}".format(
            self.subset
        )
        if self.replace_inf:
            X = self._repl_inf(X)
        return X.dropna(subset=self.subset)


class DropDuplicates(CleanerBase):
    def __init__(self, silently_fix=False, df_identifier="", **kwargs):
        super().__init__(**kwargs)
        self.silently_fix = silently_fix
        self.df_identifier = df_identifier

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.log("checking dupes")
        if not self.silently_fix:
            assert not any(X.columns.duplicated()), (
                f"{self.df_identifier} data has duplicates:"
                + f"{X.columns[X.columns.duplicated()].tolist()}"
            )
            return X
        else:
            return X.loc[:, ~X.columns.duplicated()]
