import logging

import numpy as np

logger = logging.getLogger("stocky_p")


class DropNamedCol:
    def __init__(self, drop_cols, mandatory=("target", "date", "symbol"), skip_on_fail=True):
        self.drop_cols = drop_cols
        self.mandatory = mandatory
        self.skip_on_fail = skip_on_fail

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        assert not any(
            [x in self.mandatory for x in self.drop_cols]
        ), "cannot drop mandatory_feats columns: {}".format(self.mandatory)
        logger.info("Dropping {}".format(self.drop_cols))
        for col in self.drop_cols:
            try:
                X = X.drop(columns=[col])
            except KeyError:
                if not self.skip_on_fail:
                    raise
        return X


class ReplaceBadColnameChars:
    def __init__(self, bad_chars="[, ]<>", repl_dct=None):
        self.bad_chars = bad_chars
        self.repl_dct = repl_dct

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("ReplaceBadColnameChars...")
        if not self.repl_dct:
            self.repl_dct = {}
        for col in X.columns:
            _col = col
            for sym in self.bad_chars:
                _col = _col.replace(sym, "")
            self.repl_dct[col] = _col
        return X.rename(columns=self.repl_dct)


class DropNa:
    def __init__(self, subset, replace_infinities=True):
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
        logger.info("DropNa...")
        assert all([x in X.columns for x in self.subset]), "columns not in data: {}".format(
            self.subset
        )
        if self.replace_inf:
            X = self._repl_inf(X)
        return X.dropna(subset=self.subset)
