"""Cleaners for dropping, replacing and renaming columns."""

import numpy as np

from cleaners.cleaner_base import CleanerBase


class DropNamedCol(CleanerBase):
    """
    Drop columns by Name.

    Parameters
    ----------
    drop_cols : list
    mandatory : list
    skip_on_fail : bool
    """

    def __init__(
        self, drop_cols, mandatory=("target", "date", "symbol"), skip_on_fail=True, **kwargs
    ):
        super().__init__(**kwargs)
        self.drop_cols = drop_cols
        self.mandatory = mandatory
        self.skip_on_fail = skip_on_fail

    def transform(self, X):  # noqa: D102
        assert not any(
            [x in self.mandatory for x in self.drop_cols]
        ), "cannot drop mandatory_feats columns: {}".format(self.mandatory)
        self.log("Dropping {}".format(self.drop_cols))
        for col in self.drop_cols:
            try:
                X = X.drop(columns=[col])
            except (KeyError, ValueError) as _:  # noqa: F841
                if not self.skip_on_fail:
                    raise
        return X

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
        return [x for x in input_features if x not in self.drop_cols]


class ReplaceBadColnameChars(CleanerBase):
    """Replace bad column names with characters not in `[, ]<>`."""

    def __init__(self, bad_chars="[, ]<>", repl_dct=None, **kwargs):
        super().__init__(**kwargs)
        self.bad_chars = bad_chars
        self.repl_dct = repl_dct

    def transform(self, X):  # noqa: D102
        self.log("ReplaceBadColnameChars...")
        if not self.repl_dct:
            self.repl_dct = {}
        for col in X.columns:
            _col = col
            for sym in self.bad_chars:
                _col = _col.replace(sym, "")
            self.repl_dct[col] = _col

        return X.rename(columns=self.repl_dct)

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
        _names_out = list(input_features)
        for i, name in enumerate(_names_out):
            try:
                _names_out[i] = self.repl_dct[name]
            except KeyError:
                pass
        return _names_out


class DropNa(CleanerBase):
    """
    Drop NaNs and infinities.

    Parameters
    ----------
    subset : list
    replace_infinities : bool
    """

    def __init__(self, subset, replace_infinities=True, **kwargs):
        super().__init__(**kwargs)
        self.subset = list(subset)
        self.replace_inf = replace_infinities

    def _repl_inf(self, X):
        repl_dct = {np.inf: np.nan, -np.inf: np.nan, "NaN": np.nan, "nan": np.nan, "NA": np.nan}
        if hasattr(X, "compute"):
            X[self.subset] = X[self.subset].map_partitions(lambda x: x.replace(repl_dct))
        else:
            X[self.subset] = X[self.subset].replace(repl_dct)
        return X

    def transform(self, X):  # noqa: D102
        self.log("DropNa...")
        assert all([x in X.columns for x in self.subset]), "columns not in data: {}".format(
            self.subset
        )
        if self.replace_inf:
            X = self._repl_inf(X)
        return X.dropna(subset=self.subset)


class DropDuplicates(CleanerBase):
    """
    Drop duplicate columns.

    Parameters
    ----------
    silently_fix : bool (default=False)
    df_identifier : str (default= `drop_dupes`)

    Attributes
    ----------
    dupe_cols_ : list
    feature_names_in_ : list
    """

    def __init__(self, silently_fix=False, df_identifier="drop_dupes", **kwargs):
        super().__init__(**kwargs)
        self.silently_fix = silently_fix
        self.df_identifier = df_identifier
        self.dupe_cols_ = None

    def fit(self, X, y=None):  # noqa: D102
        self.dupe_cols_ = X.columns.duplicated()
        if not self.dupe_cols_.any():
            self.dupe_cols_ = []
            self.feature_names_in_ = X.columns.tolist()
            return self

        self.feature_names_in_ = X.loc[:, ~self.dupe_cols_].columns.unique().tolist()
        self.log("checking dupes")
        if not self.silently_fix:
            raise IndexError(f"{self.df_identifier} data has duplicates:" + f"{self.dupe_cols_}")
        return self

    def transform(self, X):  # noqa: D102
        if self.dupe_cols_ is None:
            raise ValueError("DropDuplicates has not yet been fitted.")
        if len(self.dupe_cols_) > 0:
            return X.loc[:, ~self.dupe_cols_]
        return X

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
        return [x for x in input_features if x in self.feature_names_in_]
