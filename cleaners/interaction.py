"""Cleaner for 2-way interactions."""

import random

from cleaners.cleaner_base import CleanerBase


class TwoWayInteractions(CleanerBase):
    """Cleaner for two-way interctions."""

    def __init__(self, max_new_feats=300, **kwargs):
        """
        Generates interaction features up to ``max_new_feats``.

        Parameters
        ----------
        max_new_feats : int, default=300
            Max number of new feats to generate to prevent the memory from blowing up.
            for example, if you have 30 features, then 30 * 30 == 900  2-way interaction features.
            This cleaner will randomly generate 300 out of that 900. Therefore
            the suggested usage is to just use this interaction among
            the top N most important feats unless you got tons of
            memory to spare.

        Other Parameters
        ----------------
        seed : int, default=0
        exclude : list
            feats to exclude
        subset: list
            feats used to generate interactions.
        feat_pairs : list
            existing feature pairs to use
        add_new_feats : bool
            if true, adds new features from feature pairs.  This would be
            false if you are scoring new data on an existing model. if you
            are building a new model, this should be true.
        """
        super(TwoWayInteractions, self).__init__(**kwargs)
        self.max_new_feats = max_new_feats
        self.seed = kwargs.get("seed", 0)
        self.exclude = kwargs.get("exclude", [])
        self.subset = kwargs.get("subset", [])
        self.feat_pairs = kwargs.get("feat_pairs", [])
        self.add_new_feats = kwargs.get("add_new_feats", True)

    def _set_defaults(self, X):
        if not self.subset:
            self.subset = [x for x in X.columns if x not in self.exclude]
        else:
            for col in self.subset:
                assert (
                    col in X.columns and col not in self.exclude
                ), "TwoWayInteraction::{} should not be in subset".format(col)

    def _sample_list(self, feat_pairs):
        random.seed(self.seed)
        random.shuffle(feat_pairs)
        if len(feat_pairs) <= self.max_new_feats:
            return feat_pairs
        else:
            return feat_pairs[: self.max_new_feats]

    def _get_feat_pairs(self, X):
        feat_pairs = []
        for i, x in enumerate(self.subset):
            for j, y in enumerate(self.subset):
                if j <= i:
                    continue
                feat_pairs.append((x, y))

        if len(feat_pairs) > self.max_new_feats:
            feat_pairs = self._sample_list(feat_pairs)
        return [x for x in feat_pairs if x not in self.feat_pairs]

    def transform(self, X):  # noqa: D102
        self._set_defaults(X)
        if self.add_new_feats:
            self.feat_pairs += self._get_feat_pairs(X)
        self.log("Applying {} two-way feature interactions".format(len(self.feat_pairs)))
        for x, y in self.feat_pairs:
            try:
                X[f"{x}_X_{y}"] = X[x] * X[y]
            except TypeError:
                msg = f"{x}, {X[x].dtype} cannot multiply with {y}, {X[y].dtype}"
                raise TypeError(msg)
        return X


def get_expected_pairs(feat_pairs, feats):
    """
    Get expected feature pairs .

    Parameters
    ----------
    feat_pairs : list
        list of feature tuples
    feats : list
        features to consider

    Returns
    -------
    list[tuple]
        list of filtered feature pairs
    """
    expected_pairs = []
    expected_two_ways_columns = []
    for x, y in feat_pairs:
        col = "{}_X_{}".format(x, y)
        if col in feats:
            expected_pairs.append([x, y])
            expected_two_ways_columns.append(col)
    return expected_pairs
