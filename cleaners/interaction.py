import logging
import random

logger = logging.getLogger("stocky_p")


class TwoWayInteractions:
    def __init__(self, max_new_feats=300, **kwargs):
        self.max_new_feats = max_new_feats
        self.seed = kwargs.get("seed", 0)
        self.exclude = kwargs.get("exclude", [])
        self.subset = kwargs.get("subset", [])
        self.feat_pairs = kwargs.get("feat_pairs", [])
        self.add_new_feats = kwargs.get("add_new_feats", True)

    def fit(self, X, y=None):
        return self

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

    def transform(self, X):
        self._set_defaults(X)
        if self.add_new_feats:
            self.feat_pairs += self._get_feat_pairs(X)
        logger.info("Applying {} two-way feature interactions".format(len(self.feat_pairs)))
        for x, y in self.feat_pairs:
            try:
                X[f"{x}_X_{y}"] = X[x] * X[y]
            except TypeError:
                msg = f"{x}, {X[x].dtype} cannot multiply with {y}, {X[y].dtype}"
                raise TypeError(msg)
        return X


def get_expected_pairs(feat_pairs, feats):
    """

    Parameters
    ----------
    feat_pairs : list
        list of feature tuples
    feats : list

    Returns
    -------
    list
        list of filtered feature tuples
    """
    expected_pairs = []
    expected_two_ways_columns = []
    for x, y in feat_pairs:
        col = "{}_X_{}".format(x, y)
        if col in feats:
            expected_pairs.append([x, y])
            expected_two_ways_columns.append(col)
    return expected_pairs
