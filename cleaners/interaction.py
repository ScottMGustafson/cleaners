"""Cleaner for 2-way interactions."""

import random

import yaml

from cleaners.cleaner_base import CleanerBase


def _get_top_feats(int_top_feats, feat_impt):
    sorted_imp_feats = sorted(
        [(k, v) for k, v in feat_impt.items()], key=lambda x: x[1], reverse=True
    )  # noqa: C407
    if int_top_feats < len(sorted_imp_feats):
        top_n_feats = [x for x, y in sorted_imp_feats[:int_top_feats]]
    else:
        top_n_feats = [x for x, y in sorted_imp_feats]
    return sorted(set(top_n_feats))


class TwoWayInteractions(CleanerBase):
    """
    Cleaner for two-way interctions generating interaction features up to ``max_interact_feats``.

    Parameters
    ----------
    interact_top_n : int, default=30
        top n feats to interact from feature importance dict
    max_interact_feats : int, default=300
        Max number of new feats to generate to prevent the memory from blowing up.
        for example, if you have 30 features, then 30 * 30 == 900  2-way interaction features.
        This cleaner will randomly generate 300 out of that 900. Therefore
        the suggested usage is to just use this interaction among
        the top N most important feats unless you got tons of
        memory to spare.
    seed : int, default=0
        random seed
    add_new_feats : bool
        if true, adds new features from feature pairs.  This would be
        false if you are scoring new data on an existing model. if you
        are building a new model, this should be true.
    interact_feature_path : str
        If supplied, this will take the place of any best feature inferring logic.

    Other Parameters
    ----------------
    exclude : list
        feats to exclude
    subset: list
        feats used to generate interactions.
    feat_pairs : list
        existing feature pairs to use
    """

    def __init__(
        self,
        max_interact_feats=300,
        interact_top_n=30,
        seed=0,
        add_new_feats=True,
        feature_dict_path=None,
        interact_feature_path=None,
        **kwargs,
    ):
        super(TwoWayInteractions, self).__init__(**kwargs)
        self.max_interact_feats = max_interact_feats
        self.seed = seed
        self.exclude = kwargs.get("exclude", [])
        self.subset = kwargs.get("subset", [])
        self.feat_pairs = kwargs.get("feat_pairs", [])
        self.add_new_feats = add_new_feats
        self.feature_dict_path = feature_dict_path
        self.interact_top_n = interact_top_n
        self.interact_feature_path = interact_feature_path

    def _set_defaults(self, X):
        if not self.subset:
            self.subset = [x for x in X.columns if x not in self.exclude]
        else:
            for col in self.subset:
                if col not in X.columns or col in self.exclude:
                    KeyError("TwoWayInteraction::{} should not be in subset".format(col))

    def _get_interact_subset(self):
        if self.interact_top_n and self.feature_dict_path:
            with open(self.feature_dict_path, "r") as f:
                feat_impt = yaml.full_load(f)
            self.subset += _get_top_feats(self.interact_top_n, feat_impt)
        elif self.interact_feature_path:
            with open(self.interact_feature_path, "r") as f:
                self.subset += list(yaml.full_load(f))
        else:
            raise Exception("Must provide interact feature path")
        self.subset = sorted(set(self.subset))

    def _sample_list(self, feat_pairs):
        random.seed(self.seed)
        random.shuffle(feat_pairs)
        if len(feat_pairs) <= self.max_interact_feats:
            return feat_pairs
        else:
            return feat_pairs[: self.max_interact_feats]

    def _get_feat_pairs(self, X):
        feat_pairs = []
        for i, x in enumerate(self.subset):
            for j, y in enumerate(self.subset):
                if j <= i:
                    continue
                if x == y:
                    raise ValueError(f"duplicate values found in subset: {x}")
                feat_pairs.append((x, y))

        if len(feat_pairs) > self.max_interact_feats:
            feat_pairs = self._sample_list(feat_pairs)

        feat_pairs = _concat_list_of_tuples(feat_pairs, self.feat_pairs, list(X.columns))
        return feat_pairs

    def _apply_interactions(self, X):
        for x, y in self.feat_pairs:
            try:
                X[f"{x}_X_{y}"] = X[x] * X[y]
            except TypeError:
                msg = f"{x}, {X[x].dtype} cannot multiply with {y}, {X[y].dtype}"
                raise TypeError(msg)
        return X

    def fit(self, X, y=None, **kwargs):  # noqa: D102
        self._set_defaults(X)
        if self.add_new_feats:
            self._get_interact_subset()
            self.subset = [x for x in self.subset if x in X.columns]
            self.feat_pairs = self._get_feat_pairs(X)
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):  # noqa: D102
        self.log("Applying {} two-way feature interactions".format(len(self.feat_pairs)))
        X = self._apply_interactions(X)
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
        return self.feature_names_in_ + get_expected_pairs(self.feat_pairs, self.feature_names_in_)


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


def _concat_list_of_tuples(lst1, lst2, allowed_list):
    feat_pairs = [
        tuple(sorted([x[0], x[1]]))
        for x in lst1 + lst2
        if x[0] in allowed_list and x[1] in allowed_list
    ]
    return list(map(list, sorted(set(feat_pairs))))
