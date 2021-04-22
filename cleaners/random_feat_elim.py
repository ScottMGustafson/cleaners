import os

import yaml
from feets import random_feats

from cleaners.cleaner_base import CleanerBase
from cleaners.util import serializable_dict, validate_feats


def _dump_yaml(collection, dump_path):
    _pth = os.path.split(dump_path)[0]
    os.makedirs(_pth, exist_ok=True)
    with open(dump_path, "w") as f:
        yaml.dump(collection, f)


class RandomFeatureElimination(CleanerBase):
    def __init__(
        self,
        target_var,
        params,
        model_class,
        ix_vars=("date", "symbol"),
        mandatory=("target", "date", "symbol"),
        num_new_feats=10,
        feats_to_beat=9,
        min_num_folds=4,
        importance_type="total_gain",
        feature_dump_path=None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        target_var : str
        params : dict
        model_class : sklearn-XGBoost object
        feature_dump_path : str
        ix_vars : list
            list of mandatory vars useful for indexing
        mandatory : list
            list of features that cannot be dropped, but will be included in the fit
        num_new_feats : int
            number of new random features to generate
        feats_to_beat : int
            number of random feature which a feature under test must beat in terms of feature importance
        min_num_folds : int
            min number of folds in which a feature is more important than ``feats_to_beat`` random columns
        importance_type : str
            xgb-recognized importance type

        Other Parameters
        ----------------
        ignore : list
            features that cannot be included in the fit, but must not get dropped from the data
        initial_feats :
        """
        super().__init__(**kwargs)
        self.target_var = target_var
        self.params = params
        self.ix_vars = list(ix_vars)
        self.model_class = model_class
        # feats to be fit on that must not be dropped
        self.mandatory = list(mandatory)
        self.min_num_folds = min_num_folds
        self.num_new_feats = num_new_feats
        self.feats_to_beat = feats_to_beat
        self.importance_type = importance_type
        self.feature_dump_path = feature_dump_path
        self.remaining_feats = None
        self.sample_df = None
        self.feat_dct = None
        self.model_objects = None
        self.kfold_kwargs = kwargs.get("kfold_kwargs", dict(n_splits=5))
        self.ignore = list(set(self.ix_vars + [self.target_var] + kwargs.get("ignore", [])))
        self.initial_feats = kwargs.get("initial_feats", [])
        self.sample_rate = kwargs.get("sample_rate")
        self.drop = kwargs.get("drop", False)

    def _set_defaults(self, X):
        self.get_sample_df(X)
        self.ignore = list(set(self.ix_vars + [self.target_var] + self.ignore))
        if len(self.initial_feats) == 0:
            self.initial_feats = [x for x in X.columns if x not in self.ignore]
        else:
            self.initial_feats = sorted(
                {x for x in self.initial_feats + self.mandatory if x not in self.ignore}
            )
        if len(self.initial_feats) <= 1:
            raise ValueError("must have more than one feature to eliminate")
        if not self.num_new_feats:
            self.num_new_feats = len(self.initial_feats) // 3 + 1
        if not self.feats_to_beat:
            self.feats_to_beat = int(0.85 * self.num_new_feats)
        if self.feats_to_beat > self.num_new_feats:
            raise ValueError(
                "Number of feats to beat cannot be greater than number of random features"
            )
        if self.min_num_folds > self.kfold_kwargs.get("n_splits"):
            raise ValueError("Number of splits cannot be greater than number of folds")
        validate_feats(X, self.mandatory + self.initial_feats)

    def transform(self, X):  # noqa: D102
        self.log("xgb feat elim...")
        self._set_defaults(X)
        if self.sample_rate:
            df = self.sample_df
        else:
            df = X
        self.feat_dct, self.model_objects = random_feats.run_random_feats(
            df,
            features=self.initial_feats,
            target=self.target_var,
            model_class=self.model_class,
            min_num_folds=self.min_num_folds,
            num_new_feats=self.num_new_feats,
            num_random_cols_to_beat=self.feats_to_beat,
            model_kwargs=self.params,
            kfold_kwargs=self.kfold_kwargs,
            importance_type=self.importance_type,
        )
        self.remaining_feats = sorted(set(self.feat_dct.keys()))

        self.feat_dct = serializable_dict(self.feat_dct)
        if self.feature_dump_path:
            _dump_yaml(self.feat_dct, os.path.join(self.feature_dump_path, "feat_dict.yaml"))
            _dump_yaml(
                self.remaining_feats, os.path.join(self.feature_dump_path, "remaining_feats.yaml")
            )
        self.log("{} columns remain".format(len(self.remaining_feats)))
        if self.drop:
            drop_cols = [
                x for x in X.columns if x not in self.remaining_feats + self.mandatory + self.ignore
            ]
            return X.drop(columns=drop_cols)
        else:
            return X


class MultiTargetRandomFeatureElimination(CleanerBase):
    def __init__(
        self, target_var_lst, params_lst, model_class_lst, feature_dump_path, **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_var_lst = target_var_lst
        self.params_lst = params_lst
        self.model_classes = model_class_lst
        self.feature_dump_path = feature_dump_path
        self.feat_dct = dict()
        self.remaining_feats = []
        self.drop = bool(kwargs.get("drop", True))
        self.mandatory = None
        self.ignore = None
        self.eliminators = [
            RandomFeatureElimination(
                target_var=self.target_var_lst[i],
                params=self.params_lst[i],
                model_class=self.model_classes[i],
                drop=False,
                feature_dump_path=None,
                **kwargs,
            )
            for i in range(len(self.target_var_lst))
        ]

    def fit(self, X, y=None):
        for i, obj in enumerate(self.eliminators):
            self.eliminators[i] = obj.fit(X, y)
        return self

    def transform(self, X):
        for i in range(len(self.eliminators)):
            _ = self.eliminators[i].transform(X)
            self.feat_dct[i] = self.eliminators[i].feat_dct
            self.remaining_feats += self.eliminators[i].remaining_feats
        self.remaining_feats = sorted(set(self.remaining_feats))
        self.mandatory = self.eliminators[0].mandatory
        self.ignore = self.eliminators[0].ignore
        _dump_yaml(self.feat_dct, os.path.join(self.feature_dump_path, "multi_feat_dict.yaml"))
        _dump_yaml(
            self.remaining_feats,
            os.path.join(self.feature_dump_path, "multi_remaining_feats.yaml"),
        )
        if self.drop:
            drop_cols = [
                x for x in X.columns if x not in self.remaining_feats + self.mandatory + self.ignore
            ]
            return X.drop(columns=drop_cols)
        else:
            return X
