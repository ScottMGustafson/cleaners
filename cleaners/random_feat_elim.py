import logging

from feets import random_feats

from cleaners.cleaner_base import CleanerBase

logger = logging.getLogger("stocky_p")


class RandomFeatureElimination(CleanerBase):
    def __init__(
        self,
        target_var,
        params,
        model_class,
        ix_vars=("date", "symbol"),
        mandatory=("target", "date", "symbol"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_var = target_var
        self.params = params
        self.ix_vars = list(ix_vars)
        self.model_class = model_class
        self.mandatory = list(mandatory)
        self.ignore = kwargs.get("ignore", [])
        self.kfold_kwargs = kwargs.get("kfold_kwargs", dict(n_splits=5))
        self.initial_feats = kwargs.get("initial_feats", [])
        self.min_num_folds = kwargs.get("min_num_folds", 1)
        self.num_new_feats = kwargs.get("num_new_feats")
        self.feats_to_beat = kwargs.get("feats_to_beat")
        self.sample_rate = kwargs.get("sample_rate")
        self.drop = kwargs.get("drop", False)
        self.remaining_feats = None
        self.sample_rate = kwargs.get("sample_rate")
        self.sample_df = None
        self.feat_dct = None

    def fit(self, X, y=None):
        return self

    def _set_defaults(self, X):
        self.get_sample_df(X)
        if not self.initial_feats:
            self.initial_feats = [x for x in X.columns if x not in self.ix_vars + self.ignore]
            assert len(self.initial_feats) > 1, "must have more than one feature to eliminate"
        if not self.num_new_feats:
            self.num_new_feats = len(self.initial_feats) // 3 + 1
        if not self.feats_to_beat:
            self.feats_to_beat = int(0.85 * self.num_new_feats)
        assert self.feats_to_beat <= self.num_new_feats
        assert self.min_num_folds <= self.kfold_kwargs.get("n_splits")

    def transform(self, X):
        logger.info("xgb feat elim...")
        self._set_defaults(X)
        self.feat_dct = random_feats.run_random_feats(
            self.sample_df.reset_index(drop=True),  # .set_index(self.ix_vars),
            features=self.initial_feats,
            target=self.target_var,
            model_class=self.model_class,
            min_num_folds=self.min_num_folds,
            num_new_feats=self.num_new_feats,
            num_random_cols_to_beat=self.feats_to_beat,
            model_kwargs=self.params,
            kfold_kwargs=self.kfold_kwargs,
        )
        feats = sorted(set([k for k in self.feat_dct.keys()] + self.mandatory))
        _ignore = self.ix_vars + [self.target_var] + self.ignore
        _keep = list(set(self.mandatory + _ignore))
        self.remaining_feats = sorted(set([x for x in feats if x not in _ignore]))
        logger.info("{} columns remain".format(len(self.remaining_feats)))
        if self.drop:
            drop_cols = [x for x in X.columns if x not in list(set(feats + _keep))]
            return X.drop(columns=drop_cols)
        else:
            return X
