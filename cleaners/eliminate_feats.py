from cleaners import eda
from cleaners.cleaner_base import CleanerBase


class DropUninformative(CleanerBase):
    def __init__(self, unique_thresh=6, mandatory=("target", "date", "symbol"), **kwargs):
        super().__init__(**kwargs)
        self.unique_thresh = unique_thresh
        self.mandatory = mandatory
        self.feats = kwargs.get("feats")
        self.feat_type_dct = kwargs.get("feat_type_dct")
        self.feat_class_dct = kwargs.get("feat_class_dct")
        self.remaining_feats = None
        self.sample_rate = kwargs.get("sample_rate")
        self.sample_df = None

    def _set_defaults(self, X):
        self.get_sample_df(X)
        if not self.feats:
            self.feats = [x for x in X.columns if x not in self.mandatory]
        if not self.feat_class_dct:
            self.feat_type_dct, self.feat_class_dct = eda.process_feats(
                self.sample_df, unique_thresh=self.unique_thresh, feats=self.feats
            )

    def transform(self, X):
        self.log("dropping uninformative")
        self._set_defaults(X)
        drop_cols = eda.get_uninformative(self.feat_class_dct, self.mandatory)
        self.remaining_feats = list(
            set([x for x in X.columns if x not in drop_cols] + list(self.mandatory))
        )
        drop_cols = [x for x in X.columns if x not in self.remaining_feats]
        self.log("{} columns remain".format(len(self.remaining_feats)))
        return X.drop(columns=drop_cols)


class DropMostlyNaN(CleanerBase):
    def __init__(
        self,
        nan_frac_thresh=0.5,
        drop_cols=(),
        mandatory=("target", "date", "symbol"),
        skip_if_missing=True,
        apply_score_transform=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nan_frac_thresh = nan_frac_thresh
        self.mandatory = mandatory
        self.skip_if_missing = skip_if_missing
        self.drop_cols = list(drop_cols)
        self.apply_score_transform = apply_score_transform or len(drop_cols) > 0
        self.sample_rate = kwargs.get("sample_rate")
        self.sample_df = None

    @staticmethod
    def find_mostly_nan(X, nan_frac_thresh=0.5):
        sz = X.index.size
        cols = []
        for col in X.columns:
            nan_frac = X[col].isna().sum() / sz
            if nan_frac > nan_frac_thresh:
                cols.append(col)
        return cols

    def fit(self, X, y=None):
        return self

    def _validate_missing(self, X):
        assert not any(
            [x in self.mandatory for x in self.drop_cols]
        ), "drop_cols and mandatory overlap"
        missing_cols = [x for x in self.drop_cols if x not in X.columns]
        if not self.skip_if_missing:
            assert len(missing_cols) == 0, "missing one or more columns from dataframe: {}".format(
                missing_cols
            )
        else:
            self.drop_cols = missing_cols

    def build_transform(self, X):
        sz = X.index.size
        for col in X.columns:
            nan_frac = self.sample_df[col].isna().sum() / sz
            if nan_frac > self.nan_frac_thresh:
                self.drop_cols.append(col)
        self.log("dropping {} columns".format(len(self.drop_cols)))
        return X.drop(columns=self.drop_cols)

    def score_transform(self, X):
        self._validate_missing(X)
        return X.drop(columns=self.drop_cols)

    def transform(self, X):
        self.log("dropping mostly NaN cols")
        self.get_sample_df(X)
        if self.apply_score_transform or self.drop_cols:
            return self.score_transform(X)
        return self.build_transform(X)


class HighCorrelationElim(CleanerBase):
    def __init__(self, unique_thresh=6, mandatory=("target", "date", "symbol"), **kwargs):
        super().__init__(**kwargs)
        self.unique_thresh = unique_thresh
        self.mandatory = mandatory
        self.feats = kwargs.get("feats")
        self.feat_type_dct = kwargs.get("feat_type_dct")
        self.feat_class_dct = kwargs.get("feat_class_dct")
        self.rho_thresh = kwargs.get("rho_thresh", 0.98)
        self.method = kwargs.get("method", "spearman")
        self.remaining_feats = None
        self.num_cols = kwargs.get("num_cols")
        self.drop = kwargs.get("drop", False)
        self.sample_rate = kwargs.get("sample_rate")
        self.sample_df = None

    def fit(self, X, y=None):
        return self

    def _set_defaults(self, X):
        self.get_sample_df(X)
        if not self.feats:
            self.feats = [x for x in X.columns if x not in self.mandatory]
        if not self.feat_class_dct:
            self.feat_type_dct, self.feat_class_dct = eda.process_feats(
                self.sample_df, unique_thresh=self.unique_thresh, feats=self.feats
            )
        if not self.num_cols:
            self.num_cols = eda.get_type_lst(self.feat_type_dct, "numeric", self.mandatory)
        assert all(
            [col in X.columns for col in self.num_cols]
        ), "num_cols not all in data: {}".format(self.num_cols)

    def transform(self, X):
        self.log("dropping high correlation cols")
        self._set_defaults(X)
        drop_cols = eda.get_high_corr_cols(
            self.sample_df[self.num_cols], rho_thresh=self.rho_thresh, method=self.method
        )
        self.remaining_feats = list(
            set([x for x in X.columns if x not in drop_cols] + list(self.mandatory))
        )
        self.log("{} columns remain".format(len(self.remaining_feats)))
        if self.drop:
            drop_cols = [x for x in X.columns if x not in self.remaining_feats]
            return X.drop(columns=drop_cols)
        else:
            return X
