from cleaners import eda
from cleaners.cleaner_base import CleanerBase


class BaseDropColsMixin:
    def __init__(self, *args, **kwargs):
        super(BaseDropColsMixin, self).__init__(*args, **kwargs)
        self.feature_names_in_ = None
        self.drop_cols_=None
    def transform(self, X):  # noqa: D102
        return X.drop(columns=self.drop_cols_)

    def get_feature_names_out(self, input_features=None):  # noqa: D102
        input_features = input_features or self.feature_names_in_
        if not all(x in input_features for x in self.drop_cols_):
            raise IndexError(
                f"Trying to drop some columns not present in data: from {input_features}, trying to drop {self.drop_cols_}")
        return [x for x in input_features if x not in self.drop_cols_]


class DropUninformative(BaseDropColsMixin, CleanerBase):
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
        self.drop_cols_ = None

    def _set_defaults(self, X):
        self.get_sample_df(X)
        if not self.feats:
            self.feats = [x for x in X.columns if x not in self.mandatory]
        if not self.feat_class_dct:
            self.feat_type_dct, self.feat_class_dct = eda.process_feats(
                self.sample_df, unique_thresh=self.unique_thresh, feats=self.feats
            )

    def fit(self, X, y=None, **kwargs):
        self.log("dropping uninformative")
        self._set_defaults(X)
        self.drop_cols_ = eda.get_uninformative(self.feat_class_dct, self.mandatory)
        self.remaining_feats = list(
            set([x for x in X.columns if x not in self.drop_cols_] + list(self.mandatory))
        )
        self.drop_cols_ = [x for x in X.columns if x not in self.remaining_feats]
        self.log("{} columns will remain after dropping".format(len(self.remaining_feats)))
        self.feature_names_in_ = X.columns.tolist()
        return self


class DropMostlyNaN(BaseDropColsMixin, CleanerBase):
    def __init__(
        self,
        nan_frac_thresh=0.5,
        drop_cols=(),
        mandatory=("target", "date", "symbol"),
        skip_if_missing=True,
        apply_score_transform=False,
        **kwargs,
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
        """Find columns where mostly nan."""
        sz = X.index.size
        cols = []
        for col in X.columns:
            nan_frac = X[col].isna().sum() / sz
            if nan_frac > nan_frac_thresh:
                cols.append(col)
        return cols

    def _validate_missing(self, X):
        assert not any(
            [x in self.mandatory for x in self.drop_cols]
        ), "drop_cols and mandatory_feats overlap"
        missing_cols = [x for x in self.drop_cols if x not in X.columns]
        if not self.skip_if_missing:
            assert len(missing_cols) == 0, "missing one or more columns from dataframe: {}".format(
                missing_cols
            )
        else:
            self.drop_cols = missing_cols

    def fit(self, X, y=None, **kwargs):
        self.log("dropping mostly NaN cols")
        self.get_sample_df(X)
        sz = self.sample_df.index.size
        cols = [x for x in self.sample_df.columns if x not in self.mandatory]
        nan_frac = self.sample_df[cols].isna().sum() / sz
        self.drop_cols += nan_frac[nan_frac > self.nan_frac_thresh].index.tolist()
        self.drop_cols = sorted(list(set(self.drop_cols)))
        self.log("dropping {} columns".format(len(self.drop_cols)))
        self.feature_names_in_ = cols


class HighCorrelationElim(BaseDropColsMixin, CleanerBase):
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

    def fit(self, X, y=None, **kwargs):
        self.log("Checking for high correlation cols")
        self._set_defaults(X)
        self.drop_cols_ = eda.get_high_corr_cols(
            self.sample_df[self.num_cols], rho_thresh=self.rho_thresh, method=self.method
        )
        self.remaining_feats = list(
            set([x for x in X.columns if x not in self.drop_cols_] + list(self.mandatory))
        )
        self.log(f"{len(self.remaining_feats)} columns will remain after dropping {len(self.drop_cols_)} columns")

