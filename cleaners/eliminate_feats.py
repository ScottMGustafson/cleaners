"""Feature dropping utilities."""
from cleaners import eda
from cleaners.cleaner_base import CleanerBase


class BaseDropColsMixin:
    """
    Mixin Class to add transform and feature_names_out.

    Attributes
    ----------
    feature_names_in : list
    drop_cols_ : list
        list of columns to drop.
    """

    def __init__(self, *args, **kwargs):
        super(BaseDropColsMixin, self).__init__(*args, **kwargs)
        self.feature_names_in_ = None
        self.drop_cols_ = None

    def transform(self, X):  # noqa: D102
        if self.drop_cols_ is None:
            return X
        _to_drop = [x for x in self.drop_cols_ if x in X.columns]
        return X.drop(columns=_to_drop)

    def get_feature_names_out(self, input_features=None):  # noqa: D102
        input_features = input_features or self.feature_names_in_
        if not all(x in input_features for x in self.drop_cols_):
            raise IndexError(
                f"""
                Trying to drop some columns not present in data:
                from {input_features}, trying to drop {self.drop_cols_}
                """
            )
        return [x for x in input_features if x not in self.drop_cols_]


class DropUninformative(BaseDropColsMixin, CleanerBase):
    """
    Drop columns determined to be uninformative.

    Parameters
    ----------
    unique_thresh : int
    mandatory : list[str]
    feats : list[str]
    feat_type_dct : dict[str]
    feat_class_dct : dict[str]
    remaining_feats : list

    Attributes
    ----------
    feature_names_in : list
    drop_cols_ : list
        list of columns to drop.

    Other Parameters
    ----------------
    logger_name : str (default=`cleaners`)
    verbose : boolean, (default=True)
    fail_on_warning : boolean, (default=False)
    min_rows : int (default=10)
    allow_passthrough : bool (default=True)
    sample_df : dataframe (optional)
    sample_rate : float (optional)
    """

    def __init__(
        self,
        unique_thresh=6,
        feats=None,
        mandatory=None,
        feat_type_dct=None,
        feat_class_dct=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.unique_thresh = unique_thresh
        self.mandatory = mandatory or []
        self.feats = feats or []
        self.feat_type_dct = feat_type_dct
        self.feat_class_dct = feat_class_dct
        self.remaining_feats = []
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

    def fit(self, X, y=None, **kwargs):  # noqa : D102
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
    """
    Drop columns which are mostly NaNs.

    Parameters
    ----------
    mandatory : list[str]
    nan_frac_thresh : float (default=0.5)
    skip_if_missing : bool (default=True)

    Attributes
    ----------
    feature_names_in : list
    drop_cols_ : list
        list of columns to drop.

    Parameters
    ----------
    logger_name : str (default=`cleaners`)
    verbose : boolean, (default=True)
    fail_on_warning : boolean, (default=False)
    min_rows : int (default=10)
    allow_passthrough : bool (default=True)
    sample_df : dataframe (optional)
    sample_rate : float (optional)
    """

    def __init__(
        self,
        nan_frac_thresh=0.5,
        mandatory=None,
        skip_if_missing=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nan_frac_thresh = nan_frac_thresh
        self.mandatory = mandatory or []
        self.skip_if_missing = skip_if_missing
        self.drop_cols_ = []
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
            [x in self.mandatory for x in self.drop_cols_]
        ), "drop_cols and mandatory_feats overlap"
        missing_cols = [x for x in self.drop_cols_ if x not in X.columns]
        if not self.skip_if_missing and len(missing_cols) > 0:
            raise KeyError("missing one or more columns from dataframe: {}".format(missing_cols))
        else:
            self.drop_cols_ = [x for x in self.drop_cols_ if x in X.columns]

    def fit(self, X, y=None, **kwargs):  # noqa : D102
        self.log("dropping mostly NaN cols")
        self.get_sample_df(X)
        sz = self.sample_df.index.size
        cols = [x for x in self.sample_df.columns if x not in self.mandatory]

        nan_frac = self.sample_df[cols].isna().sum() / sz
        self.drop_cols_ += nan_frac[nan_frac > self.nan_frac_thresh].index.tolist()

        self.drop_cols_ = sorted(list(set(self.drop_cols_)))
        self.log("dropping {} columns".format(len(self.drop_cols_)))
        self.feature_names_in_ = cols
        return self


class HighCorrelationElim(BaseDropColsMixin, CleanerBase):
    """
    Drop one of each pairs of highly-correlated columns.

    Parameters
    ----------
    unique_thresh : int (default=6)
    rho_thresh : float (default=0.98)
    method : str (default='spearman')
    drop : bool (default=False)
    num_cols : list, optional
    mandatory : list, optional
    feat_class_dct : dict[str], optional
    feat_type_dct : dict[str], optional
    feats : list, optional

    Other Parameters
    ----------------
    logger_name : str (default=`cleaners`)
    verbose : boolean, (default=True)
    fail_on_warning : boolean, (default=False)
    min_rows : int (default=10)
    allow_passthrough : bool (default=True)
    sample_df : dataframe (optional)
    sample_rate : float (optional)
    """

    def __init__(
        self,
        unique_thresh=6,
        rho_thresh=0.98,
        method="spearman",
        drop=False,
        num_cols=None,
        mandatory=None,
        feat_class_dct=None,
        feat_type_dct=None,
        feats=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.unique_thresh = unique_thresh
        self.mandatory = mandatory or []
        self.feats = feats or []
        self.feat_type_dct = feat_type_dct
        self.feat_class_dct = feat_class_dct
        self.rho_thresh = rho_thresh
        self.method = method
        self.remaining_feats = None
        self.num_cols = num_cols or []
        self.drop = drop

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

    def fit(self, X, y=None, **kwargs):  # noqa : D102
        self.log("Checking for high correlation cols")
        self._set_defaults(X)
        self.drop_cols_ = eda.get_high_corr_cols(
            self.sample_df[self.num_cols], rho_thresh=self.rho_thresh, method=self.method
        )
        self.remaining_feats = list(
            set([x for x in X.columns if x not in self.drop_cols_] + list(self.mandatory))
        )
        self.log(
            f"{len(self.remaining_feats)} columns will remain after dropping {len(self.drop_cols_)} columns"
        )
        return self
