"""ABC for data cleaners."""

import logging

import dask.dataframe as dd
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.base import TransformerMixin, BaseEstimator


class DataTooSmallForEDA(Exception):
    """Exception for when data too small for eda."""

    pass


class DaskDataFrameNotSampled(Exception):
    """Exception for unsampled dask data."""

    pass


class CleanerBase(ABC, TransformerMixin, BaseEstimator):
    """ABC for data cleaners."""

    def __init__(
        self,
        allow_passthrough=True,
        min_rows=1000,
        fail_on_warning=False,
        verbose=True,
        logger_name="cleaners",
        sample_df=None,
        sample_rate=None,
        **kwargs,
    ):
        """
        Init for Cleaner base.

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
        # super().__init__(**kwargs)
        self.logger = logging.getLogger(logger_name)
        self.sample_rate = sample_rate
        self.sample_df = sample_df
        self.verbose = verbose
        self.fail_on_warning = fail_on_warning
        self.min_rows = min_rows
        self.allow_passthrough = allow_passthrough
        self.feature_names_in_ = None

    def log(self, msg, level="info"):
        """
        Optional logging wrapper.

        Sends logging message only if self.verbose is true fo info
        always sends warnings
        If you'd prefer standard python logging,
        use self.logger instead.

        Parameters
        ----------
        msg : str
        level : str
            if level is info, only sends messages on verbose.  sends all others regardless

        Returns
        -------
        None

        """
        if level == "info":
            if self.verbose:
                self.logger.info(msg)
        else:
            self.logger.info(msg)

    def _sample_pd(self, X, random_state=0):
        if self.sample_rate:
            self.sample_df = X.sample(frac=self.sample_rate, random_state=random_state)
        else:
            self.sample_df = X.copy(deep=True)

    def _sample_dd(self, X, random_state=0, partition_size="100MB"):
        if not self.sample_rate:
            self._fail_on_warning(
                "cleaners.cleaner_base.get_sample_df:\n"
                + "Using entire dask collection as sample dataframe."
                + "This means a single worker will have to handle "
                + "the whole dataset. Is this what you want to do?"
                + "If not, then specify a sample rate.",
                exception=DaskDataFrameNotSampled,
            )

            self.sample_df = X.compute()
        else:
            self.sample_df = (
                X.sample(frac=self.sample_rate, random_state=random_state)
                .repartition(partition_size=partition_size)
                .compute()
            )

    def get_sample_df(self, X, random_state=0, partition_size="100MB", min_rows=None):
        """
        Get data sample from either pandas or dask dataframe.

        Parameters
        ----------
        X : dataframe
        random_state : int
        min_rows : int, default=1000
            if sample df has fewer rows than this number, raise a warning.
        partition_size : str, default=100MB
            optional partition size for post-sample repartition.
        """
        if not min_rows:
            min_rows = self.min_rows
        if isinstance(X, pd.DataFrame):
            self._sample_pd(X, random_state=0)
        elif isinstance(X, dd.DataFrame):
            self._sample_dd(X, random_state=random_state, partition_size=partition_size)
        else:
            raise TypeError("Type: {} not supported".format(type(X)))
        if self.sample_df.index.size < min_rows:
            self._fail_on_warning(
                "cleaners.cleaner_base.get_sample_df:\n"
                + "The sample dataframe is smaller than {} rows".format(min_rows)
                + "This may not be large enough to adequately infer info about your data.",
                exception=DataTooSmallForEDA,
            )

    def _check_input_features(self, X):
        assert all(
            x in X.columns for x in self.feature_names_in_
        ), f"missing columns {self.feature_names_in_}"
        if not self.allow_passthrough:
            assert sorted(X.columns.tolist()) == sorted(
                self.feature_names_in_
            ), "Feature names mismatch."

    def _fail_on_warning(self, msg, exception=Exception):
        """
        Warn or raise an exception depending on the state of ``fail_on_warning_``.

        Parameters
        ----------
        msg : str
            error / warning message
        exception : Type, default=Exception
            exception to raise on failure

        Returns
        -------
        None
        """
        if self.fail_on_warning:
            raise exception(msg)
        else:
            self.logger.warning(msg)

    def _set_defaults(self, X):
        """Dummy set defaults."""
        pass

    def fit(self, X, y=None, **kwargs):
        """Dummy fit method."""
        self.feature_names_in_ = list(X.columns)
        return self

    @abstractmethod
    def transform(self, X):
        """Dummy transform method."""

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
        return input_features or self.feature_names_in_
