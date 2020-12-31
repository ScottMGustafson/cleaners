"""ABC for data cleaners."""

import logging


class DataTooSmallForEDA(Exception):
    """Exception for when data too small for eda."""

    pass


class DaskDataFrameNotSampled(Exception):
    """Exception for unsampled dask data."""

    pass


class CleanerBase:
    """ABC for data cleaners."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        """
        Init for Cleaner base.

        Other Parameters
        ----------------
        logger_name : str, default=None
        sample_rate : float, default=None
        verbose : boolean, default=True
        fail_on_warning : boolean, default=False
        """
        self.logger = logging.getLogger(kwargs.get("logger_name"))
        self.sample_rate = kwargs.get("sample_rate")
        self.sample_df = None
        self.verbose = kwargs.get("verbose", True)
        self.fail_on_warning_ = bool(kwargs.get("fail_on_warning", False))

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

    def get_sample_df(self, X, random_state=0, min_rows=1000):
        """
        Get data sample from either pandas or dask dataframe.

        Parameters
        ----------
        X : dataframe
        random_state : int
        min_rows : int, default=1000
            if sample df has fewer rows than this number, raise a warning.
        """
        if self.sample_rate:
            self.sample_df = X.sample(frac=self.sample_rate, random_state=random_state)
        else:
            self.sample_df = X
        if hasattr(self.sample_df, "compute"):
            if not self.sample_rate:
                self.fail_on_warning(
                    "cleaners.cleaner_base.get_sample_df:\n"
                    + "Using entire dask collection as sample dataframe."
                    + "This means a single worker will have to handle "
                    + "the whole dataset. Is this what you want to do?"
                    + "If not, then specify a sample rate.",
                    exception=DaskDataFrameNotSampled,
                )
            self.sample_df = self.sample_df.compute()
            if self.sample_df.index.size < min_rows:
                self.fail_on_warning(
                    "cleaners.cleaner_base.get_sample_df:\n"
                    + "The sample dataframe is smaller than {} rows".format(min_rows)
                    + "This may not be large enough to adequately infer info about your data.",
                    exception=DataTooSmallForEDA,
                )

    def fail_on_warning(self, msg, exception=Exception):
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
        if self.fail_on_warning_:
            raise exception(msg)
        else:
            self.logger.warning(msg)

    def _set_defaults(self, X):  # pylint: disable=unused-argument
        """Dummy set defaults."""
        pass

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        """Dummy fit method."""
        return self

    def transform(self, X):  # pylint: disable=no-self-use
        """Dummy transform method."""
        return X
