"""ABC for data cleaners"""


class CleanerBase:
    """ABC for data cleaners"""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        self.sample_rate = kwargs.get("sample_rate")
        self.sample_df = None

    def get_sample_df(self, X, random_state=0):
        """
        get data sample

        Parameters
        ----------
        X : dataframe
        random_state : int
        """
        if self.sample_rate:
            self.sample_df = X.sample(frac=self.sample_rate, random_state=random_state)
        else:
            self.sample_df = X
        if hasattr(self.sample_df, "compute"):
            self.sample_df = self.sample_df.compute()

    def _set_defaults(self, X):  # pylint: disable=unused-argument
        """dummy set defaults"""
        pass

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        """dummy fit method"""
        return self

    def transform(self, X):  # pylint: disable=no-self-use
        """dummy transform method"""
        return X
