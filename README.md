# cleaners
data cleaners for DS projects in the scikit-learn pipeline format.

## Example usage
This follows the standard sklearn pipeline format.
See [this notebook](notebooks/example.ipynb) for some example usage.


## It's easy to write your own cleaners
Just follow the sklearn format.  The logger handling
```python
from cleaners.cleaner_base import CleanerBase

class YourCleaner(CleanerBase):
    def __init__(self, **kwargs):
        super(YourCleaner, self).__init__(**kwargs)
        # set stuff

    def fit(self, X, y=None):
        # if you need, write an optional fit method
        return self

    def transform(self, X):
        # transform is mandatory
        if self.verbose:  # this is inherited
            self.logger.info("use the standard logging interface with the attached logger")
        # do other stuff
        return X
```
