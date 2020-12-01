# cleaners
data cleaners for DS projects in the scikit-learn pipeline format.

## Example usage
This follows the standard sklearn pipeline format.  

```python
from cleaners import data_types, drop_replace, eliminate_feats
from cleaners.indicators import AddIndicators
from cleaners.random_feat_elim import RandomFeatureElimination
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline


# assemble pipeline steps
example_pipeline_steps = [
    ("pre_drop", drop_replace.DropNamedCol(drop_cols=["cols", "to", "drop"])),
    # `verbose` and `logger_name` are optional params passed to the base class, 
    # which has a logging.Logger.  If you already have one, just pass the name.
    ("fix_types", data_types.FixDTypes(verbose=True, logger_name="some existing logger")),
    ("replace_bad_names", drop_replace.ReplaceBadColnameChars()),
    (
        "drop_mostly_nan",
        eliminate_feats.DropMostlyNaN(
            nan_frac_thresh=0.5,
            mandatory=["mandatory_columns", "to_keep"],
            skip_if_missing=True,
            apply_score_transform=False,
            sample_rate=0.1,
        ),
    ),
    ("drop_nan_target", drop_replace.DropNa(subset=["dont_drop", "NaNs_from_these_columns"])),
    (
        "drop_uninformative_1",
        eliminate_feats.DropUninformative(mandatory=["mandatory_columns", "to_keep"], sample_rate=0.5),
    ),
    ("indicators", AddIndicators(ignore=["columns_to", "ignore_indicators"])),
    (
        "gbm_feature_elim_1",
        RandomFeatureElimination(
            target_var="target_var",
            params={"max_depth":4, "min_child_weight":1},
            model_class=XGBClassifier,
            ix_vars=["index_var", ],
            mandatory=["mandatory_columns", "to_keep"],
            ignore=["ignore_me", "ignore_me2"],
            drop=True,
            sample_rate=0.123,
        ),
    ),
]

# run the pipeline
your_pipeline = Pipeline(example_pipeline_steps)
full_df = your_pipeline.fit_transform(your_dataframe)
```

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