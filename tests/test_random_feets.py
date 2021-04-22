import numpy as np
import pytest
from xgboost.dask import DaskXGBClassifier

from cleaners import random_feat_elim

from .make_data import make_date_data


@pytest.mark.regression
def test_random_feets_defaults(dask_client):
    ignore = ["target", "day_of_week"]
    ix_vars = ["date"]
    mandatory = ["1", "2", "3"]
    obj = random_feat_elim.RandomFeatureElimination(
        target_var="target",
        params={"seed": 0},
        model_class=DaskXGBClassifier,
        ix_vars=ix_vars,
        mandatory=mandatory,
        ignore=ignore,
        drop=True,
    )

    X = make_date_data(
        n_samples=10,
        n_features=100,
        n_informative=2,
        npartitions=2,
        to_dask=True,
        regressor=False,
    )

    obj._set_defaults(X)
    for k in ignore + ix_vars + ["target"]:
        assert k not in obj.initial_feats
    for k in mandatory:
        assert k in obj.initial_feats


@pytest.mark.skip
@pytest.mark.regression
def test_random_feets(dask_client):
    seed = 4130
    n_informative = 10
    num_periods = (
        2500  # this can fail if num_periods is too small,. b/c xgb won't pick up the signal.
    )
    ignore = ["target", "day_of_week"]
    ix_vars = ["date"]
    mandatory = ["1", "2", "3"]

    np.random.RandomState(seed)
    obj = random_feat_elim.RandomFeatureElimination(
        target_var="target",
        params={"seed": seed},
        model_class=DaskXGBClassifier,
        ix_vars=ix_vars,
        mandatory=mandatory,
        ignore=ignore,
        drop=True,
    )

    X = make_date_data(
        n_samples=num_periods,
        n_features=100,
        n_informative=n_informative,
        npartitions=4,
        to_dask=True,
        regressor=False,
        random_state=seed,
    )

    obj.fit(X)

    assert len(obj.remaining_feats) >= n_informative
    for k in ignore + ix_vars + ["target"]:
        assert k not in obj.remaining_feats
        assert k not in obj.feat_dct.keys()
