import pytest
from dask.distributed import Client, LocalCluster
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def dask_client():
    """Fixture to create client."""
    cluster = LocalCluster(processes=False, scheduler_port=0, dashboard_address=None)
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


def pytest_addoption(parser):
    parser.addoption(
        "--run-regression",
        action="store_true",
        default=False,
        help="Run tests marked ``regression``.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-regression"):
        # --run-regression given in cli: do not skip slow tests
        return
    skip_test = pytest.mark.skip(reason="need --run-regression option to run")
    for item in items:
        if "regression" in item.keywords:
            item.add_marker(skip_test)


def pytest_configure(config):
    config.addinivalue_line("markers", "regression: mark as a regression test")


def pytest_runtest_setup(item):
    envnames = [mark.args[0] for mark in item.iter_markers(name="env")]
    if envnames:
        if item.config.getoption("-E") not in envnames:
            pytest.skip("test requires env in {!r}".format(envnames))


@pytest.fixture()
def make_pd_data():
    n_samples = 10000
    n_bin = 5
    n_cat = 5
    n_float = 5
    n_dupes = 5
    rs = np.random.RandomState(0)
    # make floats
    X = make_classification(n_samples=n_samples, n_features=n_float, random_state=0)[0]
    df = pd.DataFrame(X, columns=[f"var_{i}" for i in range(X.shape[1])])

    # make cats
    for i in range(n_cat):
        df[f"cat_{i}"] = rs.choice(
            ["a", "b", "c", "d", "e", np.nan], p=[0.2, 0.2, 0.2, 0.2, 0.15, 0.05], size=n_samples
        )
    # make binary
    for i in range(n_bin):
        df[f"bin_{i}"] = rs.choice([1.0, 0, np.nan], p=[0.45, 0.45, 0.1], size=n_samples)

    for x in rs.choice(df.columns.to_list(), size=n_dupes):
        df[x + "_dupe"] = df[x].copy()
    return df
