import pytest
from dask.distributed import Client, LocalCluster


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
