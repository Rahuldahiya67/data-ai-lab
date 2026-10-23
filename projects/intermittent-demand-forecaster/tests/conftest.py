import pytest

from intermittent_demand_forecaster.generate import generate_demand


@pytest.fixture(scope="session")
def demand():
    return generate_demand(n_skus=12, n_weeks=80, seed=7)
