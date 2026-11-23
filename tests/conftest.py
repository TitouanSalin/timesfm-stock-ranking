import copy

import pytest

from qef.synthetic import make_synthetic_panel

SMALL_SPEC = {
    "spec_version": 0,
    "target": {"horizon": 5, "skip": 1},
    "universe": {
        "train_size": 200,
        "trade_pool": 100,
        "trade_size": 50,
        "min_price": 1.0,
        "min_history": 252,
        "adv_window": 63,
    },
    "walk_forward": {
        "first_test_year": 2009,
        "last_dev_year": 2012,
        "holdout_start": "2013-01-01",
        "inner_val_years": 1,
    },
    "portfolio": {"n_positions": 10, "exit_rank": 20, "max_weight": 0.15},
    "costs_bps_per_side": {"central": 5.0},
    "evaluation": {
        "primary_cost": "central",
        "hac_lags": 10,
        "bootstrap_block": 21,
        "bootstrap_reps": 200,
        "bootstrap_seed": 1,
    },
}


@pytest.fixture()
def spec():
    return copy.deepcopy(SMALL_SPEC)


@pytest.fixture(scope="session")
def noise_panel():
    return make_synthetic_panel(signal=0.0, seed=11)


@pytest.fixture(scope="session")
def signal_panel():
    return make_synthetic_panel(signal=0.002, seed=12)
