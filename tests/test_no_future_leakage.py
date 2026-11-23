"""The pipeline must not use the future, and a placebo must show no skill."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from qef.dataset import build_dataset
from qef.models import LightGBMAverage, LightGBMScore, RidgeScore
from qef.portfolio import backtest, equal_weight_benchmark
from qef.splits import walk_forward_folds
from qef.synthetic import make_synthetic_panel
from qef.walkforward import evaluate, run_walk_forward

CUTOFF = pd.Timestamp("2009-06-15")


def perturb_after(panel: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Replace every market value after the cutoff, keeping the same stock-dates."""
    rng = np.random.default_rng(99)
    changed = panel.copy()
    late = changed["date"] > cutoff
    n_late = int(late.sum())
    changed.loc[late, "ret"] = rng.normal(0, 0.05, n_late)
    changed.loc[late, "prc"] *= rng.uniform(1.5, 3.0, n_late)
    changed.loc[late, "mcap"] *= rng.uniform(0.2, 5.0, n_late)
    changed.loc[late, "dollar_vol"] *= rng.uniform(0.2, 5.0, n_late)
    return changed


def folds_for(dataset, spec, last_test_year=None):
    plan = spec["walk_forward"]
    return walk_forward_folds(
        dataset.ret.index,
        first_test_year=plan["first_test_year"],
        last_test_year=last_test_year or plan["last_dev_year"],
        purge=spec["target"]["horizon"] + spec["target"]["skip"],
        inner_val_years=plan["inner_val_years"],
        holdout_start=plan["holdout_start"],
    )


@pytest.fixture(scope="module")
def datasets():
    import copy

    from conftest import SMALL_SPEC

    spec = copy.deepcopy(SMALL_SPEC)
    panel = make_synthetic_panel(n_stocks=300, signal=0.001, seed=3)
    return spec, build_dataset(panel, spec), build_dataset(perturb_after(panel, CUTOFF), spec)


def test_features_and_universes_ignore_later_data(datasets):
    _, original, changed = datasets
    columns = ["date", "permno", "tradable", *original.stock_features, *original.market_features]
    before = original.frame[original.frame["date"] <= CUTOFF][columns].reset_index(drop=True)
    after = changed.frame[changed.frame["date"] <= CUTOFF][columns].reset_index(drop=True)
    assert len(before) > 0
    assert_frame_equal(before, after)


def test_later_data_does_change_later_features(datasets):
    """Guards the test above against passing because the perturbation had no effect."""
    _, original, changed = datasets
    day = original.ret.index[original.ret.index.searchsorted(CUTOFF) + 2]
    a = original.frame[original.frame["date"] == day].set_index("permno")["f_ret_1"]
    b = changed.frame[changed.frame["date"] == day].set_index("permno")["f_ret_1"]
    joined = pd.concat([a, b], axis=1, join="inner")
    assert not np.allclose(joined.iloc[:, 0], joined.iloc[:, 1])


def test_labels_are_only_known_six_days_later(datasets):
    _, original, changed = datasets
    calendar = original.ret.index
    cutoff_position = calendar.get_loc(CUTOFF)

    def labels(dataset, date):
        rows = dataset.frame[dataset.frame["date"] == date].set_index("permno")
        return rows[["y_fwd", "y_excess", "y_rank"]].dropna()

    known = calendar[cutoff_position - 6]
    assert_frame_equal(labels(original, known), labels(changed, known))

    unknown = calendar[cutoff_position - 5]
    a, b = labels(original, unknown), labels(changed, unknown)
    assert not np.allclose(a["y_fwd"], b["y_fwd"].reindex(a.index))


def test_predictions_ignore_data_after_their_date(datasets):
    """End to end: scores up to a date cannot change when anything later changes."""
    spec, original, changed = datasets
    features = original.stock_features

    def predictions(dataset):
        folds = folds_for(dataset, spec, last_test_year=spec["walk_forward"]["first_test_year"])
        scores, _ = run_walk_forward(dataset, RidgeScore, folds, features, log=lambda _: None)
        return scores[scores["date"] <= CUTOFF].reset_index(drop=True)

    before, after = predictions(original), predictions(changed)
    assert len(before) > 0
    assert_frame_equal(before, after)


def _returns_for_timing_test() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2015-01-01", periods=1000)
    return pd.DataFrame(rng.normal(0, 0.02, (1000, 60)), index=dates, columns=range(60))


def test_backtest_cannot_profit_from_the_return_before_execution():
    """A score revealing r(t+1) is useless, because execution happens at the close of t+1."""
    ret = _returns_for_timing_test()
    book = {"n_positions": 5, "exit_rank": 5, "max_weight": 1.0, "cost_bps": 0.0}
    result = backtest(ret.shift(-1).iloc[:-5], ret, **book)
    assert abs(result["gross"].mean()) * 252 < 0.10


def test_backtest_earns_the_first_return_after_execution():
    """A score revealing r(t+2) must be very profitable, which proves the alignment."""
    ret = _returns_for_timing_test()
    book = {"n_positions": 5, "exit_rank": 5, "max_weight": 1.0, "cost_bps": 0.0}
    result = backtest(ret.shift(-2).iloc[:-5], ret, **book)
    assert result["gross"].mean() * 252 > 1.0


MODEL_FACTORIES = [
    RidgeScore,
    lambda: LightGBMScore(grid={"num_leaves": (7,)}, rounds=(20, 50), num_threads=4),
    lambda: LightGBMAverage(grid={"num_leaves": (7, 15)}, rounds=(20, 50), num_threads=4),
]


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
def test_placebo_on_noise_shows_no_skill(noise_panel, spec, factory):
    dataset = build_dataset(noise_panel, spec)
    scores, _ = run_walk_forward(dataset, factory, folds_for(dataset, spec), dataset.stock_features, log=lambda _: None)
    results, _ = evaluate(scores, dataset, spec)
    assert abs(results["signal"]["rank_ic"]) < 0.02
    assert abs(results["signal"]["rank_ic_t"]) < 3
    assert results["portfolio"]["central"]["active_t"] < 2


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
def test_pipeline_finds_a_planted_signal(signal_panel, spec, factory):
    dataset = build_dataset(signal_panel, spec)
    scores, _ = run_walk_forward(dataset, factory, folds_for(dataset, spec), dataset.stock_features, log=lambda _: None)
    results, _ = evaluate(scores, dataset, spec)
    assert results["signal"]["rank_ic"] > 0.05
    assert results["signal"]["rank_ic_t"] > 3
    assert results["portfolio"]["central"]["ann_active"] > 0


def test_benchmark_uses_the_previous_close_membership():
    dates = pd.bdate_range("2020-01-01", periods=3)
    ret = pd.DataFrame({1: [0.1, 0.2, 0.3], 2: [0.5, 0.6, 0.7]}, index=dates)
    mask = pd.DataFrame({1: [True, True, True], 2: [False, True, True]}, index=dates)
    benchmark = equal_weight_benchmark(ret, mask)
    assert np.isnan(benchmark.iloc[0])
    assert benchmark.iloc[1] == pytest.approx(0.2)
    assert benchmark.iloc[2] == pytest.approx(0.5)
