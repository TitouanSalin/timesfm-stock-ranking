"""Ensembles and score smoothing."""

import numpy as np
import pandas as pd
import pytest

from qef.combine import combine_scores, resolve_ensemble, smooth_scores
from qef.portfolio import backtest


def frame(date, permnos, values):
    return pd.DataFrame({"date": date, "permno": permnos, "pred": values})


def test_models_on_different_scales_count_equally():
    tradable = pd.DataFrame({"date": ["d"] * 3, "permno": [1, 2, 3]})
    combined = combine_scores(
        {"large": frame("d", [1, 2, 3], [1000.0, 2000.0, 3000.0]), "small": frame("d", [1, 2, 3], [0.03, 0.02, 0.01])},
        tradable,
    )
    # Opposite rankings cancel out whatever the scale of the scores: no stock is preferred.
    assert combined["pred"].max() - combined["pred"].min() == pytest.approx(0.0, abs=1e-9)


def test_seeds_grouped_in_an_ensemble_count_as_one_source():
    tradable = pd.DataFrame({"date": ["d"] * 3, "permno": [1, 2, 3]})
    up, down = frame("d", [1, 2, 3], [1.0, 2.0, 3.0]), frame("d", [1, 2, 3], [3.0, 2.0, 1.0])
    scores = {"seed0": up, "seed1": up, "seed2": up, "other": down}
    ensembles = {"seeds": ["seed0", "seed1", "seed2"], "grouped": ["seeds", "other"]}
    # Listed separately, the three seeds would outvote the other model.
    combined = resolve_ensemble("grouped", ensembles, scores, tradable)
    assert combined["pred"].max() - combined["pred"].min() == pytest.approx(0.0, abs=1e-9)


def test_an_ensemble_may_list_a_model_of_its_own_name():
    tradable = pd.DataFrame({"date": ["d"] * 2, "permno": [1, 2]})
    scores = {"m": frame("d", [1, 2], [1.0, 2.0])}
    combined = resolve_ensemble("m", {"m": ["m"]}, scores, tradable)
    assert combined["pred"].tolist() == pytest.approx([0.0, 0.5])


def test_rows_stay_aligned_when_tradable_comes_from_a_filtered_table():
    """The real tradable table keeps the sparse index of the dataset it was filtered from."""
    tradable = pd.DataFrame({"date": ["d"] * 3, "permno": [1, 2, 3]}, index=[500, 700, 900])
    combined = combine_scores({"m": frame("d", [1, 2, 3], [1.0, 2.0, 3.0])}, tradable)
    assert combined["permno"].tolist() == [1, 2, 3]
    assert combined["pred"].tolist() == pytest.approx([1 / 3 - 0.5, 2 / 3 - 0.5, 0.5])


def test_only_tradable_stocks_are_ranked():
    tradable = pd.DataFrame({"date": ["d"] * 2, "permno": [1, 2]})
    combined = combine_scores({"m": frame("d", [1, 2, 3], [1.0, 2.0, 99.0])}, tradable)
    assert combined["permno"].tolist() == [1, 2]
    assert combined["pred"].tolist() == pytest.approx([0.0, 0.5])


def test_a_stock_missing_from_one_model_uses_the_others():
    tradable = pd.DataFrame({"date": ["d"] * 2, "permno": [1, 2]})
    combined = combine_scores({"a": frame("d", [1, 2], [1.0, 2.0]), "b": frame("d", [1], [5.0])}, tradable)
    assert combined.set_index("permno")["pred"][2] == pytest.approx(0.5)


def test_a_stock_no_model_scored_is_dropped():
    tradable = pd.DataFrame({"date": ["d"] * 2, "permno": [1, 2]})
    assert combine_scores({"a": frame("d", [1], [1.0])}, tradable)["permno"].tolist() == [1]


def long_scores(wide):
    return wide.rename_axis(index="date", columns="permno").stack().rename("pred").reset_index()


def test_smoothing_averages_past_ranks_only():
    dates = pd.bdate_range("2020-01-01", periods=4)
    wide = pd.DataFrame({1: [1.0, 2.0, 1.0, 2.0], 2: [2.0, 1.0, 2.0, 1.0]}, index=dates)
    result = smooth_scores(long_scores(wide), window=2).pivot(index="date", columns="permno", values="pred")
    assert result.loc[dates[0], 1] == pytest.approx(0.0)  # Rank 1 of 2 is 0.5 - 0.5.
    assert result.loc[dates[1], 1] == pytest.approx(0.25)  # Mean of ranks 0.0 and 0.5.


def test_smoothing_ignores_later_scores():
    dates = pd.bdate_range("2020-01-01", periods=30)
    rng = np.random.default_rng(0)
    wide = pd.DataFrame(rng.normal(size=(30, 5)), index=dates, columns=range(5))
    changed = wide.copy()
    changed.iloc[20:] = rng.normal(size=(10, 5))
    a = smooth_scores(long_scores(wide), 5).pivot(index="date", columns="permno", values="pred").iloc[:20]
    b = smooth_scores(long_scores(changed), 5).pivot(index="date", columns="permno", values="pred").iloc[:20]
    pd.testing.assert_frame_equal(a, b)


def test_a_stock_is_not_scored_on_a_date_it_had_no_score():
    dates = pd.bdate_range("2020-01-01", periods=3)
    wide = pd.DataFrame({1: [1.0, np.nan, 1.0], 2: [2.0, 2.0, 2.0], 3: [3.0, 3.0, 3.0]}, index=dates)
    result = smooth_scores(long_scores(wide), 5)
    assert not ((result["permno"] == 1) & (result["date"] == dates[1])).any()


def test_smoothing_reduces_turnover_on_noisy_scores():
    dates = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(1)
    columns = range(50)
    wide = pd.DataFrame(rng.normal(size=(400, 50)), index=dates, columns=columns)
    ret = pd.DataFrame(0.0, index=dates, columns=columns)
    book = {"n_positions": 10, "exit_rank": 20, "max_weight": 0.15, "cost_bps": 5}
    raw = backtest(wide, ret, **book)["turnover"].mean()
    smoothed = smooth_scores(long_scores(wide), 10).pivot(index="date", columns="permno", values="pred")
    assert backtest(smoothed, ret, **book)["turnover"].mean() < raw / 2
