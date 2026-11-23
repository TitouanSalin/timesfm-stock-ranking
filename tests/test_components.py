"""Hand-checked behaviour of each building block."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from qef.crsp import MISSING_DELISTING_RETURN, clean_daily
from qef.dataset import build_dataset
from qef.features import stock_features
from qef.metrics import (
    block_bootstrap_ci,
    cross_sectional_ic,
    deflated_sharpe,
    factor_alpha,
    hac_t_stat,
    performance,
)
from qef.panel import WidePanel, to_wide, validate_panel
from qef.portfolio import MISSING_HALF_SPREAD, backtest
from qef.splits import group_folds, plan_folds, walk_forward_folds
from qef.synthetic import make_synthetic_panel
from qef.targets import forward_return, rank_by_date
from qef.universe import build_universe, month_end_positions
from qef.walkforward import trial_sharpe_variance

ROOT = Path(__file__).resolve().parents[1]


def frame(rows, start="2020-01-06"):
    rows = np.array(rows, dtype=float)
    return pd.DataFrame(rows, index=pd.bdate_range(start, periods=len(rows)), columns=range(rows.shape[1]))


# Targets


def test_forward_return_skips_a_day_and_compounds():
    ret = frame([[0.0], [9.0], [0.1], [0.1], [0.0], [0.0], [0.0], [0.5]])
    result = forward_return(ret, horizon=5, skip=1)
    assert result.iloc[0, 0] == pytest.approx(1.1 * 1.1 - 1)
    assert result.iloc[1, 0] == pytest.approx(1.1 * 1.5 - 1)
    assert result.iloc[2:].isna().all().all()


def test_forward_return_needs_a_quote_on_the_execution_day():
    ret = frame([[0.0], [np.nan], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1]])
    assert np.isnan(forward_return(ret, horizon=5, skip=1).iloc[0, 0])


def test_delisting_during_the_holding_period_keeps_the_final_return():
    ret = frame([[0.0], [0.0], [0.1], [-0.5], [np.nan], [np.nan], [np.nan], [np.nan]])
    assert forward_return(ret, horizon=5, skip=1).iloc[0, 0] == pytest.approx(1.1 * 0.5 - 1)


def test_rank_by_date_is_centred_and_respects_the_mask():
    values = frame([[1.0, 2.0, 3.0, 4.0]])
    mask = frame([[1, 1, 1, 0]]).astype(bool)
    ranks = rank_by_date(values, mask)
    assert ranks.iloc[0, 0] == pytest.approx(1 / 3 - 0.5)
    assert ranks.iloc[0, 2] == pytest.approx(0.5)
    assert np.isnan(ranks.iloc[0, 3])


# Panel


def test_panel_validation_rejects_bad_data():
    panel = make_synthetic_panel(n_stocks=3, n_days=400)
    with pytest.raises(ValueError, match="duplicated"):
        validate_panel(pd.concat([panel, panel.iloc[:1]]))
    with pytest.raises(ValueError, match="non-positive"):
        validate_panel(panel.assign(prc=-1.0))
    with pytest.raises(ValueError, match="missing"):
        validate_panel(panel.drop(columns="mcap"))


# Universe


def _wide_for_universe(caps, volume, days=70):
    index = pd.bdate_range("2020-01-01", periods=days)
    constant = lambda values: pd.DataFrame([values] * days, index=index, columns=range(len(values)), dtype=float)
    return WidePanel(
        ret=constant([0.0] * len(caps)),
        prc=constant([10.0] * len(caps)),
        dollar_vol=constant(volume),
        mcap=constant(caps),
    )


def test_month_ends_are_the_last_trading_day_of_each_month():
    calendar = pd.bdate_range("2020-01-27", "2020-03-03")
    assert list(calendar[month_end_positions(calendar)].strftime("%m-%d")) == ["01-31", "02-28", "03-03"]


def test_universe_applies_from_the_day_after_the_month_end():
    wide = _wide_for_universe(caps=[100, 90, 80, 70], volume=[1, 2, 9, 8])
    train, trade = build_universe(
        wide, train_size=3, trade_pool=2, trade_size=1, min_price=5, min_history=1, adv_window=5
    )
    first_end = month_end_positions(wide.ret.index)[0]
    assert not train.iloc[first_end].any()
    assert train.iloc[first_end + 1].tolist() == [True, True, True, False]
    # Only the two largest are in the pool, and the more liquid of them is tradable.
    assert trade.iloc[first_end + 1].tolist() == [False, True, False, False]


def test_universe_excludes_short_histories_and_low_prices():
    wide = _wide_for_universe(caps=[100, 90], volume=[1, 1])
    wide.prc.iloc[:, 0] = 1.0
    train, _ = build_universe(wide, train_size=2, trade_pool=2, trade_size=1, min_price=5, min_history=1, adv_window=5)
    assert not train[0].any() and train[1].any()
    train, _ = build_universe(wide, train_size=2, trade_pool=2, trade_size=1, min_price=0, min_history=10_000, adv_window=5)
    assert not train.any().any()


# Features


def test_past_return_uses_the_stock_own_history():
    panel = make_synthetic_panel(n_stocks=5, n_days=400, seed=1)
    wide = to_wide(panel)
    features = stock_features(wide)
    stock = wide.ret.columns[0]
    log_ret = np.log1p(wide.ret[stock].fillna(0))
    expected = log_ret.iloc[-5:].sum()
    assert features["ret_5"][stock].iloc[-1] == pytest.approx(expected)
    # The 12-1 momentum stops 21 days before the date.
    expected = log_ret.iloc[-252:-21].sum()
    assert features["mom_252_21"][stock].iloc[-1] == pytest.approx(expected)


def test_past_return_is_missing_before_enough_history():
    wide = to_wide(make_synthetic_panel(n_stocks=5, n_days=400, seed=1))
    features = stock_features(wide)
    assert features["ret_252"].iloc[:251].isna().all().all()


def test_beta_recovers_a_known_exposure():
    rng = np.random.default_rng(0)
    n_days = 600
    index = pd.bdate_range("2015-01-01", periods=n_days)
    market = rng.normal(0, 0.01, n_days)
    ret = pd.DataFrame({0: 2.0 * market + rng.normal(0, 0.001, n_days), 1: market + rng.normal(0, 0.001, n_days)}, index=index)
    constant = pd.DataFrame(1e9, index=index, columns=ret.columns)
    features = stock_features(WidePanel(ret=ret, prc=constant, dollar_vol=constant, mcap=constant))
    betas = features["beta_252"].iloc[-1]
    assert betas[0] / betas[1] == pytest.approx(2.0, abs=0.05)


# Splits


def _calendar():
    return pd.bdate_range("2000-01-03", "2009-12-31")


def test_every_phase_is_purged():
    folds = walk_forward_folds(
        _calendar(), first_test_year=2005, last_test_year=2007, purge=6, inner_val_years=2
    )
    calendar = _calendar()
    position = lambda date: calendar.get_loc(date)
    assert [f.test_year for f in folds] == [2005, 2006, 2007]
    for fold in folds:
        assert position(fold.train_dates[-1]) + 6 < position(fold.val_dates[0])
        assert position(fold.val_dates[-1]) + 6 < position(fold.test_dates[0])
        assert position(fold.refit_dates[-1]) + 6 < position(fold.test_dates[0])
        assert fold.val_dates[0].year == fold.test_year - 2


def test_grouped_folds_train_before_each_block_and_test_every_year():
    calendar = _calendar()
    folds = walk_forward_folds(calendar, first_test_year=2003, last_test_year=2009, purge=6, inner_val_years=2)
    blocks = group_folds(folds, 3)
    assert [block.test_year for block in blocks] == [2003, 2006, 2009]
    assert blocks[1].test_dates.year.unique().tolist() == [2006, 2007, 2008]
    tested = blocks[0].test_dates.append([block.test_dates for block in blocks[1:]])
    assert tested.equals(folds[0].test_dates.append([fold.test_dates for fold in folds[1:]]))
    for block in blocks:
        assert calendar.get_loc(block.refit_dates[-1]) + 6 < calendar.get_loc(block.test_dates[0])


def _plan_spec():
    return {
        "target": {"horizon": 5, "skip": 1},
        "walk_forward": {"first_test_year": 2004, "last_dev_year": 2007, "inner_val_years": 2, "holdout_start": "2008-01-01", "panel_end": "2009-12-31"},
    }


def test_development_folds_never_reach_the_final_test():
    calendar = _calendar()
    folds = plan_folds(calendar, _plan_spec())
    assert [fold.test_year for fold in folds] == [2004, 2005, 2006, 2007]
    holdout = calendar.searchsorted(pd.Timestamp("2008-01-01"))
    assert calendar.get_loc(folds[-1].test_dates[-1]) + 6 < holdout


def test_final_test_folds_cover_its_years_and_train_on_the_past():
    calendar = _calendar()
    folds = plan_folds(calendar, _plan_spec(), final_test=True)
    assert [fold.test_year for fold in folds] == [2008, 2009]
    assert folds[0].test_dates[0] == pd.Timestamp("2008-01-01")
    assert folds[-1].test_dates[-1] == calendar[-1]
    for fold in folds:
        assert calendar.get_loc(fold.refit_dates[-1]) + 6 < calendar.get_loc(fold.test_dates[0])


def test_the_holdout_is_refused_by_default():
    with pytest.raises(ValueError, match="holdout"):
        walk_forward_folds(
            _calendar(), first_test_year=2008, last_test_year=2008, purge=6, inner_val_years=1, holdout_start="2008-01-01"
        )


def test_no_dev_label_reaches_the_holdout():
    calendar = _calendar()
    folds = walk_forward_folds(
        calendar, first_test_year=2007, last_test_year=2007, purge=6, inner_val_years=1, holdout_start="2008-01-01"
    )
    last = calendar.get_loc(folds[0].test_dates[-1])
    assert last + 6 < calendar.searchsorted(pd.Timestamp("2008-01-01"))
    assert last + 7 == calendar.searchsorted(pd.Timestamp("2008-01-01"))


def test_the_holdout_can_be_opened_explicitly():
    folds = walk_forward_folds(
        _calendar(),
        first_test_year=2008,
        last_test_year=2008,
        purge=6,
        inner_val_years=1,
        holdout_start="2008-01-01",
        allow_holdout=True,
    )
    assert folds[0].test_dates[0].year == 2008


# Portfolio


def test_positions_earn_from_two_days_after_the_signal():
    ret = frame([[0, 0], [5, 0], [0.1, 0], [0.1, 0]])
    scores = frame([[1, 0], [1, 0], [np.nan, np.nan], [np.nan, np.nan]])
    result = backtest(scores, ret, n_positions=1, exit_rank=1, max_weight=1.0, cost_bps=0)
    # Bought at the close of day 1, so the +500% of that day is not earned.
    assert result["gross"].tolist() == pytest.approx([0.0, 0.1])


def test_costs_are_charged_on_both_sides():
    ret = frame([[0, 0]] * 4)
    scores = frame([[1, 0], [0, 1], [np.nan, np.nan], [np.nan, np.nan]])
    result = backtest(scores, ret, n_positions=1, exit_rank=1, max_weight=1.0, cost_bps=10)
    assert result["cost"].tolist() == pytest.approx([0.001, 0.002])
    assert result["turnover"].tolist() == pytest.approx([0.5, 1.0])
    assert result["net"].tolist() == pytest.approx([-0.001, -0.002])


def test_rank_buffer_avoids_trading_a_small_rank_change():
    ret = frame([[0, 0, 0]] * 5)
    scores = frame([[3, 2, 1], [2, 3, 1], [1, 2, 3], [np.nan] * 3, [np.nan] * 3])
    result = backtest(scores, ret, n_positions=1, exit_rank=2, max_weight=1.0, cost_bps=10)
    # Rank 2 keeps the holding; rank 3 finally sells it.
    assert result["turnover"].tolist() == pytest.approx([0.5, 0.0, 1.0])


def test_delisted_holding_goes_to_cash_after_its_final_return():
    ret = frame([[0, 0], [0, 0], [-0.5, 0], [np.nan, 0], [np.nan, 0]])
    scores = frame([[1, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [0, 0]])
    result = backtest(scores, ret, n_positions=1, exit_rank=1, max_weight=1.0, cost_bps=0)
    assert result.loc[ret.index[2], "gross"] == pytest.approx(-0.5)
    assert result.loc[ret.index[2], "cash"] == pytest.approx(1.0)
    assert result.loc[ret.index[3], "gross"] == pytest.approx(0.0)


def test_unquoted_stock_cannot_be_bought():
    ret = frame([[0, 0], [np.nan, 0], [0.5, 0.1], [0, 0]])
    scores = frame([[2, 1], [2, 1], [np.nan, np.nan], [np.nan, np.nan]])
    result = backtest(scores, ret, n_positions=1, exit_rank=1, max_weight=1.0, cost_bps=0)
    # The top stock has no quote on the execution day, so the next one is bought.
    assert result.loc[ret.index[2], "gross"] == pytest.approx(0.1)


def test_oversized_holding_is_cut_back_to_equal_weight():
    ret = frame([[0, 0], [0, 0], [1.0, 0], [0, 0]])
    scores = frame([[2, 1], [2, 1], [2, 1], [np.nan, np.nan]])
    result = backtest(scores, ret, n_positions=2, exit_rank=2, max_weight=0.6, cost_bps=0)
    # 50/50 grows to 2/3 and 1/3 on day 2, and the first is cut back to 50%.
    assert result.loc[ret.index[2], "cash"] == pytest.approx(2 / 3 - 0.5)
    assert result.loc[ret.index[2], "turnover"] == pytest.approx((2 / 3 - 0.5) / 2)


def test_scores_after_a_date_do_not_change_earlier_results():
    rng = np.random.default_rng(0)
    ret = pd.DataFrame(rng.normal(0, 0.01, (200, 30)), index=pd.bdate_range("2020-01-01", periods=200))
    scores = pd.DataFrame(rng.normal(size=(190, 30)), index=ret.index[:190])
    book = {"n_positions": 5, "exit_rank": 10, "max_weight": 0.3, "cost_bps": 5}
    base = backtest(scores, ret, **book)
    changed = scores.copy()
    changed.iloc[100:] = rng.normal(size=(90, 30))
    # A signal from day 100 trades at the close of 101, whose cost shows on 101 and return on 102.
    last_unchanged = ret.index[100]
    pd.testing.assert_frame_equal(base.loc[:last_unchanged], backtest(changed, ret, **book).loc[:last_unchanged])


# Metrics


def test_rank_ic_is_computed_within_each_date():
    data = pd.DataFrame(
        {
            "date": ["a"] * 4 + ["b"] * 4,
            "pred": [1, 2, 3, 4, 1, 2, 3, 4],
            "y": [10, 20, 30, 40, 40, 30, 20, 10],
        }
    )
    ic = cross_sectional_ic(data, "pred", "y")
    assert ic["a"] == pytest.approx(1.0)
    assert ic["b"] == pytest.approx(-1.0)


def test_constant_predictions_have_no_ic():
    data = pd.DataFrame({"date": ["a"] * 3, "pred": [1, 1, 1], "y": [1, 2, 3]})
    assert cross_sectional_ic(data, "pred", "y").empty


def test_hac_widens_the_error_for_overlapping_data():
    rng = np.random.default_rng(0)
    noise = rng.normal(0.02, 1, 5000)
    overlapped = pd.Series(noise).rolling(5).mean().dropna()
    naive = hac_t_stat(overlapped, lags=0)
    robust = hac_t_stat(overlapped, lags=10)
    assert robust < naive / 1.5


def test_bootstrap_interval_contains_the_mean():
    series = pd.Series(np.random.default_rng(0).normal(0.5, 1, 2000))
    low, high = block_bootstrap_ci(series, block=21, reps=500, seed=1)
    assert low < series.mean() < high
    assert low > 0


def test_performance_on_a_known_series():
    stats = performance(pd.Series([0.01, -0.5, 1.0]))
    assert stats["max_drawdown"] == pytest.approx(-0.5)


def test_deflated_sharpe_falls_with_more_trials():
    returns = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 2500))
    one = deflated_sharpe(returns, 1, 0.0)
    many = deflated_sharpe(returns, 200, 0.0005)
    assert 0 <= many < one <= 1


def test_trial_variance_covers_every_saved_run(tmp_path):
    rng = np.random.default_rng(2)
    study = tmp_path / "study"
    study.mkdir()
    expected = []
    for folder, name, drift in ((tmp_path, "a", 0.0), (tmp_path, "b", 0.002), (study, "c", -0.003)):
        benchmark = pd.Series(rng.normal(0, 0.01, 500))
        net = benchmark + rng.normal(drift, 0.01, 500)
        pd.DataFrame({"net_central": net, "benchmark": benchmark}).to_parquet(folder / f"{name}_daily.parquet")
        active = net - benchmark
        expected.append(active.mean() / active.std())
    assert trial_sharpe_variance([tmp_path, study], "central") == pytest.approx(np.var(expected, ddof=1))


# Fine-tuning runs


def _evaluation_script():
    path = ROOT / "scripts" / "42_finetune_evaluate.py"
    loader = importlib.util.spec_from_file_location("finetune_evaluate", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def test_runs_split_by_seed_are_merged():
    merge = _evaluation_script().merge_settings
    merged = merge([{"steps": 10000, "every": 3, "seeds": "0"}, {"steps": 10000, "every": 3, "seeds": "1"}])
    assert merged == {"steps": 10000, "every": 3, "seeds": "0,1"}


def test_runs_with_other_settings_are_refused():
    merge = _evaluation_script().merge_settings
    with pytest.raises(SystemExit):
        merge([{"steps": 10000, "seeds": "0"}, {"steps": 1000, "seeds": "1"}])


# Script


def test_baseline_script_runs_end_to_end(tmp_path, spec):
    panel_path = tmp_path / "panel.parquet"
    make_synthetic_panel(n_stocks=250, signal=0.002, seed=5).to_parquet(panel_path)
    spec_path = tmp_path / "spec.yaml"
    spec["walk_forward"].update(first_test_year=2010, last_dev_year=2010)
    spec_path.write_text(yaml.safe_dump(spec))
    registry = tmp_path / "registry.jsonl"

    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "10_run_baselines.py"),
            "--panel", str(panel_path),
            "--spec", str(spec_path),
            "--models", "momentum,ridge",
            "--out", str(tmp_path / "out"),
            "--series-out", str(tmp_path / "series"),
            "--registry", str(registry),
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    records = [json.loads(line) for line in registry.read_text().splitlines()]
    assert [r["model"] for r in records] == ["momentum", "ridge"]
    assert (tmp_path / "series" / "ridge_predictions.parquet").exists()
    assert records[1]["selections"][0]["test_year"] == 2010
    assert len(records[0]["code_hash"]) == 12

    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "20_compare_models.py"),
            "--models", "momentum,ridge",
            "--reference", "ridge",
            "--spec", str(spec_path),
            "--summaries", str(tmp_path / "out"),
            "--series", str(tmp_path / "series"),
            "--registry", str(registry),
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert "DSR" in done.stdout and "momentum" in done.stdout


# CRSP cleaning


def _raw_rows(rows):
    columns = ["permno", "dlycaldt", "dlyret", "dlyprc", "dlybid", "dlyask", "dlyvol", "dlycap", "siccd", "primaryexch", "dlydelflg"]
    return pd.DataFrame(rows, columns=columns)


def _regular(permno, date, ret=0.01, prc=10.0, bid=9.9, ask=10.1, vol=100.0, cap=1000.0, exch="N"):
    return (permno, date, ret, prc, bid, ask, vol, cap, 3571, exch, "N")


def _delisting(permno, date, ret):
    return (permno, date, ret, None, None, None, None, None, None, "X", "Y")


def test_units_and_spread_are_converted():
    panel, _ = clean_daily(_raw_rows([_regular(1, "2010-01-04")]))
    row = panel.iloc[0]
    assert row["mcap"] == pytest.approx(1_000_000)  # dlycap is in thousands
    assert row["dollar_vol"] == pytest.approx(1000)
    assert row["half_spread"] == pytest.approx(0.2 / 20)


def test_nasdaq_volume_is_deflated_only_before_2004():
    rows = [
        _regular(1, "2000-06-01", exch="Q"),
        _regular(1, "2001-06-01", exch="Q"),
        _regular(1, "2003-06-02", exch="Q"),
        _regular(1, "2004-06-01", exch="Q"),
        _regular(2, "2000-06-01", exch="N"),
    ]
    panel, _ = clean_daily(_raw_rows(rows))
    volume = panel.set_index(["permno", "date"])["dollar_vol"]
    assert volume[(1, pd.Timestamp("2000-06-01"))] == pytest.approx(1000 / 2)
    assert volume[(1, pd.Timestamp("2001-06-01"))] == pytest.approx(1000 / 1.8)
    assert volume[(1, pd.Timestamp("2003-06-02"))] == pytest.approx(1000 / 1.6)
    assert volume[(1, pd.Timestamp("2004-06-01"))] == pytest.approx(1000)
    assert volume[(2, pd.Timestamp("2000-06-01"))] == pytest.approx(1000)


def test_delisting_row_keeps_its_return_but_has_no_price():
    rows = [_regular(1, "2010-01-04"), _regular(2, "2010-01-05"), _delisting(1, "2010-01-05", -0.4)]
    panel, report = clean_daily(_raw_rows(rows))
    final = panel[(panel["permno"] == 1) & (panel["date"] == "2010-01-05")].iloc[0]
    assert final["ret"] == pytest.approx(-0.4)
    assert np.isnan(final["prc"]) and np.isnan(final["mcap"])
    assert report["n_delisting_rows"] == 1


def test_missing_delisting_return_gets_the_shumway_value():
    rows = [_regular(1, "2010-01-04"), _regular(2, "2010-01-05"), _delisting(1, "2010-01-05", None)]
    panel, report = clean_daily(_raw_rows(rows))
    final = panel[(panel["permno"] == 1) & (panel["date"] == "2010-01-05")].iloc[0]
    assert final["ret"] == pytest.approx(MISSING_DELISTING_RETURN)
    assert report["n_delisting_missing_return"] == 1


def test_delisting_of_a_stock_outside_the_panel_is_ignored():
    panel, report = clean_daily(_raw_rows([_regular(1, "2010-01-04"), _delisting(9, "2010-01-04", -0.4)]))
    assert set(panel["permno"]) == {1}
    assert report["n_flagged_without_regular_rows"] == 1


def test_delisting_on_a_non_trading_day_moves_to_the_next_session():
    rows = [_regular(1, "2010-01-08"), _regular(2, "2010-01-11"), _delisting(1, "2010-01-09", -0.4)]
    panel, _ = clean_daily(_raw_rows(rows))
    assert set(panel["date"].dt.strftime("%m-%d")) == {"01-08", "01-11"}
    final = panel[(panel["permno"] == 1) & (panel["date"] == "2010-01-11")].iloc[0]
    assert final["ret"] == pytest.approx(-0.4)


def test_delisting_on_a_trading_session_of_the_same_stock_is_compounded():
    rows = [_regular(1, "2010-01-04", ret=0.1), _delisting(1, "2010-01-04", -0.5)]
    panel, report = clean_daily(_raw_rows(rows))
    assert len(panel) == 1
    assert panel["ret"].iloc[0] == pytest.approx(1.1 * 0.5 - 1)
    assert report["n_delisting_same_session_as_trade"] == 1


def test_a_stock_on_its_delisting_row_is_not_scored_but_its_loss_is_kept(spec):
    panel = make_synthetic_panel(n_stocks=250, n_days=800, seed=8)
    last_date = panel.groupby("permno")["date"].transform("max")
    final = (panel["date"] == last_date) & (panel["date"] < panel["date"].max())
    assert final.any()
    panel.loc[final, ["prc", "mcap", "dollar_vol"]] = np.nan
    dataset = build_dataset(panel, spec)

    keys = panel.loc[final, ["permno", "date"]].merge(dataset.frame[["permno", "date"]], on=["permno", "date"])
    assert keys.empty
    assert dataset.ret.notna().sum().sum() > 0
    delisted = panel.loc[final & panel["permno"].isin(dataset.ret.columns)].iloc[0]
    assert dataset.ret.loc[delisted["date"], delisted["permno"]] == pytest.approx(delisted["ret"])


# Costs and factor regression


def test_measured_costs_use_each_stock_spread_on_the_trade_date():
    ret = frame([[0, 0]] * 4)
    scores = frame([[1, 0], [0, 1], [np.nan, np.nan], [np.nan, np.nan]])
    spreads = frame([[0.5, 0.5], [0.001, 0.5], [0.002, 0.003], [0.5, 0.5]])
    result = backtest(scores, ret, n_positions=1, exit_rank=1, max_weight=1.0, cost_bps=10, half_spread=spreads)
    # Day 1: buy stock 0 at its half spread plus the fee.
    assert result["cost"].iloc[0] == pytest.approx(0.001 + 0.001)
    # Day 2: sell stock 0 and buy stock 1, each at its own half spread plus the fee.
    assert result["cost"].iloc[1] == pytest.approx((0.002 + 0.001) + (0.003 + 0.001))


def test_unknown_spread_falls_back_to_a_prudent_value():
    ret = frame([[0.0]] * 2)
    scores = frame([[1.0], [np.nan]])
    spreads = frame([[np.nan]] * 2)
    result = backtest(scores, ret, n_positions=1, exit_rank=1, max_weight=1.0, cost_bps=0, half_spread=spreads)
    assert result["cost"].iloc[0] == pytest.approx(MISSING_HALF_SPREAD)


def test_factor_alpha_recovers_a_known_alpha_and_beta():
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2010-01-01", periods=4000)
    factors = pd.DataFrame({"mktrf": rng.normal(0, 0.01, 4000), "smb": rng.normal(0, 0.005, 4000)}, index=index)
    returns = 0.001 + 1.2 * factors["mktrf"] - 0.5 * factors["smb"] + pd.Series(rng.normal(0, 0.002, 4000), index=index)
    result = factor_alpha(returns, factors, hac_lags=5)
    assert result["alpha_annual"] == pytest.approx(0.252, abs=0.02)
    assert result["beta_mktrf"] == pytest.approx(1.2, abs=0.02)
    assert result["beta_smb"] == pytest.approx(-0.5, abs=0.03)
    assert result["alpha_t"] > 5
