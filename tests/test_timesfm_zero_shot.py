"""The TimesFM adapter, tested with a fake forecaster so that no GPU is needed."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qef.splits import dates_before_holdout
from qef.timesfm_zero_shot import (
    add_timesfm_features,
    iter_predictions,
    make_context,
    make_series,
    summarize_forecast,
)


class FakeForecaster:
    """Forecasts the last context value plus the step number, and records its inputs."""

    def __init__(self):
        self.contexts = []
        self.options = None

    def predict_batch(self, contexts, horizon, **options):
        self.options = options
        for context in contexts:
            self.contexts.append(context)
            forecast = context[-1] + np.arange(1, horizon + 1, dtype=float)
            quantiles = np.stack([forecast + q for q in np.linspace(-0.4, 0.4, 9)], axis=1)
            yield SimpleNamespace(forecast=forecast, quantiles=quantiles)


@pytest.fixture()
def ret():
    index = pd.bdate_range("2020-01-01", periods=30)
    values = np.random.default_rng(0).normal(0, 0.01, (30, 3))
    frame = pd.DataFrame(values, index=index, columns=[101, 102, 103])
    frame.iloc[:10, 2] = np.nan  # The third stock lists late.
    return frame


def test_series_are_cumulative_log_returns_or_raw_returns(ret):
    index = make_series(ret, "log_index")
    assert index[-1, 0] == pytest.approx(np.log1p(ret[101]).sum())
    assert make_series(ret, "returns")[0, 0] == pytest.approx(ret.iloc[0, 0])
    with pytest.raises(ValueError):
        make_series(ret, "prices")


def test_context_stops_at_the_date_and_starts_at_the_first_quote(ret):
    series = make_series(ret, "returns")
    first_quote = np.argmax(ret.notna().to_numpy(), axis=0)
    context = make_context(series, first_quote, date_position=15, stock=2, context_length=512)
    assert len(context) == 6  # Quoted from position 10 through 15.
    assert context[-1] == pytest.approx(ret.iloc[15, 2], abs=1e-6)
    assert len(make_context(series, first_quote, 15, 0, context_length=4)) == 4


def test_log_index_forecast_uses_the_change_over_the_target_period():
    forecast = np.array([10.0, 11, 12, 13, 14, 17])  # Positions 0 to 5 are t+1 to t+6.
    quantiles = np.stack([forecast + q for q in np.linspace(-1, 1, 9)], axis=1)
    expected, width, skew = summarize_forecast("log_index", forecast, quantiles, skip=1, horizon=5)
    assert expected == pytest.approx(17 - 10)
    assert width == pytest.approx(2.0)
    assert skew == pytest.approx(0.0)


def test_returns_forecast_excludes_the_day_before_execution():
    forecast = np.array([100.0, 1, 1, 1, 1, 1])  # The t+1 return happens before execution.
    quantiles = np.stack([forecast + q for q in np.linspace(-1, 1, 9)], axis=1)
    expected, width, _ = summarize_forecast("returns", forecast, quantiles, skip=1, horizon=5)
    assert expected == pytest.approx(5.0)
    assert width == pytest.approx(2.0)


def test_only_masked_rows_are_forecast_and_options_are_explicit(ret):
    rows = pd.DataFrame(False, index=ret.index, columns=ret.columns)
    rows.iloc[20, [0, 2]] = True
    rows.iloc[21, 1] = True
    forecaster = FakeForecaster()
    frames = list(
        iter_predictions(forecaster, ret, rows, ret.index[20:22], kind="log_index", context_length=512, skip=1, horizon=5, chunk_size=2)
    )
    result = pd.concat(frames, ignore_index=True)
    assert list(zip(result["date"], result["permno"])) == [
        (ret.index[20], 101),
        (ret.index[20], 103),
        (ret.index[21], 102),
    ]
    assert result["tfm_pred"].tolist() == pytest.approx([5.0, 5.0, 5.0])  # 6 steps minus 1.
    # Returns can be negative, so forecasts must not be clamped to positive values.
    assert forecaster.options["make_positive"] is False
    assert forecaster.options["univariate"] is True


def test_forecasts_ignore_data_after_their_date(ret):
    rows = pd.DataFrame(False, index=ret.index, columns=ret.columns)
    rows.iloc[20] = True

    def run(data):
        return pd.concat(iter_predictions(FakeForecaster(), data, rows, data.index[20:21], kind="returns", context_length=512, skip=1, horizon=5))

    changed = ret.copy()
    changed.iloc[21:] = 0.5
    pd.testing.assert_frame_equal(run(ret), run(changed))

    changed = ret.copy()
    changed.iloc[20] = 0.5
    assert not run(ret)["tfm_pred"].equals(run(changed)["tfm_pred"])


def test_forecast_columns_are_ranked_within_each_date():
    frame = pd.DataFrame(
        {"date": ["a", "a", "a", "b"], "permno": [1, 2, 3, 1], "f_x": [0.0, 0.0, 0.0, 0.0]}
    )
    forecasts = pd.DataFrame(
        {"date": ["a", "a"], "permno": [1, 2], "tfm_pred": [5.0, -5.0], "tfm_width": [1.0, 2.0], "tfm_skew": [0.0, 1.0]}
    )
    merged, columns = add_timesfm_features(frame, forecasts)
    assert columns == ["f_tfm_pred", "f_tfm_width", "f_tfm_skew"]
    assert merged["f_tfm_pred"].iloc[0] == pytest.approx(0.5)
    assert merged["f_tfm_pred"].iloc[1] == pytest.approx(0.0)
    assert merged["f_tfm_pred"].iloc[2:].isna().all()
    assert "tfm_pred" not in merged.columns


def test_last_allowed_date_keeps_the_label_before_the_holdout():
    calendar = pd.bdate_range("2021-12-01", "2022-01-31")
    dates = dates_before_holdout(calendar, "2022-01-03", purge=6)
    position = calendar.get_loc(dates[-1])
    assert calendar[position + 6] < pd.Timestamp("2022-01-03")
    assert calendar[position + 7] == pd.Timestamp("2022-01-03")
