"""Zero-shot TimesFM 3 forecasts turned into stock-date scores.

No model is fitted on financial data here, so a forecast at date t only depends on
the context, which stops at the close of t. The weights were pretrained elsewhere,
on data with cutoffs up to 2023, so an indirect link with later market history
cannot be fully excluded. Their license only allows non-commercial, non-production use.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

SERIES_KINDS = ("log_index", "returns")

# Positions of quantiles 0.1, 0.5 and 0.9 among the model's nine outputs.
Q10, Q50, Q90 = 0, 4, 8


def make_series(ret: pd.DataFrame, kind: str) -> np.ndarray:
    """Dates-by-stocks input series. Days without a return count as zero."""
    values = np.nan_to_num(ret.to_numpy(dtype=float), nan=0.0)
    if kind == "returns":
        return values
    if kind == "log_index":
        return np.cumsum(np.log1p(np.maximum(values, -0.9999)), axis=0)
    raise ValueError(f"Unknown series kind: {kind}")


def make_context(
    series: np.ndarray, first_quote: np.ndarray, date_position: int, stock: int, context_length: int
) -> np.ndarray:
    """The most recent context of a stock, up to and including this date."""
    start = max(int(first_quote[stock]), date_position - context_length + 1)
    return series[start : date_position + 1, stock].astype(np.float32)


def summarize_forecast(
    kind: str, forecast: np.ndarray, quantiles: np.ndarray, *, skip: int, horizon: int
) -> tuple[float, float, float]:
    """Return the expected target-period return, a width and an asymmetry of the forecast.

    Position 0 of a forecast is t+1, so the target period from the close of t+skip
    to the close of t+skip+horizon covers positions skip to skip+horizon-1 for daily
    returns, and is the change between positions skip-1 and skip+horizon-1 for an index.
    """
    if kind == "log_index":
        end = skip + horizon - 1
        expected = forecast[end] - forecast[skip - 1]
        q = quantiles[end]
        return float(expected), float(q[Q90] - q[Q10]), float(q[Q90] + q[Q10] - 2 * q[Q50])
    window = slice(skip, skip + horizon)
    q = quantiles[window]
    return (
        float(forecast[window].sum()),
        float((q[:, Q90] - q[:, Q10]).mean()),
        float((q[:, Q90] + q[:, Q10] - 2 * q[:, Q50]).mean()),
    )


def iter_predictions(
    forecaster,
    ret: pd.DataFrame,
    rows: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    kind: str,
    context_length: int,
    skip: int,
    horizon: int,
    chunk_size: int = 4096,
    use_symmetric_averaging: bool = True,
) -> Iterator[pd.DataFrame]:
    """Yield frames of (date, permno, tfm_pred, tfm_width, tfm_skew) for the masked rows."""
    series = make_series(ret, kind)
    quoted = ret.notna().to_numpy()
    first_quote = np.argmax(quoted, axis=0)
    mask = rows.reindex(index=ret.index, columns=ret.columns).fillna(False).to_numpy(dtype=bool)
    permnos = ret.columns.to_numpy()
    calendar = ret.index

    pending: list[tuple[int, int]] = []

    def flush() -> pd.DataFrame:
        contexts = [make_context(series, first_quote, t, j, context_length) for t, j in pending]
        outputs = forecaster.predict_batch(
            contexts,
            horizon=skip + horizon,
            return_quantiles=True,
            use_symmetric_averaging=use_symmetric_averaging,
            make_positive=False,
            sort_quantiles=True,
            univariate=True,
        )
        summaries = [
            summarize_forecast(kind, out.forecast, out.quantiles, skip=skip, horizon=horizon) for out in outputs
        ]
        result = pd.DataFrame(summaries, columns=["tfm_pred", "tfm_width", "tfm_skew"], dtype="float32")
        result.insert(0, "date", calendar[[t for t, _ in pending]])
        result.insert(1, "permno", permnos[[j for _, j in pending]])
        return result

    for date in dates:
        t = calendar.get_loc(date)
        for j in np.flatnonzero(mask[t]):
            pending.append((t, int(j)))
            if len(pending) >= chunk_size:
                yield flush()
                pending = []
    if pending:
        yield flush()


def add_timesfm_features(frame: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add ranked forecast columns to the dataset, ranking within each date like other features."""
    names = ["tfm_pred", "tfm_width", "tfm_skew"]
    merged = frame.merge(predictions[["date", "permno", *names]], on=["date", "permno"], how="left", validate="one_to_one")
    columns = []
    for name in names:
        column = f"f_{name}"
        merged[column] = (merged.groupby("date")[name].rank(pct=True) - 0.5).astype("float32")
        columns.append(column)
    return merged.drop(columns=names), columns
