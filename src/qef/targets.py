"""Forward-return labels aligned with what can actually be executed."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(ret: pd.DataFrame, horizon: int, skip: int = 1) -> pd.DataFrame:
    """Total return from the close of t+skip to the close of t+skip+horizon.

    The label is NaN when the stock has no quote at t+skip (the trade could not
    happen) and on the last dates of the calendar (the outcome is unknown). A
    stock delisted during the holding period keeps its final return, delisting
    included, and the remaining days count as cash.
    """
    values = ret.to_numpy(dtype=float)
    log_returns = np.log1p(np.maximum(np.nan_to_num(values, nan=0.0), -0.9999))
    cumulative = np.cumsum(log_returns, axis=0)

    n_dates = len(values)
    end = skip + horizon
    result = np.full(values.shape, np.nan)
    if n_dates > end:
        result[: n_dates - end] = np.expm1(cumulative[end:] - cumulative[skip : n_dates - horizon])
        tradable = np.isfinite(values[skip : n_dates - horizon])
        result[: n_dates - end][~tradable] = np.nan
    return pd.DataFrame(result, index=ret.index, columns=ret.columns)


def rank_by_date(wide: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank within each date among masked stocks, centred on zero."""
    return wide.where(mask).rank(axis=1, pct=True) - 0.5
