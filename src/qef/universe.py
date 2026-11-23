"""Point-in-time universes, rebuilt each month from data known at that date."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import WidePanel


def month_end_positions(calendar: pd.DatetimeIndex) -> np.ndarray:
    """Position of the last trading day of each month in the calendar."""
    months = calendar.to_period("M")
    is_last = months[:-1] != months[1:]
    return np.flatnonzero(np.append(is_last, True))


def build_universe(
    wide: WidePanel,
    *,
    train_size: int,
    trade_pool: int,
    trade_size: int,
    min_price: float,
    min_history: int,
    adv_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return boolean (train, trade) frames.

    Membership chosen at a month-end applies from the next trading day through
    the next month-end, so a stock can never be selected using later data.
    """
    if not trade_size <= trade_pool <= train_size:
        raise ValueError("Expected trade_size <= trade_pool <= train_size")

    alive = wide.ret.notna()
    history = alive.cumsum()
    eligible = alive & (wide.prc >= min_price) & (history >= min_history) & wide.mcap.notna()

    calendar, permnos = wide.ret.index, wide.ret.columns
    train = np.zeros(wide.ret.shape, dtype=bool)
    trade = np.zeros(wide.ret.shape, dtype=bool)
    dollar_vol = wide.dollar_vol.to_numpy(dtype=float)

    positions = month_end_positions(calendar)
    for position, next_position in zip(positions[:-1], positions[1:]):
        caps = wide.mcap.iloc[position].where(eligible.iloc[position]).dropna()
        largest = caps.nlargest(train_size).index
        pool = caps.nlargest(trade_pool).index
        pool_columns = permnos.get_indexer(pool)

        # Median dollar volume up to this date, for the pool only.
        window = dollar_vol[max(position - adv_window + 1, 0) : position + 1, :][:, pool_columns]
        enough = np.isfinite(window).sum(axis=0) >= adv_window // 2
        median = np.full(len(pool), np.nan)
        if enough.any():
            median[enough] = np.nanmedian(window[:, enough], axis=0)
        liquid = pd.Series(median, index=pool).dropna().nlargest(trade_size).index

        active = slice(position + 1, next_position + 1)
        train[active, permnos.get_indexer(largest)] = True
        trade[active, permnos.get_indexer(liquid)] = True

    return (
        pd.DataFrame(train, index=calendar, columns=permnos),
        pd.DataFrame(trade, index=calendar, columns=permnos),
    )
