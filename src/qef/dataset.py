"""Assemble the long stock-date table used by every model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import market_features, market_return, stock_features
from .panel import WidePanel, to_wide
from .targets import forward_return, rank_by_date
from .universe import build_universe


@dataclass(frozen=True)
class Dataset:
    frame: pd.DataFrame            # one row per stock-date in the training universe
    ret: pd.DataFrame              # wide daily returns, for backtests
    trade_mask: pd.DataFrame       # wide tradable-universe membership
    stock_features: list[str]      # cross-sectionally ranked columns
    market_features: list[str]     # date-level columns
    half_spread: pd.DataFrame | None = None   # wide closing half spreads


def universe_rows(wide: WidePanel, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """Return the (rows, tradable) masks and the stocks that ever belong to the universe.

    A stock with no return or no price on a date, such as on its final delisting
    row, is neither scored nor ranked on that date. Every model uses this definition.
    """
    train_mask, trade_mask = build_universe(wide, **spec["universe"])
    members = train_mask.columns[train_mask.any(axis=0)]
    alive = wide.ret[members].notna() & wide.prc[members].notna()
    return train_mask[members] & alive, trade_mask[members] & alive, members


def build_dataset(panel: pd.DataFrame, spec: dict) -> Dataset:
    wide = to_wide(panel)
    # The market and the universes need every stock; everything else only its members.
    market = market_return(wide)
    rows, tradable, members = universe_rows(wide, spec)
    wide = wide.subset(members)

    target = spec["target"]
    forward = forward_return(wide.ret, target["horizon"], target["skip"])
    labelled = rows & forward.notna()
    excess = forward.sub(forward.where(labelled).mean(axis=1), axis=0)
    label_rank = rank_by_date(excess, labelled)

    row_index, column_index = np.nonzero(rows.to_numpy())
    calendar, permnos = wide.ret.index, wide.ret.columns

    columns: dict[str, np.ndarray] = {
        "date": calendar[row_index],
        "permno": permnos[column_index],
        "tradable": tradable.to_numpy()[row_index, column_index],
        "y_fwd": forward.to_numpy()[row_index, column_index],
        "y_excess": excess.to_numpy()[row_index, column_index],
        "y_rank": label_rank.to_numpy()[row_index, column_index],
    }

    stock_columns = []
    for name, values in stock_features(wide, market).items():
        column = f"f_{name}"
        columns[column] = rank_by_date(values, rows).to_numpy()[row_index, column_index].astype("float32")
        stock_columns.append(column)

    market_columns = []
    date_features = market_features(market)
    for name in date_features.columns:
        column = f"m_{name}"
        columns[column] = date_features[name].to_numpy()[row_index].astype("float32")
        market_columns.append(column)

    return Dataset(
        frame=pd.DataFrame(columns),
        ret=wide.ret,
        trade_mask=tradable,
        stock_features=stock_columns,
        market_features=market_columns,
        half_spread=wide.half_spread,
    )
