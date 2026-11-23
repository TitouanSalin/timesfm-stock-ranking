"""Canonical daily panel shared by every stage of the pipeline.

The WRDS extraction maps vendor columns to this schema, so nothing downstream
depends on CRSP table or column names.

Columns: permno, date, ret (total return, delisting included), prc (positive,
unadjusted close), dollar_vol (USD traded that day), mcap (USD), and optionally
half_spread (half the closing quoted spread, as a fraction of the mid price).

The final row of a delisted stock has a return but no price: the stock can no
longer be scored or traded, yet a holder still bears that return.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REQUIRED_COLUMNS = ("permno", "date", "ret", "prc", "dollar_vol", "mcap")


@dataclass(frozen=True)
class WidePanel:
    """One dates-by-permnos frame per variable, on the full trading calendar.

    Working on the calendar rather than on each stock's own rows keeps rolling
    windows aligned in time when a stock has gaps.
    """

    ret: pd.DataFrame
    prc: pd.DataFrame
    dollar_vol: pd.DataFrame
    mcap: pd.DataFrame
    half_spread: pd.DataFrame | None = None

    def subset(self, permnos: pd.Index) -> "WidePanel":
        """Restrict to some stocks. Frees memory once the universe is known."""
        return WidePanel(
            ret=self.ret[permnos],
            prc=self.prc[permnos],
            dollar_vol=self.dollar_vol[permnos],
            mcap=self.mcap[permnos],
            half_spread=None if self.half_spread is None else self.half_spread[permnos],
        )


def validate_panel(panel: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(panel.columns)
    if missing:
        raise ValueError(f"Panel is missing columns: {sorted(missing)}")
    n_duplicates = panel.duplicated(["permno", "date"]).sum()
    if n_duplicates:
        raise ValueError(f"Panel has {n_duplicates} duplicated (permno, date) rows")
    if (panel["ret"] < -1).any():
        raise ValueError("Panel has returns below -100%")
    if (panel["prc"] <= 0).any():
        raise ValueError("Panel has non-positive prices; CRSP bid/ask averages must be made positive")


def to_wide(panel: pd.DataFrame) -> WidePanel:
    validate_panel(panel)
    calendar = pd.DatetimeIndex(sorted(panel["date"].unique()))
    permnos = sorted(panel["permno"].unique())

    def pivot(column: str) -> pd.DataFrame:
        wide = panel.pivot(index="date", columns="permno", values=column)
        return wide.reindex(index=calendar, columns=permnos)

    return WidePanel(
        ret=pivot("ret"),
        prc=pivot("prc"),
        dollar_vol=pivot("dollar_vol"),
        mcap=pivot("mcap"),
        half_spread=pivot("half_spread") if "half_spread" in panel.columns else None,
    )
