"""Extraction queries for CRSP (CIZ format) and conversion to the canonical panel.

Conventions below were checked against WRDS on 2026-11-03:

- dlycap and shrout are in thousands; dlyvol is in shares.
- A delisting return is NOT in the last trading row. It sits on a later, separate
  row with dlydelflg='Y', where dlyret equals stkdelists.delret. That row carries
  conditionaltype='N/A' and primaryexch='X', so a standard common-stock filter
  silently drops it, which would hide delisting losses. The query keeps it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Shumway (1997): typical return of a delisted stock whose delisting return is missing.
MISSING_DELISTING_RETURN = -0.30

# Common stocks of US corporations on a US exchange, as known on each date.
COMMON_STOCK_FILTER = (
    "sharetype = 'NS' AND securitytype = 'EQTY' AND securitysubtype = 'COM' AND usincflg = 'Y' "
    "AND issuertype IN ('ACOR', 'CORP') AND primaryexch IN ('N', 'A', 'Q', 'R', 'B', 'I') "
    "AND conditionaltype = 'RW' AND tradingstatusflg = 'A'"
)


def daily_query(start: str, end: str) -> str:
    return (
        "SELECT permno, dlycaldt, dlyret, dlyprc, dlybid, dlyask, dlyvol, dlycap, siccd, primaryexch, dlydelflg "
        "FROM crsp.dsf_v2 "
        f"WHERE dlycaldt BETWEEN '{start}' AND '{end}' "
        f"AND (dlydelflg = 'Y' OR ({COMMON_STOCK_FILTER}))"
    )


def nasdaq_volume_factor(dates: pd.Series) -> pd.Series:
    """Gao and Ritter (2010) divisors for inter-dealer double counting of Nasdaq volume."""
    factor = pd.Series(1.0, index=dates.index)
    factor[dates < "2004-01-01"] = 1 / 1.6
    factor[dates < "2002-01-01"] = 1 / 1.8
    factor[dates < "2001-02-01"] = 1 / 2.0
    return factor


def clean_daily(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Convert raw dsf_v2 rows to the canonical panel. Return it with counts for the report."""
    raw = raw.assign(date=pd.to_datetime(raw["dlycaldt"]))
    flagged = raw["dlydelflg"] == "Y"
    regular = raw[~flagged]
    delisting = raw[flagged & raw["permno"].isin(regular["permno"])]
    calendar = pd.DatetimeIndex(sorted(regular["date"].unique()))

    prc = regular["dlyprc"].astype(float).abs().where(lambda p: p > 0)
    volume = regular["dlyvol"].astype(float).where(lambda v: v >= 0)
    factor = pd.Series(1.0, index=regular.index)
    nasdaq = regular["primaryexch"] == "Q"
    factor[nasdaq] = nasdaq_volume_factor(regular.loc[nasdaq, "date"])
    bid, ask = regular["dlybid"].astype(float), regular["dlyask"].astype(float)
    valid_quote = (bid > 0) & (ask >= bid)

    panel = pd.DataFrame(
        {
            "permno": regular["permno"].astype("int32"),
            "date": regular["date"],
            "ret": regular["dlyret"].astype(float),
            "prc": prc,
            "dollar_vol": prc * volume * factor,
            "mcap": (regular["dlycap"].astype(float) * 1000).where(lambda c: c > 0),
            "half_spread": ((ask - bid) / (ask + bid)).where(valid_quote),
            "sic": regular["siccd"].astype("float32"),
        }
    )

    # A delisting row carries a return only, moved to the next session if needed.
    n_missing_return = int(delisting["dlyret"].isna().sum())
    positions = calendar.searchsorted(delisting["date"])
    in_range = positions < len(calendar)
    final = pd.DataFrame(
        {
            "permno": delisting["permno"].astype("int32").to_numpy()[in_range],
            "date": calendar[positions[in_range]],
            "ret": delisting["dlyret"].astype(float).fillna(MISSING_DELISTING_RETURN).to_numpy()[in_range],
        }
    )
    final = final.drop_duplicates(["permno", "date"])

    # If a stock also has a regular row on that session, both returns are compounded on it.
    keys = pd.MultiIndex.from_frame(panel[["permno", "date"]])
    final_keys = pd.MultiIndex.from_frame(final[["permno", "date"]])
    collides = final_keys.isin(keys)
    if collides.any():
        extra = final[collides].set_index(["permno", "date"])["ret"]
        rows = keys.get_indexer(extra.index)
        current = panel["ret"].to_numpy(copy=True)
        current[rows] = (1 + np.nan_to_num(current[rows])) * (1 + extra.to_numpy()) - 1
        panel["ret"] = current

    panel = pd.concat([panel, final[~collides]], ignore_index=True)
    panel = panel.sort_values(["date", "permno"], ignore_index=True)

    report = {
        "n_rows": int(len(panel)),
        "n_stocks": int(panel["permno"].nunique()),
        "n_dates": int(panel["date"].nunique()),
        "first_date": str(panel["date"].min().date()),
        "last_date": str(panel["date"].max().date()),
        "n_delisting_rows": int(len(delisting)),
        "n_delisting_missing_return": n_missing_return,
        "n_delisting_after_calendar": int((~in_range).sum()),
        "n_delisting_same_session_as_trade": int(collides.sum()),
        "n_flagged_without_regular_rows": int(flagged.sum() - len(delisting)),
    }
    return panel, report


def data_quality(panel: pd.DataFrame) -> pd.DataFrame:
    """Coverage by year. Only counts and missing rates, so it reveals nothing about model results."""
    year = panel["date"].dt.year
    grouped = panel.groupby(year)
    table = pd.DataFrame(
        {
            "rows": grouped.size(),
            "stocks": grouped["permno"].nunique(),
            "dates": grouped["date"].nunique(),
            "missing_ret": grouped["ret"].apply(lambda s: s.isna().mean()),
            "missing_prc": grouped["prc"].apply(lambda s: s.isna().mean()),
            "missing_mcap": grouped["mcap"].apply(lambda s: s.isna().mean()),
            "missing_volume": grouped["dollar_vol"].apply(lambda s: s.isna().mean()),
            "missing_spread": grouped["half_spread"].apply(lambda s: s.isna().mean()),
            "abs_ret_over_50pct": grouped["ret"].apply(lambda s: (s.abs() > 0.5).mean()),
        }
    )
    table.index.name = "year"
    return table
