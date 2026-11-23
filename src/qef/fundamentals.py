"""Accounting and analyst features, joined only on data published before each date.

A feature at date t uses a Compustat quarter only after the first monthly snapshot that
shows it, and an analyst figure only after its publication date. Old values expire, so a
company that stops reporting does not keep a stale score forever.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MAX_AGE_QUARTER_DAYS = 270
MAX_AGE_CONSENSUS_DAYS = 45
MAX_AGE_SURPRISE_DAYS = 100
FAR_FUTURE = pd.Timestamp("2099-12-31")


def link_rows(rows: pd.DataFrame, links: pd.DataFrame, identifier: str, start: str, end: str) -> pd.Series:
    """The identifier linked to each (date, permno) row on its date, or NaN without a valid link."""
    table = links[["permno", identifier, start, end]].dropna(subset=["permno", start]).copy()
    table["permno"] = table["permno"].astype("int64")
    table[start] = pd.to_datetime(table[start])
    table[end] = pd.to_datetime(table[end], errors="coerce").fillna(FAR_FUTURE)
    table = table.sort_values(start)

    left = rows[["date", "permno"]].assign(permno=lambda f: f["permno"].astype("int64"), _row=np.arange(len(rows))).sort_values("date")
    joined = pd.merge_asof(left, table, left_on="date", right_on=start, by="permno")
    joined.loc[~(joined["date"] <= joined[end]), identifier] = np.nan
    return joined.sort_values("_row")[identifier].set_axis(rows.index)


def join_latest(rows: pd.DataFrame, table: pd.DataFrame, by: str, available: str, columns: list[str], max_age_days: int) -> pd.DataFrame:
    """For each row, the newest table record available strictly before the row's date.

    Records older than max_age_days give NaN. Rows without an identifier give NaN.
    """
    left = rows[["date", by]].assign(_row=np.arange(len(rows)))
    known = left[left[by].notna()].sort_values("date")
    right = table[[by, available, *columns]].dropna(subset=[by, available]).sort_values(available)
    joined = pd.merge_asof(known, right, left_on="date", right_on=available, by=by, allow_exact_matches=False)
    expired = (joined["date"] - joined[available]).dt.days > max_age_days
    joined.loc[expired, columns] = np.nan
    result = pd.DataFrame(np.nan, index=rows.index, columns=columns, dtype=float)
    result.iloc[joined["_row"].to_numpy(), [result.columns.get_loc(c) for c in columns]] = joined[columns].to_numpy(dtype=float)
    return result


def prepare_quarters(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-company quarterly measures built from first-known values.

    A quarter is usable only once every quarter it depends on is known, so availability
    is the running maximum of first_point. Trailing sums need four consecutive quarters.
    """
    table = raw.copy()
    table["datadate"] = pd.to_datetime(table["datadate"])
    table["first_point"] = pd.to_datetime(table["first_point"])
    numeric = [c for c in table.columns if c not in ("gvkey", "datadate", "first_point", "rdqe")]
    table[numeric] = table[numeric].apply(pd.to_numeric, errors="coerce").astype(float)
    table = table.sort_values(["gvkey", "datadate"]).drop_duplicates(["gvkey", "datadate"], keep="last").reset_index(drop=True)
    table["available"] = table.groupby("gvkey")["first_point"].cummax()

    groups = table.groupby("gvkey", sort=False)
    month = table["datadate"].dt.year * 12 + table["datadate"].dt.month
    table["_month"] = month
    consecutive_4 = table["_month"] - groups["_month"].shift(3) == 9
    year_ago = table["_month"] - groups["_month"].shift(4) == 12

    def trailing(column: pd.Series) -> pd.Series:
        total = column.groupby(table["gvkey"], sort=False).rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
        return total.where(consecutive_4)

    def last_year(column: pd.Series) -> pd.Series:
        return column.groupby(table["gvkey"], sort=False).shift(4).where(year_ago)

    table["ttm_ib"] = trailing(table["ibq"])
    table["ttm_sale"] = trailing(table["saleq"])
    table["ttm_gross"] = trailing(table["saleq"] - table["cogsq"])
    table["ttm_oibdp"] = trailing(table["oibdpq"])
    debt = table["dlttq"].fillna(0) + table["dlcq"].fillna(0)
    table["net_debt"] = debt - table["cheq"].fillna(0)

    assets = table["atq"].where(table["atq"] > 0)
    assets_ago = last_year(table["atq"]).where(lambda a: a > 0)
    book = table["ceqq"].where(table["ceqq"] > 0)
    sales_ago = last_year(table["ttm_sale"]).where(lambda s: s > 0)

    table["roa"] = table["ttm_ib"] / assets_ago
    table["roe"] = table["ttm_ib"] / book
    table["gross_profitability"] = table["ttm_gross"] / assets
    table["leverage"] = debt / assets
    table["cash_assets"] = table["cheq"] / assets
    table["current_ratio"] = table["actq"] / table["lctq"].where(table["lctq"] > 0)
    table["asset_growth"] = table["atq"] / assets_ago - 1
    table["sales_growth"] = table["ttm_sale"] / sales_ago - 1
    table["earnings_change"] = (table["ibq"] - last_year(table["ibq"])) / assets_ago
    return table.drop(columns="_month")


QUARTER_RATIOS = [
    "roa", "roe", "gross_profitability", "leverage", "cash_assets",
    "current_ratio", "asset_growth", "sales_growth", "earnings_change",
]


def fundamental_features(rows: pd.DataFrame, quarters: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Raw accounting features for (date, permno, mcap) rows. mcap is in USD, Compustat in millions."""
    left = rows[["date", "permno", "mcap"]].copy()
    left["gvkey"] = link_rows(rows, links, "gvkey", "linkdt", "linkenddt")
    columns = QUARTER_RATIOS + ["ceqq", "ttm_ib", "ttm_sale", "ttm_oibdp", "net_debt"]
    latest = join_latest(left, quarters, "gvkey", "available", columns, MAX_AGE_QUARTER_DAYS)

    # A company with several share classes is valued across the classes in the panel.
    company_value = left.groupby(["date", "gvkey"], dropna=True)["mcap"].transform("sum")
    value = company_value.where(company_value > 0).fillna(left["mcap"]) / 1e6

    features = latest[QUARTER_RATIOS].copy()
    features["book_to_market"] = latest["ceqq"] / value
    features["earnings_yield"] = latest["ttm_ib"] / value
    features["sales_yield"] = latest["ttm_sale"] / value
    enterprise = (value + latest["net_debt"]).where(lambda e: e > 0)
    features["ebitda_yield"] = latest["ttm_oibdp"] / enterprise
    return features


def prepare_consensus(raw: pd.DataFrame, links: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Monthly analyst measures per stock.

    The estimate is unadjusted, so it is divided by the unadjusted price of its own date,
    which keeps it comparable through a later split. Revisions only compare estimates for
    the same fiscal year, otherwise a new fiscal year would look like a revision.
    """
    table = raw.copy()
    table["statpers"] = pd.to_datetime(table["statpers"])
    table["fpedats"] = pd.to_datetime(table["fpedats"])
    for column in ("numest", "numup", "numdown", "meanest", "stdev"):
        table[column] = pd.to_numeric(table[column], errors="coerce").astype(float)

    table = _link_by_ticker(table, links, "statpers")

    priced = pd.merge_asof(
        table.dropna(subset=["permno"]).assign(permno=lambda f: f["permno"].astype("int64")).sort_values("statpers"),
        prices[["date", "permno", "prc"]].assign(permno=lambda f: f["permno"].astype("int64")).dropna(subset=["prc"]).sort_values("date"),
        left_on="statpers",
        right_on="date",
        by="permno",
        tolerance=pd.Timedelta(days=5),
    ).drop(columns="date")
    priced = priced.sort_values(["permno", "statpers"]).drop_duplicates(["permno", "statpers"], keep="last").reset_index(drop=True)

    priced["estimate_to_price"] = priced["meanest"] / priced["prc"]
    groups = priced.groupby("permno", sort=False)

    def revision(months: int) -> pd.Series:
        same_year = groups["fpedats"].shift(months) == priced["fpedats"]
        recent = (priced["statpers"] - groups["statpers"].shift(months)).dt.days <= 32 * months
        return (priced["estimate_to_price"] - groups["estimate_to_price"].shift(months)).where(same_year & recent)

    priced["revision_1m"] = revision(1)
    priced["revision_3m"] = revision(3)
    priced["up_down_ratio"] = (priced["numup"] - priced["numdown"]) / priced["numest"].where(lambda n: n > 0)
    priced["dispersion"] = priced["stdev"] / priced["meanest"].abs().clip(lower=0.01)
    priced["log_analysts"] = np.log1p(priced["numest"])
    # The summary dated statpers is treated as usable only from the following day.
    priced["available"] = priced["statpers"] + pd.Timedelta(days=1)
    return priced


CONSENSUS_FEATURES = ["estimate_to_price", "revision_1m", "revision_3m", "up_down_ratio", "dispersion", "log_analysts"]


def prepare_surprises(raw: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    table = raw.copy()
    table["anndats"] = pd.to_datetime(table["anndats"])
    table["suescore"] = pd.to_numeric(table["suescore"], errors="coerce").astype(float)
    table = _link_by_ticker(table, links, "anndats").dropna(subset=["permno"])
    table["permno"] = table["permno"].astype("int64")
    # An announcement can come after the close, so it only counts from the next day.
    return table.sort_values(["permno", "anndats"]).drop_duplicates(["permno", "anndats"], keep="last")


def _link_by_ticker(table: pd.DataFrame, links: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Attach the permno linked to each I/B/E/S ticker on its date, or NaN without a valid link."""
    right = links[["ticker", "permno", "sdate", "edate"]].copy()
    right["sdate"] = pd.to_datetime(right["sdate"])
    right["edate"] = pd.to_datetime(right["edate"], errors="coerce").fillna(FAR_FUTURE)
    # Each row keeps its own order: merge_asof needs sorted dates.
    ordered = table.assign(_row=np.arange(len(table))).sort_values(date_column)
    joined = pd.merge_asof(ordered, right.sort_values("sdate"), left_on=date_column, right_on="sdate", by="ticker")
    joined.loc[~(joined[date_column] <= joined["edate"]), "permno"] = np.nan
    return joined.sort_values("_row").drop(columns=["_row", "sdate", "edate"]).reset_index(drop=True)


def analyst_features(rows: pd.DataFrame, consensus: pd.DataFrame, surprises: pd.DataFrame) -> pd.DataFrame:
    """Raw analyst features for (date, permno) rows."""
    left = rows[["date", "permno"]].assign(permno=lambda f: f["permno"].astype("int64"))
    features = join_latest(left, consensus, "permno", "available", CONSENSUS_FEATURES, MAX_AGE_CONSENSUS_DAYS)
    latest = join_latest(left, surprises, "permno", "anndats", ["suescore"], MAX_AGE_SURPRISE_DAYS)
    features["sue"] = latest["suescore"]
    return features


def _read_shards(folder: Path) -> pd.DataFrame:
    frame = pd.concat([pd.read_parquet(path) for path in sorted(folder.glob("*.parquet"))], ignore_index=True)
    return _plain_text(frame)


def _plain_text(frame: pd.DataFrame) -> pd.DataFrame:
    """Text keys come back as a pandas string type, which merge_asof cannot match with objects."""
    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.StringDtype):
            frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
    return frame


def add_fundamental_columns(frame: pd.DataFrame, panel: pd.DataFrame, raw: Path) -> tuple[pd.DataFrame, list[str], list[str], pd.DataFrame]:
    """Add ranked accounting and analyst features to the dataset table.

    Returns the table, the two lists of new columns, and a report of coverage and median
    raw values over tradable stocks, by period.
    """
    stocks = frame["permno"].unique()
    known = panel.loc[panel["permno"].isin(stocks), ["date", "permno", "prc", "mcap"]]
    rows = frame[["date", "permno"]].merge(known[["date", "permno", "mcap"]], on=["date", "permno"], how="left", validate="one_to_one")

    quarters = prepare_quarters(_read_shards(raw / "compustat_pit"))
    links_ccm = _plain_text(pd.read_parquet(raw / "link_ccm.parquet"))
    fundamentals = fundamental_features(rows, quarters, links_ccm)

    links_ibes = _plain_text(pd.read_parquet(raw / "link_ibes.parquet"))
    consensus = prepare_consensus(_read_shards(raw / "ibes_consensus"), links_ibes, known)
    surprises = prepare_surprises(_read_shards(raw / "ibes_surprises"), links_ibes)
    analysts = analyst_features(rows, consensus, surprises)

    result = frame.copy()
    fundamental_columns, analyst_columns = [], []
    for values, prefix, names in ((fundamentals, "f_fund_", fundamental_columns), (analysts, "f_ana_", analyst_columns)):
        for name in values.columns:
            column = f"{prefix}{name}"
            result[column] = (values[name].groupby(frame["date"].to_numpy()).rank(pct=True) - 0.5).to_numpy(dtype="float32")
            names.append(column)

    raw_values = pd.concat([fundamentals, analysts], axis=1)
    tradable = frame["tradable"].to_numpy()
    report_rows = raw_values[tradable]
    period = (frame.loc[tradable, "date"].dt.year // 5 * 5).to_numpy()
    report = pd.concat(
        {
            "coverage": report_rows.notna().groupby(period).mean().T,
            "median": report_rows.groupby(period).median().T,
        },
        axis=1,
    )
    return result, fundamental_columns, analyst_columns, report
