"""Accounting and analyst features must only use what was published before each date."""

import numpy as np
import pandas as pd
import pytest

from qef.fundamentals import (
    analyst_features,
    fundamental_features,
    join_latest,
    link_rows,
    prepare_consensus,
    prepare_quarters,
    prepare_surprises,
)


def quarters_raw(gvkey="A", n=9, start="2010-03-31", value=100.0, lag_months=2):
    ends = pd.date_range(start, periods=n, freq="QE")
    frame = pd.DataFrame(
        {
            "gvkey": gvkey,
            "datadate": ends,
            "first_point": ends + pd.offsets.MonthEnd(lag_months),
            "rdqe": ends + pd.Timedelta(days=30),
        }
    )
    for item in ("atq", "ceqq", "saleq", "ibq", "niq", "cogsq", "oibdpq", "dlttq", "dlcq", "cheq", "actq", "lctq"):
        frame[item] = value
    frame["ibq"] = 10.0
    frame["cogsq"] = 60.0
    return frame


LINK = pd.DataFrame({"gvkey": ["A"], "permno": [1.0], "linktype": ["LC"], "linkprim": ["P"], "linkdt": ["2000-01-01"], "linkenddt": [pd.NaT]})


def rows_on(dates, permno=1, mcap=1e9):
    return pd.DataFrame({"date": pd.to_datetime(dates), "permno": permno, "mcap": mcap})


# Availability


def test_a_quarter_is_invisible_until_after_its_first_snapshot():
    quarters = prepare_quarters(quarters_raw()).assign(position=lambda q: np.arange(len(q), dtype=float))
    point = quarters["first_point"].iloc[4]
    rows = rows_on([point, point + pd.Timedelta(days=1)]).assign(gvkey="A")
    seen = join_latest(rows, quarters, "gvkey", "available", ["position"], 400)
    # On the snapshot date itself, the previous quarter is still the newest one known.
    assert seen["position"].tolist() == [3.0, 4.0]


def test_later_quarters_do_not_change_earlier_features():
    raw = quarters_raw(n=12)
    cutoff = raw["first_point"].iloc[7]
    rows = rows_on([cutoff, cutoff - pd.Timedelta(days=40)])
    base = fundamental_features(rows, prepare_quarters(raw), LINK)
    changed = raw.copy()
    later = changed["first_point"] >= cutoff
    changed.loc[later, ["atq", "ceqq", "saleq", "ibq", "cogsq", "oibdpq"]] *= 7
    pd.testing.assert_frame_equal(base, fundamental_features(rows, prepare_quarters(changed), LINK))


def test_a_late_filed_quarter_delays_the_quarters_after_it():
    raw = quarters_raw(n=6)
    raw.loc[3, "first_point"] = raw.loc[5, "first_point"] + pd.Timedelta(days=60)
    quarters = prepare_quarters(raw)
    assert quarters.loc[4, "available"] == raw.loc[3, "first_point"]
    assert quarters.loc[5, "available"] == raw.loc[3, "first_point"]


def test_old_records_expire():
    quarters = prepare_quarters(quarters_raw(n=9))
    last = quarters["available"].iloc[-1]
    rows = rows_on([last + pd.Timedelta(days=30), last + pd.Timedelta(days=500)])
    features = fundamental_features(rows, quarters, LINK)
    assert features["roe"].notna().iloc[0]
    assert features.iloc[1].isna().all()


# Measures


def test_trailing_sum_needs_four_consecutive_quarters():
    raw = quarters_raw(n=7)
    quarters = prepare_quarters(raw.drop(index=2).reset_index(drop=True))
    # The three quarters after the gap cannot form a full year; the fourth one can.
    assert quarters["ttm_ib"].iloc[:5].isna().all()
    assert quarters["ttm_ib"].iloc[5] == pytest.approx(40.0)


def test_year_ago_comparison_needs_the_same_quarter():
    raw = quarters_raw(n=8)
    raw["atq"] = 100.0 * np.arange(1, 9)
    quarters = prepare_quarters(raw)
    assert quarters["asset_growth"].iloc[4] == pytest.approx(500 / 100 - 1)
    assert quarters["asset_growth"].iloc[:4].isna().all()


def test_ratios_on_known_values():
    quarters = prepare_quarters(quarters_raw(n=9))
    row = quarters.iloc[8]
    assert row["roe"] == pytest.approx(40 / 100)
    assert row["gross_profitability"] == pytest.approx(4 * 40 / 100)
    assert row["leverage"] == pytest.approx(2.0)


def test_price_ratios_use_compustat_millions_and_market_value_in_dollars():
    quarters = prepare_quarters(quarters_raw(n=9))
    rows = rows_on([quarters["available"].iloc[-1] + pd.Timedelta(days=1)], mcap=2e9)
    features = fundamental_features(rows, quarters, LINK)
    assert features["book_to_market"].iloc[0] == pytest.approx(100 / 2000)
    assert features["earnings_yield"].iloc[0] == pytest.approx(40 / 2000)


def test_share_classes_of_a_company_are_valued_together():
    quarters = prepare_quarters(quarters_raw(n=9))
    links = pd.concat([LINK, LINK.assign(permno=2.0)], ignore_index=True)
    date = quarters["available"].iloc[-1] + pd.Timedelta(days=1)
    rows = pd.DataFrame({"date": [date, date], "permno": [1, 2], "mcap": [1e9, 3e9]})
    features = fundamental_features(rows, quarters, links)
    assert features["book_to_market"].tolist() == pytest.approx([100 / 4000, 100 / 4000])


# Links


def test_link_applies_only_within_its_dates():
    links = pd.DataFrame({"permno": [1.0, 1.0], "gvkey": ["A", "B"], "linkdt": ["2000-01-01", "2010-01-01"], "linkenddt": ["2005-12-31", pd.NaT]})
    rows = rows_on(["1999-06-01", "2003-06-02", "2007-06-01", "2015-06-01"])
    assert link_rows(rows, links, "gvkey", "linkdt", "linkenddt").tolist()[1] == "A"
    linked = link_rows(rows, links, "gvkey", "linkdt", "linkenddt")
    assert linked.isna().tolist() == [True, False, True, False]
    assert linked.iloc[3] == "B"


def test_link_keeps_the_row_order_of_an_unsorted_table():
    links = pd.DataFrame({"permno": [1.0, 2.0], "gvkey": ["A", "B"], "linkdt": ["2000-01-01", "2000-01-01"], "linkenddt": [pd.NaT, pd.NaT]})
    rows = pd.DataFrame({"date": pd.to_datetime(["2010-01-05", "2010-01-04"]), "permno": [2, 1]}, index=[10, 20])
    linked = link_rows(rows, links, "gvkey", "linkdt", "linkenddt")
    assert linked.loc[10] == "B" and linked.loc[20] == "A"


# Analysts


IBES_LINK = pd.DataFrame({"ticker": ["T"], "permno": [1], "sdate": ["2000-01-01"], "edate": ["2030-01-01"], "score": [1.0]})


def consensus_raw(means=(1.0, 1.1, 1.2, 1.3), fiscal="2011-12-31"):
    dates = pd.to_datetime(["2011-01-20", "2011-02-17", "2011-03-17", "2011-04-21"])[: len(means)]
    return pd.DataFrame(
        {
            "ticker": "T",
            "statpers": dates,
            "fpedats": pd.Timestamp(fiscal),
            "numest": 10.0,
            "numup": 6.0,
            "numdown": 2.0,
            "meanest": list(means),
            "stdev": 0.2,
        }
    )


def prices_for(dates, prices):
    return pd.DataFrame({"date": pd.to_datetime(dates), "permno": 1, "prc": prices})


def test_estimates_are_compared_to_the_price_of_their_own_date():
    raw = consensus_raw()
    prices = prices_for(raw["statpers"], [10.0, 10.0, 20.0, 20.0])
    table = prepare_consensus(raw, IBES_LINK, prices)
    assert table["estimate_to_price"].tolist() == pytest.approx([0.1, 0.11, 0.06, 0.065])
    assert table["revision_1m"].iloc[1] == pytest.approx(0.01)
    assert table["revision_3m"].iloc[3] == pytest.approx(0.065 - 0.1)


def test_a_split_does_not_create_a_revision():
    raw = consensus_raw(means=(2.0, 1.0))  # Unadjusted estimate halves with a 2-for-1 split.
    table = prepare_consensus(raw, IBES_LINK, prices_for(raw["statpers"], [40.0, 20.0]))
    assert table["revision_1m"].iloc[1] == pytest.approx(0.0)


def test_a_new_fiscal_year_is_not_a_revision():
    raw = consensus_raw(means=(1.0, 1.5))
    raw.loc[1, "fpedats"] = pd.Timestamp("2012-12-31")
    table = prepare_consensus(raw, IBES_LINK, prices_for(raw["statpers"], [10.0, 10.0]))
    assert np.isnan(table["revision_1m"].iloc[1])


def test_consensus_counts_only_from_after_the_day_following_its_date():
    raw = consensus_raw(means=(1.0,))
    table = prepare_consensus(raw, IBES_LINK, prices_for(raw["statpers"], [10.0]))
    date = raw["statpers"].iloc[0]
    rows = pd.DataFrame({"date": [date, date + pd.Timedelta(days=1), date + pd.Timedelta(days=2)], "permno": [1, 1, 1]})
    surprises = prepare_surprises(pd.DataFrame({"ticker": ["T"], "anndats": ["2000-02-01"], "suescore": [1.0]}), IBES_LINK)
    features = analyst_features(rows, table, surprises)
    assert features["estimate_to_price"].isna().tolist() == [True, True, False]


def test_a_surprise_counts_only_after_its_announcement_day():
    raw = consensus_raw(means=(1.0,))
    table = prepare_consensus(raw, IBES_LINK, prices_for(raw["statpers"], [10.0]))
    surprises = prepare_surprises(pd.DataFrame({"ticker": ["T"], "anndats": ["2011-02-10"], "suescore": [2.5]}), IBES_LINK)
    rows = pd.DataFrame({"date": pd.to_datetime(["2011-02-10", "2011-02-11", "2011-07-01"]), "permno": [1, 1, 1]})
    features = analyst_features(rows, table, surprises)
    assert np.isnan(features["sue"].iloc[0])
    assert features["sue"].iloc[1] == pytest.approx(2.5)
    assert np.isnan(features["sue"].iloc[2])  # Expired after 100 days.


def test_a_ticker_outside_its_link_dates_is_ignored():
    links = IBES_LINK.assign(sdate="2015-01-01")
    raw = consensus_raw(means=(1.0,))
    assert prepare_consensus(raw, links, prices_for(raw["statpers"], [10.0])).empty
