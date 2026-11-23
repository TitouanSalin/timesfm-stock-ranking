#!/usr/bin/env python
"""Download point-in-time Compustat, I/B/E/S and the link tables to data/raw/.

Conventions checked against WRDS on 2026-11-04:

- The value of a quarter in the first monthly snapshot that shows it equals
  coalesce(xr, x) of comp_pit.pitqtrdataus in over 99.7% of cases from 1998 to 2024.
  Column x alone only matches 60-70% before 2015, because it holds later values.
- The first snapshot is the minimum pointdate with qtrsback = 0 in
  comp_pit.pit_hist_date_tableus. It precedes the report date in only 0.03% of cases.
- The adjusted I/B/E/S table restates old estimates for later splits, so a ratio to the
  price of the time needs the unadjusted statsumu_epsus.

Each year goes to its own file, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ITEMS = ("atq", "ceqq", "saleq", "ibq", "niq", "cogsq", "oibdpq", "dlttq", "dlcq", "cheq", "actq", "lctq")
FIRST_YEAR, LAST_YEAR = 1992, 2025


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.rename(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default=os.environ.get("WRDS_USERNAME"), required="WRDS_USERNAME" not in os.environ)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    args = parser.parse_args()

    import wrds

    db = wrds.Connection(wrds_username=args.username)
    available = set(
        db.raw_sql(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'comp_pit' AND table_name = 'pitqtrdataus'"
        )["column_name"]
    )
    missing = [item for item in ITEMS if item not in available]
    if missing:
        raise SystemExit(f"Missing Compustat columns: {missing}")
    first_known = ", ".join(
        f"coalesce(q.{item}r, q.{item}) AS {item}" if f"{item}r" in available else f"q.{item} AS {item}" for item in ITEMS
    )

    queries = {
        "compustat_pit": lambda start, end: (
            "WITH firsts AS ("
            "  SELECT gvkey, datadate, min(pointdate) AS first_point FROM comp_pit.pit_hist_date_tableus"
            f" WHERE qtrsback = 0 AND datadate BETWEEN '{start}' AND '{end}' GROUP BY gvkey, datadate) "
            f"SELECT f.gvkey, f.datadate, f.first_point, q.rdqe, {first_known} "
            "FROM firsts f JOIN comp_pit.pitqtrdataus q ON f.gvkey = q.gvkey AND f.datadate = q.datadate"
        ),
        "ibes_consensus": lambda start, end: (
            "SELECT ticker, statpers, fpedats, numest, numup, numdown, meanest, stdev FROM ibes.statsumu_epsus "
            "WHERE fpi = '1' AND measure = 'EPS' AND fiscalp = 'ANN' AND curcode = 'USD' AND usfirm = 1 "
            f"AND statpers BETWEEN '{start}' AND '{end}'"
        ),
        "ibes_surprises": lambda start, end: (
            "SELECT ticker, anndats, surpmean, surpstdev, suescore FROM ibes.surpsum "
            "WHERE measure = 'EPS' AND fiscalp = 'QTR' AND usfirm = 1 "
            f"AND anndats BETWEEN '{start}' AND '{end}'"
        ),
    }

    counts: dict[str, int] = {}
    try:
        for name, query in queries.items():
            folder = args.out / name
            folder.mkdir(parents=True, exist_ok=True)
            total = 0
            for year in range(FIRST_YEAR, LAST_YEAR + 1):
                path = folder / f"{year}.parquet"
                if path.exists():
                    continue
                frame = db.raw_sql(query(f"{year}-01-01", f"{year}-12-31"))
                write_parquet(frame, path)
                total += len(frame)
                print(f"{name} {year}: {len(frame):,} rows", flush=True)
            counts[name] = total

        links = {
            "link_ccm": (
                "SELECT gvkey, lpermno AS permno, linktype, linkprim, linkdt, linkenddt FROM crsp.ccmxpf_lnkhist "
                "WHERE lpermno IS NOT NULL AND linktype IN ('LC', 'LU', 'LS') AND linkprim IN ('P', 'C')"
            ),
            "link_ibes": (
                "SELECT ticker, permno, sdate, edate, score FROM wrdsapps_link_crsp_ibes.ibcrsphist "
                "WHERE permno IS NOT NULL AND score <= 2"
            ),
        }
        for name, sql in links.items():
            frame = db.raw_sql(sql)
            write_parquet(frame, args.out / f"{name}.parquet")
            counts[name] = len(frame)
            print(f"{name}: {len(frame):,} rows", flush=True)
    finally:
        db.close()

    (args.out / "manifest_fundamentals.json").write_text(
        json.dumps(
            {
                "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "first_year": FIRST_YEAR,
                "last_year": LAST_YEAR,
                "items": ITEMS,
                "rows_this_run": counts,
            },
            indent=2,
        )
    )
    print("Extraction complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
