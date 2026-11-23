#!/usr/bin/env python
"""Download CRSP daily data, factors, the S&P 500 and the VIX to data/raw/.

Each CRSP year goes to its own Parquet file, so an interrupted run resumes where it
stopped. Run it where the WRDS connection has already been validated with Duo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qef.crsp import daily_query  # noqa: E402


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write through a temporary name, so a partial file never looks finished."""
    temporary = path.with_suffix(".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.rename(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default=os.environ.get("WRDS_USERNAME"), required="WRDS_USERNAME" not in os.environ)
    parser.add_argument("--spec", type=Path, default=ROOT / "config" / "spec.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    args = parser.parse_args()

    if not (Path.home() / ".pgpass").exists():
        print("No ~/.pgpass: run scripts/00_wrds_inventory.py once from a terminal.", file=sys.stderr)
        return 2

    import wrds

    plan = yaml.safe_load(args.spec.read_text())["walk_forward"]
    start, end = plan["panel_start"], plan["panel_end"]
    daily_dir = args.out / "crsp_daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    db = wrds.Connection(wrds_username=args.username)
    try:
        counts = {}
        for year in range(int(start[:4]), int(end[:4]) + 1):
            path = daily_dir / f"{year}.parquet"
            if path.exists():
                print(f"{year}: already extracted", flush=True)
                continue
            first, last = max(start, f"{year}-01-01"), min(end, f"{year}-12-31")
            frame = db.raw_sql(daily_query(first, last))
            write_parquet(frame, path)
            counts[str(year)] = len(frame)
            print(f"{year}: {len(frame):,} rows", flush=True)

        extras = {
            "ff_daily": (
                f"SELECT date, mktrf, smb, hml, rmw, cma, rf, umd FROM ff.fivefactors_daily "
                f"WHERE date BETWEEN '{start}' AND '{end}'"
            ),
            "sp500": f"SELECT caldt AS date, vwretd, sprtrn FROM crsp.dsp500_v2 WHERE caldt BETWEEN '{start}' AND '{end}'",
            "vix": f"SELECT date, vix FROM cboe.cboe WHERE date BETWEEN '{start}' AND '{end}'",
        }
        for name, sql in extras.items():
            frame = db.raw_sql(sql)
            write_parquet(frame, args.out / f"{name}.parquet")
            print(f"{name}: {len(frame):,} rows", flush=True)
    finally:
        db.close()

    manifest = {
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "crsp.dsf_v2, ff.fivefactors_daily, crsp.dsp500_v2, cboe.cboe",
        "start": start,
        "end": end,
        "query": daily_query(start, end),
        "rows_this_run": counts,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Extraction complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
