#!/usr/bin/env python
"""Write an inventory of the accessible WRDS tables to reports/inventory/.

Run it from a terminal the first time, because WRDS asks for your username and
password and can send a Duo request. Accept its offer to create ~/.pgpass so that
later runs do not prompt. Credentials are never stored in this repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qef.wrds_inventory import collect_inventory, render_markdown  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default=os.environ.get("WRDS_USERNAME"))
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "inventory")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    if not (Path.home() / ".pgpass").exists() and not sys.stdin.isatty():
        print(
            "No ~/.pgpass and no interactive terminal: WRDS cannot ask for your credentials.\n"
            "Run this script once from a terminal and accept the creation of ~/.pgpass.",
            file=sys.stderr,
        )
        return 2

    import sqlalchemy as sa
    import wrds

    db = wrds.Connection(wrds_username=args.username) if args.username else wrds.Connection()
    try:
        db.connection.execute(sa.text(f"SET statement_timeout = {args.timeout_seconds * 1000}"))
        db.connection.commit()

        def query(sql: str):
            try:
                return db.raw_sql(sql)
            except Exception:
                db.connection.rollback()  # A failed statement would poison the next ones.
                raise

        report = collect_inventory(query, set(db.list_libraries()))
    finally:
        db.close()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "inventory.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (args.out / "inventory.md").write_text(render_markdown(report))
    print(f"Inventory written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
