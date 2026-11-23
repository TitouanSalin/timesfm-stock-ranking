#!/usr/bin/env python
"""Build data/clean/panel.parquet from the raw extraction and write a coverage report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qef.crsp import clean_daily, data_quality  # noqa: E402
from qef.panel import validate_panel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "clean" / "panel.parquet")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "data_quality")
    args = parser.parse_args()

    files = sorted((args.raw / "crsp_daily").glob("*.parquet"))
    if not files:
        print("No raw CRSP files found.", file=sys.stderr)
        return 2
    raw = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    panel, counts = clean_daily(raw)
    del raw
    validate_panel(panel)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.out, index=False, compression="zstd")

    coverage = data_quality(panel)
    args.report.mkdir(parents=True, exist_ok=True)
    (args.report / "counts.json").write_text(json.dumps(counts, indent=2))
    coverage.to_csv(args.report / "coverage_by_year.tsv", sep="\t", float_format="%.5f")
    print(json.dumps(counts, indent=2))
    print(coverage.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
