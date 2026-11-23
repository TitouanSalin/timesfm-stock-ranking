#!/usr/bin/env python
"""Run zero-shot TimesFM 3 over the panel, one Parquet shard per year.

Finished years are skipped, so an interrupted job can be resubmitted. Dates whose
target would reach the final test period are refused unless --allow-holdout is set.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qef.dataset import universe_rows  # noqa: E402
from qef.finetune import check_weights  # noqa: E402
from qef.panel import to_wide  # noqa: E402
from qef.splits import dates_before_holdout  # noqa: E402
from qef.timesfm_zero_shot import SERIES_KINDS, iter_predictions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=ROOT / "data" / "clean" / "panel.parquet")
    parser.add_argument("--spec", type=Path, default=ROOT / "config" / "spec.yaml")
    parser.add_argument("--series", choices=SERIES_KINDS, default="log_index")
    parser.add_argument("--universe", choices=("trade", "train"), default="trade")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--first-year", type=int, default=None)
    parser.add_argument("--last-year", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "predictions" / "timesfm")
    parser.add_argument("--allow-holdout", action="store_true")
    args = parser.parse_args()

    from timesfm3 import ModelConfig, TimesFM3Evaluator

    spec = yaml.safe_load(args.spec.read_text())
    plan, target = spec["walk_forward"], spec["target"]

    panel = pd.read_parquet(args.panel, columns=["permno", "date", "ret", "prc", "dollar_vol", "mcap"])
    wide = to_wide(panel)
    del panel
    rows, tradable, members = universe_rows(wide, spec)
    mask = tradable if args.universe == "trade" else rows
    ret = wide.ret[members]
    del wide

    purge = target["horizon"] + target["skip"]
    dates = ret.index
    if not args.allow_holdout:
        dates = dates_before_holdout(dates, plan["holdout_start"], purge)
    active = mask.index[mask.any(axis=1)]
    dates = dates[dates.isin(active)]
    if args.first_year:
        dates = dates[dates.year >= args.first_year]
    if args.last_year:
        dates = dates[dates.year <= args.last_year]
    print(f"{int(mask.loc[dates].to_numpy().sum()):,} forecasts on {len(dates):,} dates", flush=True)

    forecaster = TimesFM3Evaluator(
        ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", per_core_batch_size=args.batch_size, device="cuda")
    )
    check_weights(forecaster.model)

    out_dir = args.out / f"{args.series}_{args.universe}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in sorted(set(dates.year)):
        path = out_dir / f"{year}.parquet"
        if path.exists():
            print(f"{year}: already done", flush=True)
            continue
        started, n_forecasts = time.time(), 0
        frames = []
        for frame in iter_predictions(
            forecaster,
            ret,
            mask,
            dates[dates.year == year],
            kind=args.series,
            context_length=args.context_length,
            skip=target["skip"],
            horizon=target["horizon"],
        ):
            frames.append(frame)
            n_forecasts += len(frame)
        temporary = path.with_suffix(".tmp")
        pd.concat(frames, ignore_index=True).to_parquet(temporary, index=False, compression="zstd")
        temporary.rename(path)
        elapsed = time.time() - started
        print(f"{year}: {n_forecasts:,} forecasts in {elapsed:.0f}s ({n_forecasts / elapsed:.0f}/s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
