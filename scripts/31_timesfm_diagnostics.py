#!/usr/bin/env python
"""Check that zero-shot TimesFM outputs are meaningful, and what they relate to.

For each input series, reports within-date Spearman correlations of the forecast with
past returns, and of the forecast width with realized and past volatility. A width
that anticipates volatility shows the adapter passes sensible inputs to the model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qef.metrics import cross_sectional_ic, hac_t_stat  # noqa: E402


def stack(wide: pd.DataFrame, name: str) -> pd.DataFrame:
    return wide.rename_axis(index="date", columns="permno").stack().rename(name).reset_index()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=ROOT / "data" / "clean" / "panel.parquet")
    parser.add_argument("--forecasts", type=Path, default=ROOT / "data" / "predictions" / "timesfm")
    parser.add_argument("--runs", default="log_index_trade,returns_trade")
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel, columns=["permno", "date", "ret"])
    ret = panel.pivot(index="date", columns="permno", values="ret")
    del panel

    # Mean absolute return over the target period t+2 to t+6.
    future_vol = ret.abs().rolling(5, min_periods=3).mean().shift(-6)
    past_vol = ret.rolling(21, min_periods=10).std()
    log_index = np.log1p(ret.clip(lower=-0.9999).fillna(0.0)).cumsum()
    measures = {
        "future_vol": future_vol,
        "past_vol_21": past_vol,
        "past_ret_5": log_index - log_index.shift(5),
        "past_ret_21": log_index - log_index.shift(21),
        "momentum_12_1": log_index.shift(21) - log_index.shift(252),
    }

    pairs = [
        ("tfm_pred", "past_ret_5"),
        ("tfm_pred", "past_ret_21"),
        ("tfm_pred", "momentum_12_1"),
        ("tfm_width", "past_vol_21"),
        ("tfm_width", "future_vol"),
        ("past_vol_21", "future_vol"),
    ]
    for run in args.runs.split(","):
        forecasts = pd.concat(
            [pd.read_parquet(path) for path in sorted((args.forecasts / run).glob("*.parquet"))], ignore_index=True
        )
        forecasts["date"] = pd.to_datetime(forecasts["date"])
        forecasts = forecasts.dropna(subset=["tfm_pred"])
        keep = forecasts["permno"].unique()
        data = forecasts
        for name, wide in measures.items():
            data = data.merge(stack(wide[keep], name), on=["date", "permno"], how="left")

        print(f"\n{run}: {len(data):,} forecasts on {data['date'].nunique():,} dates")
        print(
            f"  forecast: mean {data['tfm_pred'].mean():+.5f}, within-date std "
            f"{data.groupby('date')['tfm_pred'].std().mean():.5f}, share positive {(data['tfm_pred'] > 0).mean():.1%}"
        )
        for left, right in pairs:
            ic = cross_sectional_ic(data, left, right)
            print(f"  {left:12s} vs {right:14s} {ic.mean():+.3f}  (t={hac_t_stat(ic, 10):+.1f})")
        # A wrong alignment or a broken quantile order would show up here.
        print(f"  dates where width is not positive: {(data['tfm_width'] <= 0).mean():.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
