#!/usr/bin/env python
"""Compare models from the registry: results, Deflated Sharpe and paired bootstrap tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qef.metrics import block_bootstrap_ci, deflated_sharpe, hac_t_stat  # noqa: E402
from qef.walkforward import trial_sharpe_variance  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="momentum,reversal,ridge,lightgbm")
    parser.add_argument("--reference", default="lightgbm", help="Model the others are compared with")
    parser.add_argument("--spec", type=Path, default=ROOT / "config" / "spec.yaml")
    parser.add_argument("--summaries", type=Path, default=ROOT / "reports" / "baselines")
    parser.add_argument("--series", type=Path, default=ROOT / "data" / "predictions")
    parser.add_argument("--registry", type=Path, default=ROOT / "experiments" / "registry.jsonl")
    args = parser.parse_args()

    settings = yaml.safe_load(args.spec.read_text())["evaluation"]
    scenario = settings["primary_cost"]
    names = args.models.split(",")
    summaries = {name: json.loads((args.summaries / f"{name}.json").read_text()) for name in names}
    daily = {name: pd.read_parquet(args.series / f"{name}_daily.parquet") for name in names}

    # Every registered run counts as a trial, including those no longer compared.
    n_trials = sum(1 for _ in args.registry.open()) if args.registry.exists() else len(names)
    trial_variance = trial_sharpe_variance([args.series, args.series / "study"], scenario)

    rows = []
    for name in names:
        signal, book = summaries[name]["signal"], summaries[name]["portfolio"][scenario]
        measured = summaries[name]["portfolio"].get("measured", {})
        active = daily[name][f"net_{scenario}"] - daily[name]["benchmark"]
        rows.append(
            {
                "model": name,
                "rank_ic": signal["rank_ic"],
                "ic_t": signal["rank_ic_t"],
                "net_active": book["ann_active"],
                "IR": book["information_ratio"],
                "active_t": book["active_t"],
                "IR_measured": measured.get("information_ratio", np.nan),
                "alpha": book.get("alpha_annual", np.nan),
                "alpha_t": book.get("alpha_t", np.nan),
                "turnover": book["annual_turnover"],
                "max_dd": book["max_drawdown"],
                "DSR": deflated_sharpe(active, n_trials, trial_variance),
            }
        )
    table = pd.DataFrame(rows).set_index("model")
    print(f"Cost scenario: {scenario}; trials in the registry: {n_trials}\n")
    print(table.round(4).to_string())

    reference = daily[args.reference]
    print(f"\nPaired gaps with {args.reference} (block bootstrap):")
    for name in names:
        if name == args.reference:
            continue
        print(f"  {name}:")
        for label, column in (("Rank IC", "rank_ic"), ("net return", f"net_{scenario}")):
            gap = (reference[column] - daily[name][column]).dropna()
            low, high = block_bootstrap_ci(
                gap,
                block=settings["bootstrap_block"],
                reps=settings["bootstrap_reps"],
                seed=settings["bootstrap_seed"],
            )
            print(
                f"    {label:14s} gap {gap.mean():+.6f}  t={hac_t_stat(gap, settings['hac_lags']):+.2f}  "
                f"95% CI [{low:+.6f}, {high:+.6f}]"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
