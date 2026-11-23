#!/usr/bin/env python
"""Report the final test of the models declared in config/frozen.yaml, next to their development results.

Only the comparisons declared in advance are computed. Nothing is fitted or selected here.
"""

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


def find(folder: Path, name: str, suffix: str) -> Path:
    for path in (folder / f"{name}{suffix}", folder / "study" / f"{name}{suffix}"):
        if path.exists():
            return path
    raise SystemExit(f"Missing final result {name}{suffix} in {folder}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, default=ROOT / "config" / "frozen.yaml")
    parser.add_argument("--spec", type=Path, default=ROOT / "config" / "spec.yaml")
    parser.add_argument("--summaries", type=Path, default=ROOT / "reports" / "final")
    parser.add_argument("--series", type=Path, default=ROOT / "data" / "predictions" / "final")
    args = parser.parse_args()

    frozen = yaml.safe_load(args.frozen.read_text())
    settings = yaml.safe_load(args.spec.read_text())["evaluation"]
    scenario = settings["primary_cost"]
    names = list(frozen["models"])
    summaries = {name: json.loads(find(args.summaries, name, ".json").read_text()) for name in names}
    daily = {name: pd.read_parquet(find(args.series, name, "_daily.parquet")) for name in names}
    development = {name: json.loads((ROOT / path).read_text()) for name, path in frozen["models"].items()}

    rows = {}
    for name in names:
        summary, book = summaries[name], summaries[name]["portfolio"]
        active = (daily[name][f"net_{scenario}"] - daily[name]["benchmark"]).dropna()
        dev_book = development[name]["portfolio"][scenario]
        rows[name] = {
            "dates": summary["signal"]["n_dates"],
            "rank_ic": summary["signal"]["rank_ic"],
            "ic_t": summary["signal"]["rank_ic_t"],
            "ic_low": summary["signal"]["rank_ic_ci"][0],
            "ic_high": summary["signal"]["rank_ic_ci"][1],
            "return": book[scenario]["ann_return"],
            "active": book[scenario]["ann_active"],
            "IR": book[scenario]["information_ratio"],
            "active_t": book[scenario]["active_t"],
            "active_10bp": book["prudent"]["ann_active"],
            "IR_10bp": book["prudent"]["information_ratio"],
            "IR_measured": book["measured"]["information_ratio"],
            "alpha": book[scenario].get("alpha_annual", np.nan),
            "alpha_t": book[scenario].get("alpha_t", np.nan),
            "beta": book[scenario].get("beta_mktrf", np.nan),
            "turnover": book[scenario]["annual_turnover"],
            "max_dd": book[scenario]["max_drawdown"],
            "P(IR>0)": deflated_sharpe(active, 1, 0.0),
            "dev_rank_ic": development[name]["signal"]["rank_ic"],
            "dev_active": dev_book["ann_active"],
            "dev_IR": dev_book["information_ratio"],
        }
    table = pd.DataFrame(rows).T
    reference = summaries[names[0]]
    print(f"Final test, cost {scenario}; primary model: {frozen['primary']}")
    print(f"Equal-weight benchmark: {reference['benchmark']['ann_return']:+.2%}/year; "
          + ", ".join(f"{k}: {v['ann_return']:+.2%}/year" for k, v in reference["references"].items()))
    print(table.round(4).to_string())

    gaps = []
    for a, b in frozen["gaps"]:
        for label, column, scale in (("rank_ic", "rank_ic", 1), ("net_return", f"net_{scenario}", 252)):
            gap = (daily[a][column] - daily[b][column]).dropna()
            low, high = block_bootstrap_ci(gap, block=settings["bootstrap_block"], reps=settings["bootstrap_reps"], seed=settings["bootstrap_seed"])
            gaps.append({"a": a, "b": b, "measure": label, "gap": gap.mean() * scale, "t": hac_t_stat(gap, settings["hac_lags"]), "low": low * scale, "high": high * scale})
    gaps = pd.DataFrame(gaps)
    print("\nPaired gaps declared in advance (net return annualized)")
    print(gaps.round(4).to_string(index=False))

    by_year = pd.DataFrame(
        {name: (daily[name][f"net_{scenario}"] - daily[name]["benchmark"]).dropna().groupby(lambda d: d.year).mean() * 252 for name in names}
    )
    ic_by_year = pd.DataFrame({name: pd.Series(summaries[name]["signal"]["rank_ic_by_year"]) for name in names})
    print("\nNet active return by year"); print(by_year.round(3).to_string())
    print("\nRank IC by year"); print(ic_by_year.astype(float).round(4).to_string())

    table.to_csv(args.summaries / "summary.tsv", sep="\t", float_format="%.5f")
    gaps.to_csv(args.summaries / "gaps.tsv", sep="\t", index=False, float_format="%.5f")
    by_year.to_csv(args.summaries / "active_by_year.tsv", sep="\t", float_format="%.5f")
    ic_by_year.to_csv(args.summaries / "rank_ic_by_year.tsv", sep="\t")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
