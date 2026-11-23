#!/usr/bin/env python
"""Export the small aggregate tables drawn by the paper's figures (pgfplots reads them).

Reads the saved daily series and summaries of the development period and of the final test.
Writes only aggregates: month-end cumulative returns, yearly Rank IC and cost sensitivity.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRED, REP = ROOT / "data" / "predictions", ROOT / "reports"
OUT = Path(__file__).resolve().parent / "data"
MODELS = {  # column name: (development daily, final daily, development summary, final summary)
    "candidate": ("study/avg3_signals_s1_x40", "final/study/avg3_signals_s1_x40", "study_avg/avg3_signals_s1_x40", "final/study/avg3_signals_s1_x40"),
    "reversal": ("reversal", "final/reversal", "baselines/reversal", "final/reversal"),
    "lightgbm": ("lightgbm", "final/lightgbm", "baselines/lightgbm", "final/lightgbm"),
    "tfmft": ("tfm_finetuned_10k", "final/tfm_finetuned_10k", "baselines/tfm_finetuned_10k", "final/tfm_finetuned_10k"),
    "tfmzs": ("timesfm_returns_trade", "final/timesfm_returns_trade", "baselines/timesfm_returns_trade", "final/timesfm_returns_trade"),
    "momentum": ("momentum", "final/momentum", "baselines/momentum", "final/momentum"),
}


def active(path: str) -> pd.Series:
    frame = pd.read_parquet(PRED / f"{path}_daily.parquet")
    return (frame["net_central"] - frame["benchmark"]).dropna()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cumulative, ic = {}, {}
    for name, (dev, final, dev_summary, final_summary) in MODELS.items():
        series = pd.concat([active(dev), active(final)]).sort_index()
        assert series.index.is_unique
        cumulative[name] = series.cumsum().resample("ME").last() * 100
        years = {**json.loads((REP / f"{dev_summary}.json").read_text())["signal"]["rank_ic_by_year"],
                 **json.loads((REP / f"{final_summary}.json").read_text())["signal"]["rank_ic_by_year"]}
        ic[name] = pd.Series({int(y): v for y, v in years.items()}).sort_index()
    table = pd.DataFrame(cumulative)
    table.insert(0, "t", table.index.year + (table.index.month - 0.5) / 12)
    table.round(4).to_csv(OUT / "cumulative_active.tsv", sep="\t", index=False)
    pd.DataFrame(ic).rename_axis("year").round(5).to_csv(OUT / "rank_ic_by_year.tsv", sep="\t")

    rows = []
    for cost, scenario in ((2, "optimistic"), (5, "central"), (10, "prudent")):
        row = {"cost": cost}
        for name, (_, _, dev_summary, _) in MODELS.items():
            book = json.loads((REP / f"{dev_summary}.json").read_text())["portfolio"][scenario]
            row[name] = book["ann_active"] * 100
            row[f"{name}_turnover"] = book["annual_turnover"] * 100
        rows.append(row)
    pd.DataFrame(rows).round(3).to_csv(OUT / "cost_sensitivity.tsv", sep="\t", index=False)
    print(table.tail(3).to_string(), "\n", pd.DataFrame(ic).round(4).to_string(), "\n", pd.DataFrame(rows).round(2).to_string())


if __name__ == "__main__":
    main()
