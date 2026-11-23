#!/usr/bin/env python
"""Evaluate the ensembles and turnover settings of config/study_portfolio.yaml.

Reuses the out-of-sample scores of the development period, so nothing is retrained
and the final test period is not read. Each run is added to the experiment registry.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qef.combine import resolve_ensemble, smooth_scores  # noqa: E402
from qef.dataset import build_dataset  # noqa: E402
from qef.metrics import block_bootstrap_ci, deflated_sharpe  # noqa: E402
from qef.walkforward import evaluate, load_market_data, log_experiment, trial_sharpe_variance  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=ROOT / "data" / "clean" / "panel.parquet")
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--spec", type=Path, default=ROOT / "config" / "spec.yaml")
    parser.add_argument("--study", type=Path, default=ROOT / "config" / "study_portfolio.yaml")
    parser.add_argument("--scores", type=Path, default=ROOT / "data" / "predictions")
    parser.add_argument("--reference", default="lightgbm", help="Existing model the runs are compared with")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "study")
    parser.add_argument("--series-out", type=Path, default=ROOT / "data" / "predictions" / "study")
    parser.add_argument("--registry", type=Path, default=ROOT / "experiments" / "registry.jsonl")
    args = parser.parse_args()

    spec = yaml.safe_load(args.spec.read_text())
    study = yaml.safe_load(args.study.read_text())
    settings = spec["evaluation"]
    scenario = settings["primary_cost"]
    references, factors = load_market_data(args.raw)

    dataset = build_dataset(pd.read_parquet(args.panel), spec)
    tradable = dataset.frame.loc[dataset.frame["tradable"], ["date", "permno"]]

    ensembles = study["ensembles"]
    names = {source for name, sources in ensembles.items() for source in sources if source == name or source not in ensembles}
    scores = {name: pd.read_parquet(args.scores / f"{name}_predictions.parquet") for name in names}
    # Common dates, so that every run is measured on the same period as the existing models.
    dates = sorted(set(scores[args.reference]["date"].unique()))
    tradable = tradable[tradable["date"].isin(dates)]

    args.out.mkdir(parents=True, exist_ok=True)
    args.series_out.mkdir(parents=True, exist_ok=True)
    reference = pd.read_parquet(args.scores / f"{args.reference}_daily.parquet")

    rows, dailies = [], {}
    for run in study["runs"]:
        name = f"{run['model']}_s{run['smooth']}_x{run['exit_rank']}"
        if "n_positions" in run:
            name += f"_n{run['n_positions']}"
        predictions = resolve_ensemble(run["model"], ensembles, scores, tradable)
        if run["smooth"] > 1:
            predictions = smooth_scores(predictions, run["smooth"])

        run_spec = copy.deepcopy(spec)
        run_spec["portfolio"].update({key: run[key] for key in ("exit_rank", "n_positions", "max_weight") if key in run})
        results, daily = evaluate(predictions, dataset, run_spec, references=references, factors=factors)
        (args.out / f"{name}.json").write_text(json.dumps(results, indent=2))
        daily.to_parquet(args.series_out / f"{name}_daily.parquet")
        log_experiment(args.registry, model=name, spec=run_spec, data_version=str(args.panel), results=results, selections=[run])
        dailies[name] = daily

        book, measured = results["portfolio"][scenario], results["portfolio"].get("measured", {})
        rows.append(
            {
                "run": name,
                "rank_ic": results["signal"]["rank_ic"],
                "ic_t": results["signal"]["rank_ic_t"],
                "active": book["ann_active"],
                "IR": book["information_ratio"],
                "active_t": book["active_t"],
                "IR_measured": measured.get("information_ratio", np.nan),
                "IR_prudent": results["portfolio"].get("prudent", {}).get("information_ratio", np.nan),
                "alpha": book.get("alpha_annual", np.nan),
                "alpha_t": book.get("alpha_t", np.nan),
                "turnover": book["annual_turnover"],
                "cost": book["annual_cost"],
            }
        )
        print(f"{name}: IC {rows[-1]['rank_ic']:+.4f}, IR {rows[-1]['IR']:+.2f}, turnover {rows[-1]['turnover']:.1f}", flush=True)

    # Trials and the spread of Sharpe ratios include every run of the registry.
    records = [json.loads(line) for line in args.registry.read_text().splitlines()]
    trial_variance = trial_sharpe_variance([args.scores, args.series_out], scenario)

    for row in rows:
        daily = dailies[row["run"]]
        active = (daily[f"net_{scenario}"] - daily["benchmark"]).dropna()
        row["DSR"] = deflated_sharpe(active, len(records), trial_variance)
        for label, column in (("ic_gap", "rank_ic"), ("net_gap", f"net_{scenario}")):
            gap = (daily[column] - reference[column]).dropna()
            low, high = block_bootstrap_ci(
                gap, block=settings["bootstrap_block"], reps=settings["bootstrap_reps"], seed=settings["bootstrap_seed"]
            )
            row[label] = gap.mean() * (252 if label == "net_gap" else 1)
            row[f"{label}_low"], row[f"{label}_high"] = low * (252 if label == "net_gap" else 1), high * (252 if label == "net_gap" else 1)

    table = pd.DataFrame(rows).set_index("run")
    table.to_csv(args.out / "summary.tsv", sep="\t", float_format="%.5f")
    print(f"\nTrials in the registry: {len(records)}; variance of daily Sharpe across runs: {trial_variance:.6f}")
    print(f"Gaps are against {args.reference}; the net gap is annualized.\n")
    print(table.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
