#!/usr/bin/env python
"""Run the baseline models through the walk-forward on the development period."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qef.dataset import build_dataset  # noqa: E402
from qef.fundamentals import add_fundamental_columns  # noqa: E402
from qef.models import FactorScore, LightGBMAverage, LightGBMScore, RidgeScore  # noqa: E402
from qef.splits import plan_folds  # noqa: E402
from qef.timesfm_zero_shot import add_timesfm_features  # noqa: E402
from qef.walkforward import evaluate, load_market_data, log_experiment, run_walk_forward  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True, help="Parquet file in the canonical panel schema")
    parser.add_argument("--spec", type=Path, default=ROOT / "config" / "spec.yaml")
    parser.add_argument("--models", default="momentum,reversal,ridge,lightgbm")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0, help="LightGBM seed; other than 0, it is added to the run's name")
    parser.add_argument("--raw", type=Path, default=None, help="Folder holding sp500.parquet and ff_daily.parquet")
    parser.add_argument("--timesfm", type=Path, default=None, help="Folder of TimesFM shards; enables the timesfm models")
    parser.add_argument("--fundamentals", type=Path, default=None, help="Raw data folder; enables the accounting and analyst models")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "baselines", help="Summaries (small)")
    parser.add_argument("--series-out", type=Path, default=ROOT / "data" / "predictions", help="Scores and daily series")
    parser.add_argument("--registry", type=Path, default=ROOT / "experiments" / "registry.jsonl")
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Run on the final test years, once, with the configuration of config/frozen.yaml. "
        "Results go to reports/final, data/predictions/final and experiments/final_test.jsonl.",
    )
    args = parser.parse_args()
    if args.final_test:
        args.out = ROOT / "reports" / "final"
        args.series_out = ROOT / "data" / "predictions" / "final"
        args.registry = ROOT / "experiments" / "final_test.jsonl"

    spec = yaml.safe_load(args.spec.read_text())
    references, factors = load_market_data(args.raw)
    panel = pd.read_parquet(args.panel)
    dataset = build_dataset(panel, spec)
    fundamental_columns, analyst_columns = [], []
    if args.fundamentals is not None:
        frame, fundamental_columns, analyst_columns, report = add_fundamental_columns(dataset.frame, panel, args.fundamentals)
        dataset = replace(dataset, frame=frame)
        args.out.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.out / "fundamentals_coverage.tsv", sep="\t", float_format="%.4f")
        print(report["coverage"].round(2).to_string(), flush=True)
        print(report["median"].round(3).to_string(), flush=True)
    del panel
    print(
        f"Dataset: {len(dataset.frame):,} stock-dates, {dataset.frame['date'].nunique():,} dates, "
        f"{dataset.frame['permno'].nunique():,} stocks, {int(dataset.frame['tradable'].sum()):,} tradable",
        flush=True,
    )

    folds = plan_folds(dataset.ret.index, spec, final_test=args.final_test)

    tree_features = dataset.stock_features + dataset.market_features
    selected = lambda: LightGBMScore(num_threads=args.threads, seed=args.seed)  # noqa: E731
    averaged = lambda: LightGBMAverage(num_threads=args.threads, seed=args.seed)  # noqa: E731
    catalog = {
        "momentum": (lambda: FactorScore("f_mom_252_21"), dataset.stock_features),
        "reversal": (lambda: FactorScore("f_ret_5", sign=-1.0), dataset.stock_features),
        "ridge": (RidgeScore, dataset.stock_features),
        "lightgbm": (selected, tree_features),
        "lightgbm_avg": (averaged, tree_features),
    }

    if args.fundamentals is not None:
        catalog["sue"] = (lambda: FactorScore("f_ana_sue"), analyst_columns)
        catalog["revision"] = (lambda: FactorScore("f_ana_revision_3m"), analyst_columns)
        for name, extra in (
            ("lightgbm_fund", fundamental_columns),
            ("lightgbm_ana", analyst_columns),
            ("lightgbm_fund_ana", fundamental_columns + analyst_columns),
        ):
            catalog[name] = (selected, tree_features + extra)
            catalog[f"{name}_avg"] = (averaged, tree_features + extra)

    timesfm_models = set()
    if args.timesfm is not None:
        shards = sorted(args.timesfm.glob("*.parquet"))
        forecasts = pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True)
        frame, forecast_columns = add_timesfm_features(dataset.frame, forecasts)
        dataset = replace(dataset, frame=frame)
        catalog["timesfm"] = (lambda: FactorScore("f_tfm_pred"), forecast_columns)
        catalog["lightgbm_tfm"] = (selected, tree_features + forecast_columns)
        catalog["lightgbm_tfm_avg"] = (averaged, tree_features + forecast_columns)
        timesfm_models = {"timesfm", "lightgbm_tfm", "lightgbm_tfm_avg"}
        if args.fundamentals is not None:
            catalog["lightgbm_all_avg"] = (averaged, tree_features + fundamental_columns + analyst_columns + forecast_columns)
            timesfm_models.add("lightgbm_all_avg")

    args.out.mkdir(parents=True, exist_ok=True)
    args.series_out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.timesfm.name}" if args.timesfm is not None else ""
    for model in args.models.split(","):
        factory, features = catalog[model]
        name = f"{model}{tag}" if model in timesfm_models else model
        if args.seed:
            name = f"{name}_seed{args.seed}"
        print(f"\n{name}", flush=True)
        predictions, selections = run_walk_forward(dataset, factory, folds, features)
        results, daily = evaluate(predictions, dataset, spec, references=references, factors=factors)
        predictions.to_parquet(args.series_out / f"{name}_predictions.parquet", index=False)
        daily.to_parquet(args.series_out / f"{name}_daily.parquet")
        (args.out / f"{name}.json").write_text(json.dumps(results, indent=2))
        log_experiment(
            args.registry,
            model=name,
            spec=spec,
            data_version=str(args.panel),
            results=results,
            selections=selections,
        )
        signal = results["signal"]
        central = results["portfolio"][spec["evaluation"]["primary_cost"]]
        print(
            f"  Rank IC {signal['rank_ic']:+.4f} (t={signal['rank_ic_t']:.2f}) | "
            f"net active {central['ann_active']:+.2%} | IR {central['information_ratio']:.2f} | "
            f"turnover {central['annual_turnover']:.0%}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
