#!/usr/bin/env python
"""Evaluate the fine-tuned TimesFM scores: each seed, then the equal-weight average of their ranks.

Every run counts as a trial. An incomplete set of runs is refused, because the results
would not cover the same period as the other models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qef.combine import combine_scores  # noqa: E402
from qef.dataset import build_dataset  # noqa: E402
from qef.splits import group_folds, plan_folds  # noqa: E402
from qef.walkforward import evaluate, load_market_data, log_experiment  # noqa: E402


def merge_settings(reports: list[dict]) -> dict:
    """Settings of runs split across several reports, which may only differ by their seeds."""
    others = [{key: value for key, value in report.items() if key != "seeds"} for report in reports]
    if any(other != others[0] for other in others):
        raise SystemExit(f"These runs used different settings: {others}")
    return {**others[0], "seeds": ",".join(report["seeds"] for report in reports)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=ROOT / "data" / "clean" / "panel.parquet")
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--spec", type=Path, default=ROOT / "config" / "spec.yaml")
    parser.add_argument("--scores", type=Path, default=ROOT / "data" / "predictions" / "timesfm" / "finetuned")
    parser.add_argument("--report", type=Path, nargs="+", default=[ROOT / "reports" / "finetune_walkforward.json"])
    parser.add_argument("--seed-prefix", default="tfm_ft", help="Start of the name of each seed's run")
    parser.add_argument("--average-name", default="tfm_finetuned", help="Name of the run averaging the seeds")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "baselines")
    parser.add_argument("--series-out", type=Path, default=ROOT / "data" / "predictions")
    parser.add_argument("--registry", type=Path, default=ROOT / "experiments" / "registry.jsonl")
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Evaluate the final test years, once. Results go to reports/final, data/predictions/final "
        "and experiments/final_test.jsonl.",
    )
    args = parser.parse_args()
    if args.final_test:
        args.out = ROOT / "reports" / "final"
        args.series_out = ROOT / "data" / "predictions" / "final"
        args.registry = ROOT / "experiments" / "final_test.jsonl"
        args.out.mkdir(parents=True, exist_ok=True)
        args.series_out.mkdir(parents=True, exist_ok=True)

    spec = yaml.safe_load(args.spec.read_text())
    settings = merge_settings([json.loads(path.read_text())["settings"] for path in args.report])
    references, factors = load_market_data(args.raw)
    dataset = build_dataset(pd.read_parquet(args.panel), spec)

    blocks = group_folds(plan_folds(dataset.ret.index, spec, final_test=args.final_test), settings["every"])
    seeds = [int(value) for value in settings["seeds"].split(",")]
    paths = {(seed, block.test_year): args.scores / f"seed{seed}_block{block.test_year}.parquet" for seed in seeds for block in blocks}
    missing = [path.name for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing runs: {missing}")

    by_seed = {
        f"seed{seed}": pd.concat([pd.read_parquet(paths[seed, block.test_year]) for block in blocks], ignore_index=True)
        for seed in seeds
    }
    runs = {f"{args.seed_prefix}_{name}": scores for name, scores in by_seed.items()}
    tradable = dataset.frame.loc[dataset.frame["tradable"], ["date", "permno"]]
    tradable = tradable[tradable["date"].isin(by_seed[f"seed{seeds[0]}"]["date"].unique())]
    runs[args.average_name] = combine_scores(by_seed, tradable)
    print(f"Scored {len(runs[args.average_name]):,} of {len(tradable):,} tradable stock-dates", flush=True)

    for name, predictions in runs.items():
        results, daily = evaluate(predictions, dataset, spec, references=references, factors=factors)
        predictions.to_parquet(args.series_out / f"{name}_predictions.parquet", index=False)
        daily.to_parquet(args.series_out / f"{name}_daily.parquet")
        (args.out / f"{name}.json").write_text(json.dumps(results, indent=2))
        log_experiment(args.registry, model=name, spec=spec, data_version=str(args.panel), results=results, selections=[settings])
        signal, central = results["signal"], results["portfolio"][spec["evaluation"]["primary_cost"]]
        print(
            f"{name}: Rank IC {signal['rank_ic']:+.4f} (t={signal['rank_ic_t']:.2f}) | net active {central['ann_active']:+.2%} | "
            f"IR {central['information_ratio']:.2f} | turnover {central['annual_turnover']:.0%}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
