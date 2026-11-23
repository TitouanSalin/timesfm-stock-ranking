#!/usr/bin/env python
"""Fine-tune TimesFM 3 before each block of the development period and score its tradable stocks.

The settings are those of the earlier study on data before 2005, with no hyperparameter
selection. Dates of the final test are never loaded. Weights are never saved, as their
license forbids distributing them. Finished runs are skipped, so a job can be resubmitted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qef.dataset import universe_rows  # noqa: E402
from qef.finetune import CONTEXT, check_weights, eligible_rows, score_rows, train_run  # noqa: E402
from qef.panel import to_wide  # noqa: E402
from qef.splits import group_folds, plan_folds  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=ROOT / "data" / "clean" / "panel.parquet")
    parser.add_argument("--spec", type=Path, default=ROOT / "config" / "spec.yaml")
    parser.add_argument("--every", type=int, default=3, help="Years between retrainings")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--scope", default="last_4")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "predictions" / "timesfm" / "finetuned")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "finetune_walkforward.json")
    parser.add_argument("--final-test", action="store_true", help="Score the final test years, once, with the frozen configuration")
    args = parser.parse_args()
    if args.final_test and not ("final" in args.out.name and "final" in str(args.report)):
        raise SystemExit("The final test must be written to its own folders: pass --out and --report paths naming 'final'")

    from timesfm3 import ModelConfig, TimesFM3Evaluator

    spec = yaml.safe_load(args.spec.read_text())
    plan, target = spec["walk_forward"], spec["target"]
    skip, horizon = target["skip"], target["horizon"]
    n_steps = skip + horizon

    # Outside the final test, the dates of the final test are not even loaded.
    panel = pd.read_parquet(
        args.panel,
        columns=["permno", "date", "ret", "prc", "dollar_vol", "mcap"],
        filters=None if args.final_test else [("date", "<", pd.Timestamp(plan["holdout_start"]))],
    )
    wide = to_wide(panel)
    del panel
    assert args.final_test or wide.ret.index.max() < pd.Timestamp(plan["holdout_start"])
    rows, tradable, members = universe_rows(wide, spec)
    ret = wide.ret[members]
    del wide
    calendar = ret.index
    permnos = members.to_numpy()
    rows, tradable = rows.to_numpy(), tradable.to_numpy()
    series = np.nan_to_num(ret.to_numpy(dtype=np.float32), nan=0.0)
    history = ret.notna().cumsum().to_numpy()

    blocks = group_folds(plan_folds(calendar, spec, final_test=args.final_test), args.every)

    settings = {key: getattr(args, key) for key in ("every", "seeds", "steps", "scope", "lr", "batch_size")}
    report = json.loads(args.report.read_text()) if args.report.exists() else {"settings": settings, "blocks": {}, "runs": {}}
    if report["settings"] != settings:
        raise SystemExit(f"{args.report} holds other settings: {report['settings']}")
    args.out.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    forecaster = TimesFM3Evaluator(
        ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", per_core_batch_size=64, device="cuda")
    )
    model = forecaster.model
    check_weights(model)
    device = next(model.parameters()).device
    pretrained = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    failures = []
    for block in blocks:
        first = int(calendar.get_loc(block.test_dates[0]))
        last = int(calendar.get_loc(block.test_dates[-1])) + 1
        train_rows = eligible_rows(rows, history, CONTEXT, 0, len(block.refit_dates))
        test_rows = eligible_rows(tradable, history, CONTEXT, first, last)
        assert train_rows[:, 0].max() + n_steps < first
        report["blocks"][str(block.test_year)] = {
            "train_windows": len(train_rows),
            "last_train_date": str(calendar[train_rows[:, 0].max()].date()),
            "test_windows": len(test_rows),
            "tradable_stock_dates": int(tradable[first:last].sum()),
            "first_test_date": str(calendar[first].date()),
            "last_test_date": str(calendar[last - 1].date()),
        }
        print(f"Block {block.test_year}: {report['blocks'][str(block.test_year)]}", flush=True)

        for seed in [int(value) for value in args.seeds.split(",")]:
            path = args.out / f"seed{seed}_block{block.test_year}.parquet"
            if path.exists():
                print(f"  seed {seed}: already done", flush=True)
                continue
            model.load_state_dict(pretrained)
            torch.manual_seed(seed)
            training = train_run(
                model, series, train_rows, n_steps, skip, horizon, device,
                loss_name="correlation", scope=args.scope, steps=args.steps, lr=args.lr,
                batch_size=args.batch_size, rng=np.random.default_rng([seed, block.test_year]),
            )
            run = {"training": training}
            if "failed" in training:
                failures.append((seed, block.test_year))
            else:
                started = time.time()
                scores = score_rows(model, series, test_rows, skip=skip, horizon=horizon, device=device)
                invalid = ~np.isfinite(scores)
                scores[invalid] = np.nan
                run.update(scoring_seconds=round(time.time() - started, 1), invalid_scores=int(invalid.sum()))
                frame = pd.DataFrame({"date": calendar[test_rows[:, 0]], "permno": permnos[test_rows[:, 1]], "pred": scores})
                temporary = path.with_suffix(".tmp")
                frame.to_parquet(temporary, index=False, compression="zstd")
                temporary.rename(path)
            report["runs"][f"seed{seed}_block{block.test_year}"] = run
            args.report.write_text(json.dumps(report, indent=2))
            print(f"  seed {seed}: {run}", flush=True)

    if failures:
        print(f"Failed runs (seed, block): {failures}", flush=True)
        return 1
    print("Done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
