#!/usr/bin/env python
"""Test whether TimesFM 3 can be fine-tuned, and at what cost.

Only data before 2005 is read, which every walk-forward fold already trains on, so
neither the development test years nor the final test are touched. The short runs
show that training works; two validation years cannot establish a performance gain.
No weights are saved.
"""

from __future__ import annotations

import argparse
import json
import math
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
from qef.finetune import (  # noqa: E402
    CONTEXT,
    batch_loss,
    decode_with_grad,
    eligible_rows,
    five_day_score,
    make_batch,
    pinball_loss,
    set_trainable,
    to_device,
    train_run,
)
from qef.metrics import block_bootstrap_ci, cross_sectional_ic, hac_t_stat  # noqa: E402
from qef.panel import to_wide  # noqa: E402

LAST_DATE_READ = "2005-01-01"
VAL_START = "2003-01-01"


def check_parity(forecaster, series, rows, n_steps, device) -> dict:
    """The training path must give the same forecasts as the package's own prediction path."""
    contexts, _ = make_batch(series, rows[:8], CONTEXT, n_steps)
    outputs = list(
        forecaster.predict_batch(
            [c.astype(np.float32) for c in contexts],
            horizon=n_steps,
            return_quantiles=True,
            use_symmetric_averaging=False,
            make_positive=False,
            sort_quantiles=False,
            univariate=True,
        )
    )
    official = np.stack([o.quantiles for o in outputs])
    with torch.no_grad():
        ours = decode_with_grad(forecaster.model, to_device(contexts, device), n_steps).cpu().numpy()
    return {"max_abs_gap": float(np.abs(official - ours).max()), "typical_size": float(np.abs(official).mean())}


def check_gradients(model, series, rows, n_steps, device) -> dict:
    """Backpropagate through the whole model and report which parameters receive a gradient."""
    set_trainable(model, "full")
    model.zero_grad(set_to_none=True)
    loss = batch_loss(model, series, rows[:8], n_steps, device)
    loss.backward()
    report: dict[str, dict] = {}
    finite = True
    for name, parameter in model.named_parameters():
        group = name.split(".")[0]
        entry = report.setdefault(group, {"tensors": 0, "no_grad": 0, "zero_grad": 0, "grad_norm_sq": 0.0})
        entry["tensors"] += 1
        if parameter.grad is None:
            entry["no_grad"] += 1
            continue
        norm = float(parameter.grad.norm())
        finite = finite and math.isfinite(norm)
        entry["zero_grad"] += int(norm == 0)
        entry["grad_norm_sq"] += norm**2
    model.zero_grad(set_to_none=True)
    return {"loss": float(loss.detach()), "all_finite": finite, "groups": report}


def measure_cost(model, series, rows, n_steps, device, scopes, batch_sizes, rng) -> list[dict]:
    """Peak GPU memory and time per step for each trainable scope and batch size."""
    results = []
    for scope in scopes:
        n_parameters = set_trainable(model, scope)
        for batch_size in batch_sizes:
            entry = {"scope": scope, "trainable_millions": round(n_parameters / 1e6, 1), "batch": batch_size}
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0)
            try:
                torch.cuda.reset_peak_memory_stats(device)
                times = []
                for step in range(4):
                    sample = rows[rng.integers(0, len(rows), batch_size)]
                    torch.cuda.synchronize(device)
                    started = time.time()
                    optimizer.zero_grad(set_to_none=True)
                    batch_loss(model, series, sample, n_steps, device).backward()
                    optimizer.step()
                    torch.cuda.synchronize(device)
                    if step > 0:
                        times.append(time.time() - started)
                entry["peak_gb"] = round(torch.cuda.max_memory_allocated(device) / 1024**3, 2)
                entry["seconds_per_step"] = round(float(np.mean(times)), 3)
            except torch.cuda.OutOfMemoryError:
                entry["out_of_memory"] = True
            del optimizer
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"  {entry}", flush=True)
            results.append(entry)
            if entry.get("out_of_memory"):
                break
    return results


def overfit_one_batch(model, series, rows, n_steps, device, scope, steps, lr) -> dict:
    """A model able to learn must nearly memorize a small fixed batch."""
    set_trainable(model, scope)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(model, series, rows[:32], n_steps, device)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return {"scope": scope, "steps": steps, "lr": lr, "first_loss": losses[0], "last_loss": losses[-1], "ratio": losses[-1] / losses[0]}


def evaluate(model, series, rows, calendar, n_steps, skip, horizon, device, batch_size=256, return_frame=False):
    """Validation loss and Rank IC of the five-day score, on tradable stocks."""
    scores, pinball, n_batches = [], 0.0, 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            sample = rows[start : start + batch_size]
            contexts, future = make_batch(series, sample, CONTEXT, n_steps)
            contexts, future = to_device(contexts, device), to_device(future, device)
            quantiles = decode_with_grad(model, contexts, n_steps)
            pinball += float(pinball_loss(quantiles, future, contexts.std(dim=1).clamp_min(1e-4)))
            n_batches += 1
            scores.append(five_day_score(quantiles, skip, horizon).cpu().numpy())
    _, future = make_batch(series, rows, CONTEXT, n_steps)
    frame = pd.DataFrame(
        {
            "date": calendar[rows[:, 0]],
            "pred": np.concatenate(scores),
            "y": np.expm1(np.log1p(np.maximum(future[:, skip : skip + horizon], -0.9999)).sum(axis=1)),
        }
    )
    ic = cross_sectional_ic(frame, "pred", "y")
    summary = {"pinball": pinball / n_batches, "rank_ic": float(ic.mean()), "rank_ic_t": hac_t_stat(ic, 10), "n_windows": len(frame)}
    return (summary, frame) if return_frame else summary


def seed_study(model, restore, series, train_rows, val_rows, calendar, members, n_steps, skip, horizon, device, args, settings) -> dict:
    """Repeat the correlation-loss run over several seeds and compare it with simple signals."""
    zero_shot, base = evaluate(model, series, val_rows, calendar, n_steps, skip, horizon, device, return_frame=True)
    contexts, _ = make_batch(series, val_rows, CONTEXT, n_steps)
    log_context = np.log1p(np.maximum(contexts, -0.9999))
    data = base.rename(columns={"pred": "zero_shot"}).assign(
        permno=members[val_rows[:, 1]],
        reversal=-log_context[:, -5:].sum(axis=1),
        momentum=log_context[:, -252:-21].sum(axis=1),
    )
    runs = {}
    seed_columns = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        restore()
        torch.manual_seed(seed)
        training = train_run(
            model, series, train_rows, n_steps, skip, horizon, device,
            loss_name="correlation", scope=args.scope, steps=args.steps, lr=args.lr,
            batch_size=args.batch_size, rng=np.random.default_rng(seed),
        )
        if "failed" in training:
            runs[str(seed)] = {"training": training}
            continue
        summary, frame = evaluate(model, series, val_rows, calendar, n_steps, skip, horizon, device, return_frame=True)
        column = f"seed_{seed}"
        data[column] = frame["pred"].to_numpy()
        seed_columns.append(column)
        runs[str(seed)] = {"training": training, "validation": summary}
        print(f"  seed {seed}: {training} | Rank IC {summary['rank_ic']:+.4f} (t={summary['rank_ic_t']:+.2f})", flush=True)

    # Saved first, so that a failure below cannot lose the runs.
    args.scores_out.parent.mkdir(parents=True, exist_ok=True)
    data.drop(columns="y").to_parquet(args.scores_out, index=False)

    ranks = data.groupby("date")[seed_columns + ["zero_shot", "reversal", "momentum"]].rank(pct=True) - 0.5
    ranked = pd.concat([data[["date", "y"]], ranks], axis=1)
    ranked["seed_mean"] = ranks[seed_columns].mean(axis=1)
    ranked["seed_mean_rank"] = ranked.groupby("date")["seed_mean"].rank(pct=True) - 0.5
    ranked["with_reversal"] = ranked["seed_mean_rank"] + ranked["reversal"]

    def ic_of(column: str) -> pd.Series:
        return cross_sectional_ic(ranked, column, "y")

    def describe(column: str) -> dict:
        ic = ic_of(column)
        return {"rank_ic": float(ic.mean()), "t": hac_t_stat(ic, settings["hac_lags"])}

    def correlation(left: str, right: str) -> float:
        return float(cross_sectional_ic(ranked, left, right).mean())

    gap = (ic_of("seed_mean_rank") - ic_of("reversal")).dropna()
    low, high = block_bootstrap_ci(gap, block=settings["bootstrap_block"], reps=settings["bootstrap_reps"], seed=settings["bootstrap_seed"])
    single = [runs[c.split("_")[1]]["validation"]["rank_ic"] for c in seed_columns]
    pairs = [correlation(a, b) for i, a in enumerate(seed_columns) for b in seed_columns[i + 1 :]]

    return {
        "settings": {"steps": args.steps, "scope": args.scope, "lr": args.lr, "batch_size": args.batch_size},
        "zero_shot": zero_shot,
        "runs": runs,
        "single_seed_rank_ic": {"mean": float(np.mean(single)), "std": float(np.std(single, ddof=1)), "min": min(single), "max": max(single)},
        "signals": {name: describe(name) for name in ("seed_mean_rank", "reversal", "momentum", "zero_shot", "with_reversal")},
        "correlation_of_seed_mean_with": {name: correlation("seed_mean_rank", name) for name in ("reversal", "momentum", "zero_shot")},
        "mean_correlation_between_seeds": float(np.mean(pairs)),
        "ic_gap_with_reversal": {"mean": float(gap.mean()), "t": hac_t_stat(gap, settings["hac_lags"]), "ci_95": [low, high]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=ROOT / "data" / "clean" / "panel.parquet")
    parser.add_argument("--spec", type=Path, default=ROOT / "config" / "spec.yaml")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--scope", default="last_4")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--losses", default="pinball,correlation")
    parser.add_argument("--skip-checks", action="store_true", help="Only run the fine-tuning runs")
    parser.add_argument("--diagnose-nan", action="store_true", help="Name the operation that produces invalid gradients")
    parser.add_argument("--seeds", default="", help="Comma-separated seeds for the multi-seed study of the correlation loss")
    parser.add_argument("--scores-out", type=Path, default=ROOT / "data" / "predictions" / "finetune_seed_scores.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "finetune_feasibility.json")
    args = parser.parse_args()

    from timesfm3 import ModelConfig, TimesFM3Evaluator

    spec = yaml.safe_load(args.spec.read_text())
    skip, horizon = spec["target"]["skip"], spec["target"]["horizon"]
    n_steps = skip + horizon
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    panel = pd.read_parquet(
        args.panel,
        columns=["permno", "date", "ret", "prc", "dollar_vol", "mcap"],
        filters=[("date", "<", pd.Timestamp(LAST_DATE_READ))],
    )
    wide = to_wide(panel)
    del panel
    assert wide.ret.index.max() < pd.Timestamp(LAST_DATE_READ)
    rows, tradable, members = universe_rows(wide, spec)
    ret = wide.ret[members]
    calendar = ret.index
    series = np.nan_to_num(ret.to_numpy(dtype=np.float32), nan=0.0)
    history = ret.notna().cumsum().to_numpy()
    del wide

    purge = skip + horizon
    val_start = int(calendar.searchsorted(pd.Timestamp(VAL_START)))
    train_rows = eligible_rows(rows.to_numpy(), history, CONTEXT, 0, val_start - purge)
    val_rows = eligible_rows(tradable.to_numpy(), history, CONTEXT, val_start, len(calendar) - n_steps)
    print(
        f"Train: {len(train_rows):,} windows through {calendar[train_rows[:, 0].max()].date()} | "
        f"Validation: {len(val_rows):,} windows from {calendar[val_rows[:, 0].min()].date()} "
        f"through {calendar[val_rows[:, 0].max()].date()}",
        flush=True,
    )

    forecaster = TimesFM3Evaluator(
        ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", per_core_batch_size=64, device="cuda")
    )
    model = forecaster.model
    device = next(model.parameters()).device
    pretrained = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    report: dict = {
        "gpu": torch.cuda.get_device_name(device),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(device).total_memory / 1024**3, 1),
        "total_parameters_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "train_windows": len(train_rows),
    }

    def save() -> None:
        args.out.write_text(json.dumps(report, indent=2))

    def restore() -> None:
        model.load_state_dict(pretrained)

    if not args.skip_checks:
        print("\n1. Parity with the prediction path", flush=True)
        report["parity"] = check_parity(forecaster, series, val_rows, n_steps, device)
        print(f"  {report['parity']}", flush=True)
        save()

        print("\n2. Gradient flow through the full model", flush=True)
        report["gradients"] = check_gradients(model, series, train_rows, n_steps, device)
        print(f"  {report['gradients']}", flush=True)
        save()

        print("\n3. Memory and time per step", flush=True)
        report["cost"] = measure_cost(
            model, series, train_rows, n_steps, device, ["head", "last_2", "last_4", "last_8", "full"], [16, 32, 64, 128], rng
        )
        restore()
        save()

        print("\n4. Overfitting one fixed batch", flush=True)
        report["overfit"] = overfit_one_batch(model, series, train_rows, n_steps, device, args.scope, 150, 1e-4)
        print(f"  {report['overfit']}", flush=True)
        restore()
        save()

        print("\n5. Zero-shot on validation (2003-2004)", flush=True)
        report["validation_zero_shot"] = evaluate(model, series, val_rows, calendar, n_steps, skip, horizon, device)
        print(f"  {report['validation_zero_shot']}", flush=True)
        save()

    if args.diagnose_nan:
        print("\nLooking for the operation that produces invalid gradients", flush=True)
        restore()
        torch.autograd.set_detect_anomaly(True, check_nan=True)
        try:
            outcome = train_run(
                model, series, train_rows, n_steps, skip, horizon, device,
                loss_name="correlation", scope=args.scope, steps=200, lr=args.lr, batch_size=args.batch_size, rng=rng,
            )
            report["nan_diagnosis"] = {"anomaly": None, "outcome": outcome}
        except RuntimeError as error:
            report["nan_diagnosis"] = {"anomaly": str(error).splitlines()[0][:300]}
        finally:
            torch.autograd.set_detect_anomaly(False)
        print(f"  {report['nan_diagnosis']}", flush=True)
        save()

    if args.seeds:
        print(f"\nCorrelation loss over seeds {args.seeds}", flush=True)
        report["seed_study"] = seed_study(
            model, restore, series, train_rows, val_rows, calendar, members.to_numpy(),
            n_steps, skip, horizon, device, args, spec["evaluation"],
        )
        print(json.dumps({k: v for k, v in report["seed_study"].items() if k not in ("runs", "zero_shot")}, indent=1), flush=True)
        save()

    report["runs"] = {}
    for loss_name in [name for name in args.losses.split(",") if name]:
        print(f"\n6. Fine-tuning with the {loss_name} loss ({args.steps} steps, {args.scope}, lr {args.lr})", flush=True)
        restore()
        training = train_run(
            model, series, train_rows, n_steps, skip, horizon, device,
            loss_name=loss_name, scope=args.scope, steps=args.steps, lr=args.lr, batch_size=args.batch_size, rng=rng,
        )
        print(f"  training: {training}", flush=True)
        run = {"training": training}
        if "failed" not in training:
            run["validation"] = evaluate(model, series, val_rows, calendar, n_steps, skip, horizon, device)
            print(f"  validation: {run['validation']}", flush=True)
        report["runs"][loss_name] = run
        save()
    print("\nDone", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
