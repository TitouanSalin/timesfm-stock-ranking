"""Out-of-sample prediction, evaluation and experiment logging."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .dataset import Dataset
from .metrics import (
    TRADING_DAYS,
    active_performance,
    block_bootstrap_ci,
    cross_sectional_ic,
    factor_alpha,
    hac_t_stat,
    performance,
)
from .portfolio import backtest, equal_weight_benchmark
from .splits import Fold


def run_walk_forward(
    dataset: Dataset,
    model_factory: Callable[[], object],
    folds: list[Fold],
    features: list[str],
    log: Callable[[str], None] = print,
) -> tuple[pd.DataFrame, list[dict]]:
    """Fit one model per fold. Return the out-of-sample scores and each fold's selection."""
    frame = dataset.frame
    labelled = frame[frame["y_rank"].notna()]
    predictions, selections = [], []

    for fold in folds:
        model = model_factory().fit(
            labelled[labelled["date"].isin(fold.train_dates)],
            labelled[labelled["date"].isin(fold.val_dates)],
            labelled[labelled["date"].isin(fold.refit_dates)],
            features,
        )
        test = frame[frame["date"].isin(fold.test_dates)]
        predictions.append(test[["date", "permno"]].assign(pred=model.predict(test), fold=fold.test_year))
        selections.append({"test_year": fold.test_year, **model.selected})
        log(f"  {fold.test_year}: {model.selected}")

    return pd.concat(predictions, ignore_index=True), selections


def evaluate(
    predictions: pd.DataFrame,
    dataset: Dataset,
    spec: dict,
    references: dict[str, pd.Series] | None = None,
    factors: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Signal quality on tradable stocks and portfolio results at each cost level.

    references are daily return series reported for comparison, such as the S&P 500.
    factors must hold the daily factor returns and rf. Returns the summary and the
    daily series that model comparisons need.
    """
    settings = spec["evaluation"]
    horizon = spec["target"]["horizon"]
    labels = dataset.frame[["date", "permno", "tradable", "y_excess"]]
    merged = predictions.merge(labels, on=["date", "permno"], validate="one_to_one")
    tradable = merged[merged["tradable"]].dropna(subset=["pred"])

    rank_ic = cross_sectional_ic(tradable, "pred", "y_excess", "spearman")
    pearson_ic = cross_sectional_ic(tradable, "pred", "y_excess", "pearson")
    low, high = block_bootstrap_ci(
        rank_ic,
        block=settings["bootstrap_block"],
        reps=settings["bootstrap_reps"],
        seed=settings["bootstrap_seed"],
    )

    # The extreme-decile spread measures the signal; it is not a traded portfolio.
    known = tradable.dropna(subset=["y_excess"])
    percentile = known.groupby("date")["pred"].rank(pct=True)
    spread = known[percentile > 0.9].groupby("date")["y_excess"].mean() - (
        known[percentile <= 0.1].groupby("date")["y_excess"].mean()
    )

    signal = {
        "n_dates": int(rank_ic.size),
        "rank_ic": float(rank_ic.mean()),
        "rank_ic_t": hac_t_stat(rank_ic, settings["hac_lags"]),
        "rank_ic_ci": [low, high],
        "pearson_ic": float(pearson_ic.mean()),
        "hit_rate": float(((known["pred"] - known.groupby("date")["pred"].transform("mean")) * known["y_excess"] > 0).mean()),
        "decile_spread_annual": float(spread.mean() * TRADING_DAYS / horizon),
        "rank_ic_by_year": {str(y): float(v) for y, v in rank_ic.groupby(rank_ic.index.year).mean().items()},
    }

    scores = tradable.pivot(index="date", columns="permno", values="pred")
    benchmark = equal_weight_benchmark(dataset.ret, dataset.trade_mask)
    book = spec["portfolio"]
    scenarios = {name: (bps, None) for name, bps in spec["costs_bps_per_side"].items()}
    if "measured_cost_fee_bps" in spec and dataset.half_spread is not None:
        scenarios["measured"] = (spec["measured_cost_fee_bps"], dataset.half_spread)

    portfolios: dict[str, dict] = {}
    daily = pd.DataFrame({"rank_ic": rank_ic})
    for scenario, (cost_bps, half_spread) in scenarios.items():
        result = backtest(scores, dataset.ret, cost_bps=cost_bps, half_spread=half_spread, **book)
        aligned = benchmark.reindex(result.index)
        portfolios[scenario] = {
            "cost_bps": cost_bps,
            **performance(result["net"]),
            **active_performance(result["net"], aligned, settings["hac_lags"]),
            "annual_turnover": float(result["turnover"].mean() * TRADING_DAYS),
            "annual_cost": float(result["cost"].mean() * TRADING_DAYS),
            "mean_positions": float(result["n_positions"].mean()),
            "mean_cash": float(result["cash"].mean()),
        }
        if factors is not None:
            excess = result["net"] - factors["rf"].reindex(result.index)
            portfolios[scenario].update(
                factor_alpha(excess, factors.drop(columns="rf"), settings["hac_lags"])
            )
        daily = daily.join(result["net"].rename(f"net_{scenario}"), how="outer")

    window = result.index
    daily = daily.join(benchmark.reindex(window).rename("benchmark"), how="outer")
    summary = {
        "signal": signal,
        "portfolio": portfolios,
        "benchmark": performance(benchmark.reindex(window)),
        "references": {name: performance(series.reindex(window)) for name, series in (references or {}).items()},
    }
    return summary, daily


def trial_sharpe_variance(folders: list[Path], scenario: str) -> float:
    """Variance of the daily active Sharpe across every saved run.

    The Deflated Sharpe needs the spread of all the trials, so it must not depend on
    which models happen to be compared.
    """
    sharpes = []
    for folder in folders:
        for path in sorted(folder.glob("*_daily.parquet")):
            frame = pd.read_parquet(path)
            active = (frame[f"net_{scenario}"] - frame["benchmark"]).dropna()
            sharpes.append(active.mean() / active.std())
    return float(np.var(sharpes, ddof=1))


def load_market_data(raw: Path | None) -> tuple[dict[str, pd.Series], pd.DataFrame | None]:
    """Load the S&P 500 total return and the factor returns, if they have been extracted."""
    if raw is None or not (raw / "sp500.parquet").exists() or not (raw / "ff_daily.parquet").exists():
        return {}, None
    sp500 = pd.read_parquet(raw / "sp500.parquet")
    sp500["date"] = pd.to_datetime(sp500["date"])
    factors = pd.read_parquet(raw / "ff_daily.parquet")
    factors["date"] = pd.to_datetime(factors["date"])
    factors = factors.set_index("date").astype(float)
    return {"sp500": sp500.set_index("date")["vwretd"].astype(float)}, factors


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def code_hash() -> str:
    """Hash of the package source, which identifies the code even without a Git commit."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _git_state(cwd: Path) -> dict:
    def run(*args: str) -> str:
        done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else ""

    return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}


def log_experiment(path: Path, *, model: str, spec: dict, data_version: str, results: dict, selections: list) -> None:
    """Append a run to the registry. Every run counts toward the Deflated Sharpe trial count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "spec_hash": spec_hash(spec),
        "code_hash": code_hash(),
        "data_version": data_version,
        "git": _git_state(path.resolve().parent),
        "results": results,
        "selections": selections,
    }
    with path.open("a") as registry:
        registry.write(json.dumps(record, default=float) + "\n")
