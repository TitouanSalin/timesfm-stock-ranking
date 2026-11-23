"""Predictive and portfolio metrics, with inference robust to overlapping labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252


def cross_sectional_ic(frame: pd.DataFrame, pred: str, target: str, method: str = "spearman") -> pd.Series:
    """Correlation between predictions and outcomes on each date."""
    data = frame[["date", pred, target]].dropna()
    x, y = data[pred], data[target]
    if method == "spearman":
        x = x.groupby(data["date"]).rank()
        y = y.groupby(data["date"]).rank()
    elif method != "pearson":
        raise ValueError(f"Unknown method: {method}")

    dx = x - x.groupby(data["date"]).transform("mean")
    dy = y - y.groupby(data["date"]).transform("mean")
    numerator = (dx * dy).groupby(data["date"]).sum()
    denominator = np.sqrt((dx**2).groupby(data["date"]).sum() * (dy**2).groupby(data["date"]).sum())
    return (numerator / denominator.where(denominator > 0)).dropna()


def hac_t_stat(series: pd.Series, lags: int) -> float:
    """t-statistic of the mean with a Newey-West (Bartlett kernel) variance."""
    values = series.dropna().to_numpy(dtype=float)
    n = len(values)
    if n < 2:
        return float("nan")
    errors = values - values.mean()
    variance = errors @ errors / n
    for lag in range(1, min(lags, n - 1) + 1):
        variance += 2 * (1 - lag / (lags + 1)) * (errors[lag:] @ errors[:-lag]) / n
    if variance <= 0:
        return float("nan")
    return float(values.mean() / np.sqrt(variance / n))


def block_bootstrap_ci(
    series: pd.Series, *, block: int, reps: int, seed: int, level: float = 0.95
) -> tuple[float, float]:
    """Moving-block bootstrap interval for the mean of a dependent series."""
    values = series.dropna().to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        raise ValueError("Cannot bootstrap an empty series")
    block = min(block, n)
    n_blocks = int(np.ceil(n / block))
    starts = np.random.default_rng(seed).integers(0, n - block + 1, size=(reps, n_blocks))
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    means = (cumulative[starts + block] - cumulative[starts]).sum(axis=1) / (n_blocks * block)
    low, high = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(low), float(high)


def performance(returns: pd.Series) -> dict[str, float]:
    """Annualized statistics of a daily return series."""
    values = returns.dropna()
    if values.empty:
        raise ValueError("The return series has no values on this window")
    growth = (1 + values).cumprod()
    years = len(values) / TRADING_DAYS
    volatility = values.std() * np.sqrt(TRADING_DAYS)
    return {
        "ann_return": float(growth.iloc[-1] ** (1 / years) - 1),
        "ann_vol": float(volatility),
        "sharpe": float(values.mean() * TRADING_DAYS / volatility) if volatility > 0 else float("nan"),
        "max_drawdown": float((growth / growth.cummax().clip(lower=1.0) - 1).min()),
    }


def active_performance(returns: pd.Series, benchmark: pd.Series, hac_lags: int) -> dict[str, float]:
    """Return relative to a benchmark, annualized and with its t-statistic."""
    active = (returns - benchmark).dropna()
    tracking_error = active.std() * np.sqrt(TRADING_DAYS)
    annual_active = active.mean() * TRADING_DAYS
    return {
        "ann_active": float(annual_active),
        "tracking_error": float(tracking_error),
        "information_ratio": float(annual_active / tracking_error) if tracking_error > 0 else float("nan"),
        "active_t": hac_t_stat(active, hac_lags),
    }


def factor_alpha(excess_returns: pd.Series, factors: pd.DataFrame, hac_lags: int) -> dict[str, float]:
    """Annualized alpha and factor loadings from OLS, with a Newey-West t-statistic for alpha."""
    data = pd.concat([excess_returns.rename("y"), factors], axis=1, join="inner").dropna()
    y = data["y"].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(data)), data[factors.columns].to_numpy(dtype=float)])
    inverse = np.linalg.inv(x.T @ x)
    coefficients = inverse @ x.T @ y
    scores = x * (y - x @ coefficients)[:, None]
    meat = scores.T @ scores
    for lag in range(1, hac_lags + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (hac_lags + 1)) * (cross + cross.T)
    covariance = inverse @ meat @ inverse
    return {
        "alpha_annual": float(coefficients[0] * TRADING_DAYS),
        "alpha_t": float(coefficients[0] / np.sqrt(covariance[0, 0])),
        **{f"beta_{name}": float(value) for name, value in zip(factors.columns, coefficients[1:])},
    }


def deflated_sharpe(returns: pd.Series, n_trials: int, trial_sharpe_variance: float) -> float:
    """Probability that the true Sharpe exceeds the best expected among n_trials null trials.

    Bailey and Lopez de Prado (2014). Sharpe ratios are per period, not annualized,
    and trial_sharpe_variance is the variance of per-period Sharpe across trials.
    """
    values = returns.dropna().to_numpy(dtype=float)
    n = len(values)
    sharpe = values.mean() / values.std(ddof=1)
    expected_max = 0.0
    if n_trials > 1:
        euler = 0.5772156649
        expected_max = np.sqrt(trial_sharpe_variance) * (
            (1 - euler) * norm.ppf(1 - 1 / n_trials) + euler * norm.ppf(1 - 1 / (n_trials * np.e))
        )
    centred = values - values.mean()
    scale = values.std(ddof=0)
    skew = np.mean(centred**3) / scale**3
    kurtosis = np.mean(centred**4) / scale**4
    denominator = np.sqrt(1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2)
    return float(norm.cdf((sharpe - expected_max) * np.sqrt(n - 1) / denominator))
