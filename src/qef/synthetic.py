"""Synthetic daily panels with a known amount of predictability.

With signal=0 returns are unpredictable, so any out-of-sample skill the pipeline
reports is a leak. With signal>0 an autoregressive variable visible in dollar
volume drives the next day's return, so the pipeline should find it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_panel(
    n_stocks: int = 400, n_days: int = 2520, signal: float = 0.0, seed: int = 0, start: str = "2005-01-03"
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)

    market = rng.normal(0.0004, 0.01, n_days)
    beta = rng.uniform(0.6, 1.4, n_stocks)
    noise = rng.standard_t(5, (n_days, n_stocks)) * 0.012

    phi = 0.9
    latent = np.zeros((n_days, n_stocks))
    latent[0] = rng.normal(size=n_stocks)
    shocks = rng.normal(size=(n_days, n_stocks)) * np.sqrt(1 - phi**2)
    for day in range(1, n_days):
        latent[day] = phi * latent[day - 1] + shocks[day]

    ret = market[:, None] * beta[None, :] + noise
    ret[1:] += signal * latent[:-1]

    # About a third of the stocks list late and about a third delist early.
    birth = np.where(rng.random(n_stocks) < 0.3, rng.integers(0, n_days // 2, n_stocks), 0)
    death = np.where(rng.random(n_stocks) < 0.3, birth + rng.integers(300, n_days, n_stocks), n_days - 1)
    death = np.minimum(death, n_days - 1)
    delisting_return = rng.choice([-0.5, 0.2], n_stocks)

    price = rng.uniform(10, 200, n_stocks) * np.exp(np.cumsum(np.log1p(np.maximum(ret, -0.9)), axis=0))
    shares = np.exp(rng.normal(18, 1, n_stocks))
    dollar_vol = np.exp(rng.normal(16, 1, n_stocks)[None, :] + 0.5 * latent + rng.normal(0, 0.2, (n_days, n_stocks)))

    frames = []
    for stock in range(n_stocks):
        days = np.arange(birth[stock], death[stock] + 1)
        stock_ret = ret[days, stock].copy()
        if death[stock] < n_days - 1:
            stock_ret[-1] = delisting_return[stock]
        frames.append(
            pd.DataFrame(
                {
                    "permno": 10000 + stock,
                    "date": dates[days],
                    "ret": stock_ret,
                    "prc": price[days, stock],
                    "dollar_vol": dollar_vol[days, stock],
                    "mcap": price[days, stock] * shares[stock],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
