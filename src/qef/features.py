"""Market-data features. A value at date t only uses data up to the close of t."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import WidePanel


def market_return(wide: WidePanel) -> pd.Series:
    """Daily cap-weighted return of the whole panel, with weights from the previous close."""
    weights = wide.mcap.shift(1)
    weighted = (wide.ret * weights).sum(axis=1)
    total = weights.where(wide.ret.notna()).sum(axis=1)
    return (weighted / total.where(total > 0)).fillna(0.0)


def _rolling_beta_and_idio_vol(
    ret: pd.DataFrame, market: pd.Series, beta_window: int, vol_window: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_wide = pd.DataFrame(
        np.repeat(market.to_numpy()[:, None], ret.shape[1], axis=1), index=ret.index, columns=ret.columns
    ).where(ret.notna())

    def moments(window: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        min_periods = window // 2
        r_mean = ret.rolling(window, min_periods=min_periods).mean()
        m_mean = market_wide.rolling(window, min_periods=min_periods).mean()
        cov = (ret * market_wide).rolling(window, min_periods=min_periods).mean() - r_mean * m_mean
        m_var = (market_wide**2).rolling(window, min_periods=min_periods).mean() - m_mean**2
        r_var = (ret**2).rolling(window, min_periods=min_periods).mean() - r_mean**2
        return cov, m_var, r_var

    cov, m_var, _ = moments(beta_window)
    beta = cov / m_var.where(m_var > 0)

    cov, m_var, r_var = moments(vol_window)
    explained = cov**2 / m_var.where(m_var > 0)
    idio_vol = np.sqrt((r_var - explained).clip(lower=0))
    return beta, idio_vol


def stock_features(wide: WidePanel, market: pd.Series | None = None) -> dict[str, pd.DataFrame]:
    """Stock-level features as raw wide frames. Pass market when wide is a subset of stocks."""
    ret = wide.ret
    history = ret.notna().cumsum()
    log_index = np.log1p(ret.clip(lower=-0.9999).fillna(0.0)).cumsum()

    def past_return(days: int, gap: int = 0) -> pd.DataFrame:
        change = log_index.shift(gap) - log_index.shift(days)
        return change.where(history >= days)

    features: dict[str, pd.DataFrame] = {f"ret_{k}": past_return(k) for k in (1, 5, 10, 21, 63, 126, 252)}
    features["mom_252_21"] = past_return(252, gap=21)

    for window in (20, 50, 200):
        average = log_index.rolling(window).mean()
        features[f"ma_gap_{window}"] = (log_index - average).where(history >= window)
    features["drawdown_252"] = (log_index - log_index.rolling(252, min_periods=21).max()).where(history >= 21)

    for window in (5, 21, 63):
        features[f"vol_{window}"] = ret.rolling(window, min_periods=max(3, window // 2)).std()
    features["max_ret_21"] = ret.rolling(21, min_periods=10).max()
    features["skew_63"] = ret.rolling(63, min_periods=30).skew()

    if market is None:
        market = market_return(wide)
    features["beta_252"], features["idio_vol_63"] = _rolling_beta_and_idio_vol(ret, market, 252, 63)

    dollar_vol = wide.dollar_vol.where(wide.dollar_vol > 0)
    adv_5 = dollar_vol.rolling(5, min_periods=3).mean()
    adv_21 = dollar_vol.rolling(21, min_periods=10).mean()
    adv_63 = dollar_vol.rolling(63, min_periods=30).mean()
    features["log_adv_21"] = np.log(adv_21)
    features["vol_ratio_5_63"] = np.log(adv_5 / adv_63)
    features["turnover_21"] = (dollar_vol / wide.mcap).rolling(21, min_periods=10).mean()
    features["amihud_21"] = (ret.abs() / dollar_vol).rolling(21, min_periods=10).mean()
    features["log_mcap"] = np.log(wide.mcap.where(wide.mcap > 0))
    features["log_age"] = np.log(history.where(history > 0))
    return features


def market_features(market: pd.Series) -> pd.DataFrame:
    """Date-level features. They are identical across stocks, so they are not ranked."""
    log_market = np.log1p(market).cumsum()
    return pd.DataFrame(
        {
            "mkt_ret_5": log_market - log_market.shift(5),
            "mkt_ret_21": log_market - log_market.shift(21),
            "mkt_vol_21": market.rolling(21, min_periods=10).std(),
        }
    )
