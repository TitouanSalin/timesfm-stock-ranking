"""Daily-decision, long-only portfolio backtest with a rank buffer."""

from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_HALF_SPREAD = 0.001  # 10 bp, deliberately above the usual level of these stocks


def backtest(
    scores: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    n_positions: int,
    exit_rank: int,
    max_weight: float,
    cost_bps: float,
    half_spread: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Simulate the portfolio over the dates covered by the scores.

    A score from the close of t (NaN means not tradable) is executed at the close
    of t+1 and earns returns from t+2 onward.

    A holding is sold when its rank exceeds exit_rank. Freed capital is shared by
    the best-ranked non-holders needed to return to n_positions. A holding above
    max_weight is cut back to the equal weight, and entrants are capped at
    max_weight, so any excess waits in cash at a zero return. A stock without a
    quote on the execution date cannot be traded. A delisted holding moves to cash
    after its final return, at no cost.

    Each trade costs cost_bps per side, plus the stock's half spread on the execution
    date when half_spread is given (MISSING_HALF_SPREAD if it is unknown that day).

    Each row covers one day: gross, cost, net, turnover (one-way), n_positions, cash.
    """
    ret_values = ret.to_numpy(dtype=float)
    score_values = scores.reindex(index=ret.index, columns=ret.columns).to_numpy(dtype=float)
    quoted = np.isfinite(ret_values)
    n_dates = len(ret_values)
    last_quote = n_dates - 1 - np.argmax(quoted[::-1], axis=0)

    signal_days = np.flatnonzero(np.isfinite(score_values).any(axis=1))
    columns = ["gross", "cost", "net", "turnover", "n_positions", "cash"]
    if len(signal_days) == 0:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"), dtype=float)

    fee_rate = cost_bps / 1e4
    spreads = None
    if half_spread is not None:
        spreads = half_spread.reindex(index=ret.index, columns=ret.columns).to_numpy(dtype=float)
    equal_weight = 1.0 / n_positions
    weights: dict[int, float] = {}
    cash = 1.0
    rows, dates = [], []

    for day in range(signal_days[0] + 1, min(signal_days[-1] + 2, n_dates)):
        # Day's return on the holdings set at the previous close.
        gross = 0.0
        for stock, weight in weights.items():
            if quoted[day, stock]:
                gross += weight * ret_values[day, stock]
                weights[stock] = weight * (1 + ret_values[day, stock])
        nav = 1.0 + gross
        weights = {stock: weight / nav for stock, weight in weights.items()}
        cash /= nav

        for stock in [s for s in weights if day >= last_quote[s]]:
            cash += weights.pop(stock)

        # Trades at this close, from the signal of the previous close.
        traded = 0.0
        cost = 0.0

        def charge(stock: int, amount: float) -> float:
            """Cost of trading this amount of the stock on this day."""
            if spreads is None:
                return amount * fee_rate
            spread = spreads[day, stock]
            return amount * (fee_rate + (spread if np.isfinite(spread) else MISSING_HALF_SPREAD))

        signal = score_values[day - 1]
        scored = np.flatnonzero(np.isfinite(signal))
        if len(scored):
            order = scored[np.argsort(-signal[scored], kind="stable")]
            rank = {stock: position + 1 for position, stock in enumerate(order)}

            sold = {s for s in weights if rank.get(s, np.inf) > exit_rank and quoted[day, s]}
            for stock in sold:
                weight = weights.pop(stock)
                cash += weight
                traded += weight
                cost += charge(stock, weight)

            for stock, weight in weights.items():
                if weight > max_weight and quoted[day, stock]:
                    excess = weight - equal_weight
                    weights[stock] = equal_weight
                    cash += excess
                    traded += excess
                    cost += charge(stock, excess)

            open_slots = n_positions - len(weights)
            entrants = []
            for stock in order:
                if len(entrants) == open_slots:
                    break
                if stock not in weights and stock not in sold and quoted[day, stock]:
                    entrants.append(stock)
            if entrants:
                allocation = min(cash / len(entrants), max_weight)
                for stock in entrants:
                    weights[stock] = allocation
                    cost += charge(stock, allocation)
                invested = allocation * len(entrants)
                cash -= invested
                traded += invested

        rows.append((gross, cost, gross - cost, traded / 2, len(weights), cash))
        dates.append(ret.index[day])

    return pd.DataFrame(rows, columns=columns, index=pd.DatetimeIndex(dates, name="date"))


def equal_weight_benchmark(ret: pd.DataFrame, trade_mask: pd.DataFrame) -> pd.Series:
    """Daily return, before costs, of holding the previous close's tradable stocks equally."""
    return ret.where(trade_mask.shift(1, fill_value=False)).mean(axis=1)
