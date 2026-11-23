"""Combine model scores and smooth them over time, using equal weights and no fitting."""

from __future__ import annotations

import pandas as pd


def combine_scores(sources: dict[str, pd.DataFrame], tradable: pd.DataFrame) -> pd.DataFrame:
    """Average each model's within-date rank over tradable stocks.

    Ranks put models on a common scale. A stock a model did not score on a date is
    averaged over the other models. Stocks that no model scored are dropped.
    """
    # The merged scores have a fresh index, so the keys must too for the rows to line up.
    keys = tradable[["date", "permno"]].reset_index(drop=True)
    ranks = []
    for name, predictions in sources.items():
        scored = keys.merge(predictions[["date", "permno", "pred"]], on=["date", "permno"], how="left", validate="one_to_one")
        ranks.append((scored.groupby("date")["pred"].rank(pct=True) - 0.5).rename(name))
    combined = keys.assign(pred=pd.concat(ranks, axis=1).mean(axis=1))
    return combined.dropna(subset=["pred"]).reset_index(drop=True)


def resolve_ensemble(name: str, ensembles: dict[str, list[str]], scores: dict[str, pd.DataFrame], tradable: pd.DataFrame) -> pd.DataFrame:
    """Score of a named ensemble, where a source naming another ensemble counts as one source.

    This keeps several seeds of a model from outweighing the other sources. An ensemble
    may list a model of its own name, which then means that model.
    """
    sources = {
        source: resolve_ensemble(source, ensembles, scores, tradable) if source in ensembles and source != name else scores[source]
        for source in ensembles[name]
    }
    return combine_scores(sources, tradable)


def smooth_scores(predictions: pd.DataFrame, window: int) -> pd.DataFrame:
    """Average each stock's within-date rank over the last `window` scoring dates.

    This only uses past and current scores, so a decision can still be taken every
    day. A stock is only scored on dates when it had a score to begin with.
    """
    wide = predictions.pivot(index="date", columns="permno", values="pred")
    ranks = wide.rank(axis=1, pct=True) - 0.5
    smoothed = ranks.rolling(window, min_periods=1).mean().where(wide.notna())
    return smoothed.rename_axis(columns="permno").stack().rename("pred").reset_index()
