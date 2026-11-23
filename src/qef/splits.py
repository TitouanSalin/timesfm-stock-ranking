"""Walk-forward folds with purging and a guarded final test period."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    test_year: int
    train_dates: pd.DatetimeIndex    # used to fit candidate hyperparameters
    val_dates: pd.DatetimeIndex      # used to choose among them
    refit_dates: pd.DatetimeIndex    # used to fit the final model of the fold
    test_dates: pd.DatetimeIndex


def dates_before_holdout(calendar: pd.DatetimeIndex, holdout_start: str, purge: int) -> pd.DatetimeIndex:
    """Dates whose label is fully known before the first date of the holdout."""
    position = int(calendar.searchsorted(pd.Timestamp(holdout_start)))
    return calendar[: max(position - purge, 0)]


def plan_folds(calendar: pd.DatetimeIndex, spec: dict, *, final_test: bool = False) -> list[Fold]:
    """Yearly folds of the development period, or of the final test when final_test is set.

    The final test is opened once, with the configuration frozen in config/frozen.yaml.
    """
    plan, target = spec["walk_forward"], spec["target"]
    first, last = plan["first_test_year"], plan["last_dev_year"]
    if final_test:
        first, last = pd.Timestamp(plan["holdout_start"]).year, pd.Timestamp(plan["panel_end"]).year
    return walk_forward_folds(
        calendar,
        first_test_year=first,
        last_test_year=last,
        purge=target["horizon"] + target["skip"],
        inner_val_years=plan["inner_val_years"],
        holdout_start=plan["holdout_start"],
        allow_holdout=final_test,
    )


def group_folds(folds: list[Fold], every: int) -> list[Fold]:
    """Retrain every few years: a group keeps its first fold's training dates and tests on all its years."""
    groups = []
    for start in range(0, len(folds), every):
        first, *others = folds[start : start + every]
        groups.append(replace(first, test_dates=first.test_dates.append([fold.test_dates for fold in others])))
    return groups


def walk_forward_folds(
    calendar: pd.DatetimeIndex,
    *,
    first_test_year: int,
    last_test_year: int,
    purge: int,
    inner_val_years: int,
    holdout_start: str | None = None,
    allow_holdout: bool = False,
) -> list[Fold]:
    """Build one fold per test year with an expanding training window.

    A label at date t is only known at t+purge, so a date may be used for training
    only if t+purge falls before the first date of the next phase. Unless
    allow_holdout is set, no test date and no label may reach the holdout.
    """
    calendar = pd.DatetimeIndex(calendar)
    years = calendar.year
    holdout_position = len(calendar)
    if holdout_start is not None:
        holdout_position = int(calendar.searchsorted(pd.Timestamp(holdout_start)))

    def before(position: int) -> pd.DatetimeIndex:
        """Dates whose label is fully known before the given calendar position."""
        return calendar[: max(position - purge, 0)]

    folds = []
    for year in range(first_test_year, last_test_year + 1):
        positions = np.flatnonzero(years == year)
        if len(positions) == 0:
            continue
        if not allow_holdout:
            if positions[0] >= holdout_position:
                raise ValueError(f"Test year {year} is in the holdout, which requires allow_holdout=True")
            positions = positions[positions + purge < holdout_position]
            if len(positions) == 0:
                continue

        test_start = int(positions[0])
        val_positions = np.flatnonzero((years >= year - inner_val_years) & (years < year))
        if len(val_positions) == 0:
            raise ValueError(f"No validation dates before test year {year}")
        val_start = int(val_positions[0])

        refit_dates = before(test_start)
        folds.append(
            Fold(
                test_year=year,
                train_dates=before(val_start),
                val_dates=calendar[val_start:][calendar[val_start:].isin(refit_dates)],
                refit_dates=refit_dates,
                test_dates=calendar[positions],
            )
        )
    return folds
