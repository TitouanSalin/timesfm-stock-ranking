# Primary specification (version 1)

Frozen on 3 November 2026, before any result. The executable values are in [config/spec.yaml](../config/spec.yaml). Everything else (other horizons, other targets, other portfolio sizes) is exploratory and is not used to claim a result.

## Research question

Does a time-series foundation model (TimesFM 3), possibly fine-tuned, improve the ranking of U.S. stocks and the resulting portfolio, compared with simple references and LightGBM, out of sample and after costs?

## Timing of a decision

```text
Session           t              t+1            t+2  ...  t+6
Signal            computed after
                  the close
Execution                        at the close
Target                                          r(t+2) ... r(t+6)
```

The target at date `t` is the total return from the close of `t+1` to the close of `t+6`, minus the mean of the training universe on that date. Training uses the rank of this value within each date. Executing at the close of `t+1` avoids assuming a purchase at the very close used to compute the signal.

## Universes

At each month-end, using only data known on that date:

- **Training universe:** the 1,000 largest market values, with a price of at least USD 5 and at least 252 sessions of history.
- **Tradable universe:** the 100 stocks with the largest 63-session median dollar volume, among the 500 largest market values.

These universes apply from the next session. The identifier is the `PERMNO`.

**Eligible stocks (table `crsp.dsf_v2`):** ordinary common shares of U.S. companies, normally traded on a U.S. exchange, according to the codes known at each date.

**Delistings:** in `dsf_v2`, the delisting return sits on a separate row (`dlydelflg = 'Y'`) that the standard common-stock filter removes. The pipeline keeps it explicitly, otherwise delisting losses would vanish from the backtest. When this return is missing, it is set to -30% (Shumway, 1997). On that day the stock no longer has a price: it can no longer be ranked or traded, but its holder takes the loss.

## Portfolio

Decisions are taken every day, but a position is only traded when its rank justifies it:

- The portfolio targets 20 stocks, long-only and without leverage.
- A holding is sold when its rank exceeds 40.
- The best-ranked stocks replace the exits and share the freed capital.
- A position above 7.5% of the portfolio is cut back to 5%.

With a 5-day target and daily decisions, turnover is the main risk of this specification. It is therefore measured and reported for every model.

## Time split

| Period | Dates | Use |
|---|---|---|
| Warm-up and first training | 1995 to 2004 | Training only |
| Development | 2005 to 2021 | Yearly walk-forward, 17 out-of-sample years |
| Final test | 2022 to 2025 | Evaluated once, with the frozen configuration |

CRSP ends on 31 December 2025 in this WRDS edition, so 2026 is not available.

- Each test year uses a model retrained on all available past data.
- The 3 preceding years are used to choose hyperparameters.
- A purge of 6 sessions (horizon + lag) separates each phase, because a target at date `t` is only known at `t+6`.

With t ≈ information ratio × √years, 17 years can reach t ≈ 2 for an information ratio of about 0.5. The code refuses to build a split that touches the final test without explicit permission.

## Models

| Level | Models |
|---|---|
| References | Equal-weight universe (no skill), 12-1 momentum, 5-day reversal |
| Tabular models | Ridge, LightGBM |
| Foundation model | TimesFM 3 zero-shot, as a predictor and then as a LightGBM feature |
| Last | Fine-tuned TimesFM, if a compatible implementation exists |

Version 1 only uses variables derived from CRSP. Point-in-time Compustat and I/B/E/S are then added one at a time, measuring their contribution.

## Evaluation

- **Signal:** mean Rank IC per date over the tradable universe, with a Newey-West t-statistic and a block-bootstrap interval.
- **Portfolio:** active return and information ratio against the equal-weight universe, net of costs (2, 5 and 10 basis points per side, 5 being the central scenario). An additional scenario uses each stock's closing half-spread plus 1 basis point.
- **Alpha:** regression on the market, size, value, profitability, investment and momentum factors.
- **Monitoring:** turnover, drawdown, results by year, spread between extreme deciles (a measure of the signal, not traded).
- **Comparison with the S&P 500:** reported but not used as a criterion, because it mostly measures the effect of market-value weighting.

## Choice of the best model

1. In each year, hyperparameters are chosen on the validation Rank IC.
2. Between models, the criterion is the net information ratio at the central cost over the development period.
3. A more complex model only replaces the previous one if the paired bootstrap of the Rank IC or active return difference excludes zero.
4. A result is considered established if the Newey-West t exceeds 2 and the Deflated Sharpe Ratio exceeds 0.95, given all the trials in the registry `experiments/registry.jsonl`.
5. The final test is evaluated once and reported as is, even if disappointing.

## Additions after freezing version 1

These additions change neither the target, nor the universes, nor the periods, nor the reference portfolio. Every run counts as a trial in the registry.

- **Point-in-time Compustat:** a quarterly value is `coalesce(xr, x)` from `comp_pit.pitqtrdataus`, which equals the first known value in over 99.7% of cases from 1998 to 2024. It is only usable after the first monthly snapshot that shows the quarter, and a late filing also delays the following quarters.
- **I/B/E/S:** the consensus comes from the unadjusted table `statsumu_epsus`, divided by the price on the same day, because the adjusted table embeds future splits. A consensus counts from the day after its date, and an earnings surprise from the day after its announcement.
- **Validity periods:** 270 days for accounting data, 45 days for the consensus and 100 days for a surprise.
- **Ensembles:** equal-weight average of within-date ranks, with no estimated weights, from lists fixed in advance in `config/study_*.yaml`.
- **Grid average (`LightGBMAverage`):** it departs from rule 1, since no hyperparameter is selected. It averages the ranks of 12 LightGBM versions: the 4 tree settings at 100, 200 and 400 rounds.
- **Fine-tuned TimesFM, reduced version:** it departs from rule 1 and from yearly retraining, with retraining every 3 years on all available past data and a 6-session purge. The setting is unique and was fixed on data before 2005: per-date correlation loss, last 4 layers, 1,000 steps, learning rate 3e-5, batches of 64. The score is the average rank of 5 seeds. This test therefore covers a single setting, without hyperparameter search, and does not use the fine-tuned score as a LightGBM feature.
- **Fine-tuned TimesFM with 10,000 steps:** only the number of steps changes from the reduced version. Rule fixed before the results: this version replaces the 1,000-step version as the TimesFM comparator of the final test if its net information ratio at the central cost is higher. It only replaces the candidate `avg3_signals_s1_x40` if its ratio exceeds the candidate's, with a paired bootstrap whose interval on the Rank IC or net return difference excludes zero.

Rule 3 applies to these additions as to the models of version 1.

## Final test

Configuration frozen on 7 November 2026, before any evaluation on 2022-2025, in [config/frozen.yaml](../config/frozen.yaml), which also records the code hash.

- **Primary model:** `avg3_signals_s1_x40`, the only model judged by the final test.
- **Comparators declared in advance:** reversal, momentum, baseline LightGBM, zero-shot TimesFM, TimesFM fine-tuned with 10,000 steps, and fine-tuned TimesFM + reversal.
- **Results:** written separately (`reports/final/`, `data/predictions/final/`, `experiments/final_test.jsonl`) and reported as is by `scripts/50_final_report.py`, with only the paired comparisons declared in `config/frozen.yaml`.
- **Power:** four years only give t ≈ 2 × the information ratio. A ratio of 0.5 therefore remains non-significant, even if it is real.

## Licenses

- **WRDS data:** used under a university license; raw data are never redistributed or committed.
- **TimesFM 3.0 (license read on 3 November 2026):** the code is under Apache-2.0, but the weights are under `timesfm-non-commercial-license-v1.0`.
  - Academic research and experimentation are allowed, as long as the results are not used for commercial decisions.
  - Any revenue-generating activity is excluded, including through the model's predictions. Trading real money with the self-hosted weights would therefore fall outside this scope.
  - The weights, including fine-tuned ones, may not be distributed.
  - According to the repository, weights up to version 2.5 are under Apache-2.0, and commercial use of 3.0 goes through Google Cloud services.

## Known constraints of TimesFM 3.0

From `config.json`: at most 32 variates for multivariate attention (`max_variates`), input patches of 32 steps, output patches of 64 steps, and quantiles from 0.1 to 0.9. The 100 stocks therefore cannot be given as a single block.
