# Do time-series foundation models pick stocks?

A leakage-controlled, cost-aware evaluation of **TimesFM 3** (zero-shot and fine-tuned) for daily cross-sectional ranking of U.S. stocks, against LightGBM and classic factor signals.

Every day, models rank the 100 most liquid U.S. stocks by their next five-day return, and a long-only portfolio holds the top 20, net of transaction costs. The protocol uses point-in-time CRSP, Compustat and I/B/E/S data, a one-day execution lag, purged yearly walk-forward retraining over 2005-2021, a registry of every development trial for Deflated Sharpe ratios, and a final test on 2022-2025 run once with a configuration frozen beforehand.

The paper is in [paper/main.pdf](paper/main.pdf) (ACM ICAIF format), and the primary specification, frozen before any result, in [docs/SPEC.md](docs/SPEC.md).

## Main results

Net of 5 basis points per side, against the equal-weight tradable universe.

| Model | Rank IC, 2005-2021 | Information ratio, 2005-2021 | Rank IC, 2022-2025 | Net active return, 2022-2025 | Information ratio, 2022-2025 |
|---|---:|---:|---:|---:|---:|
| **Primary model** (grid-averaged LightGBM with accounting and analyst data + 4 simple signals) | 0.0335 | 0.64 | 0.0263 | +6.7%/yr | 0.60 |
| Short-term reversal (5 days) | 0.0193 | 0.54 | 0.0104 | +3.5%/yr | 0.24 |
| LightGBM, market features | 0.0263 | 0.33 | 0.0205 | +4.0%/yr | 0.30 |
| TimesFM 3, fine-tuned (10k steps) | 0.0228 | 0.39 | 0.0194 | +5.6%/yr | 0.49 |
| TimesFM 3, zero-shot | 0.0072 | -0.21 | 0.0053 | +0.7%/yr | 0.05 |

- Zero-shot TimesFM has no ranking skill, and its forecasts add nothing as features to LightGBM.
- Fine-tuning with a cross-sectional correlation loss gives TimesFM a real but modest signal, comparable to LightGBM on market data, that adds nothing to the primary model.
- The primary model reproduced its development performance on the untouched final test, but four years are not enough to make it statistically significant (active t = 1.2; development Deflated Sharpe ratio 0.68 over 70 trials).

## Repository layout

| Path | Content |
|---|---|
| `src/qef/` | Library: data cleaning, point-in-time universes and features, purged walk-forward, models, TimesFM wrappers, backtest and metrics |
| `scripts/` | Numbered pipeline steps, and Slurm job templates in `scripts/slurm/` |
| `tests/` | Unit tests, including tests against look-ahead leakage |
| `config/` | Primary specification, trial lists fixed before their results (`study_*.yaml`) and the frozen final-test configuration (`frozen.yaml`) |
| `experiments/` | Registry of the 70 development trials (`registry.jsonl`) and of the final test (`final_test.jsonl`) |
| `reports/` | Aggregate results of every run (no stock-level data) |
| `paper/` | LaTeX sources of the paper, with scripts that generate its tables and figure data from the results |

Main modules:

| Module | Role |
|---|---|
| `crsp.py`, `panel.py` | CRSP query and daily panel, delisting returns included |
| `universe.py`, `targets.py`, `features.py` | Point-in-time universes, executable forward returns, market features |
| `fundamentals.py` | Point-in-time Compustat and I/B/E/S features |
| `dataset.py`, `splits.py` | Stock-date table, purged walk-forward with a guarded final test |
| `models.py`, `combine.py` | Factor scores, Ridge, LightGBM (selected or grid-averaged), rank ensembles |
| `timesfm_zero_shot.py`, `finetune.py` | TimesFM 3 forecasts and fine-tuning building blocks |
| `portfolio.py`, `metrics.py`, `walkforward.py` | Backtest with rank buffer and costs, Rank IC, Newey-West, block bootstrap, Deflated Sharpe, trial registry |

## Setup

Python 3.11 or later. The exact versions behind the reported results are in [requirements-lock.txt](requirements-lock.txt).

```bash
pip install -e ".[dev]"          # main pipeline and tests
pip install -e ".[timesfm]"      # TimesFM 3 (PyTorch, GPU recommended)
pytest                           # unit and leakage tests; the fine-tuning tests need PyTorch
```

**Data.** The data are not included: they come from WRDS and require a subscription to CRSP (daily stock file, current format), point-in-time Compustat (`comp_pit`), I/B/E/S, the Fama-French factors and the CBOE VIX. WRDS credentials stay in `~/.pgpass`, never in the repository. All data go to `data/`, which is ignored by Git and can be a symlink to a larger disk (about 2.3 GB with predictions).

## Reproducing the results

The heavy steps were run with Slurm (`scripts/slurm/`, submitted from the repository root after `mkdir -p logs`). `QEF_PYTHON` and `QEF_TFM_PYTHON` select the interpreters of the main and TimesFM environments.

```bash
# 1. WRDS inventory (metadata only), then extraction of CRSP, factors, the S&P 500 and the VIX to data/raw/.
python scripts/00_wrds_inventory.py --username YOUR_WRDS_ID
python scripts/01_extract_wrds.py --username YOUR_WRDS_ID

# 2. Cleaning into data/clean/panel.parquet, with a coverage report.
sbatch scripts/slurm/cpu_job.sbatch scripts/02_clean_panel.py

# 3. Baselines on the development period.
sbatch scripts/slurm/cpu_job.sbatch scripts/10_run_baselines.py --panel data/clean/panel.parquet --raw data/raw

# 4. Zero-shot TimesFM 3 forecasts on GPU, then evaluation and diagnostics.
sbatch scripts/slurm/gpu_job.sbatch scripts/30_timesfm_predict.py --series returns --universe trade
sbatch scripts/slurm/cpu_job.sbatch scripts/10_run_baselines.py --panel data/clean/panel.parquet --raw data/raw \
    --timesfm data/predictions/timesfm/returns_trade --models timesfm
python scripts/31_timesfm_diagnostics.py

# 5. Point-in-time accounting and analyst data, then the enriched and grid-averaged models.
python scripts/03_extract_fundamentals.py --username YOUR_WRDS_ID
sbatch scripts/slurm/cpu_job.sbatch scripts/10_run_baselines.py --panel data/clean/panel.parquet --raw data/raw \
    --fundamentals data/raw --models sue,revision,lightgbm_fund,lightgbm_ana,lightgbm_fund_ana,lightgbm_fund_ana_avg
# Repeat the last model with --seed 1 and --seed 2 for the three-seed average.

# 6. Ensembles and portfolio settings, without retraining; each list is fixed before its results.
sbatch scripts/slurm/cpu_job.sbatch scripts/21_portfolio_study.py --study config/study_portfolio.yaml
sbatch scripts/slurm/cpu_job.sbatch scripts/21_portfolio_study.py --study config/study_signals.yaml \
    --reference reversal --out reports/study_signals
sbatch scripts/slurm/cpu_job.sbatch scripts/21_portfolio_study.py --study config/study_avg.yaml \
    --reference reversal --out reports/study_avg

# 7. Fine-tuning feasibility on data before 2005 only.
sbatch scripts/slurm/gpu_job.sbatch scripts/40_finetune_feasibility.py

# 8. Fine-tuning walk-forward over 2005-2021, retrained every 3 years (1,000 steps), then evaluation and ensembles.
sbatch scripts/slurm/gpu_job.sbatch scripts/41_finetune_walkforward.py
sbatch scripts/slurm/cpu_job.sbatch scripts/42_finetune_evaluate.py
sbatch scripts/slurm/cpu_job.sbatch scripts/21_portfolio_study.py --study config/study_finetune.yaml \
    --reference reversal --out reports/study_finetune

# 9. Same walk-forward with 10,000 steps, one seed per GPU (seeds 0 to 4), then evaluation once all seeds are done.
sbatch scripts/slurm/gpu_job.sbatch scripts/41_finetune_walkforward.py \
    --steps 10000 --seeds 0 --out data/predictions/timesfm/finetuned_10k --report reports/finetune_10k/seed0.json
sbatch scripts/slurm/cpu_job.sbatch scripts/42_finetune_evaluate.py --scores data/predictions/timesfm/finetuned_10k \
    --report reports/finetune_10k/seed*.json --seed-prefix tfm_ft10k --average-name tfm_finetuned_10k
sbatch scripts/slurm/cpu_job.sbatch scripts/21_portfolio_study.py --study config/study_finetune_10k.yaml \
    --reference reversal --out reports/study_finetune_10k

# 10. Robustness of the primary model; these variants cannot replace it.
sbatch scripts/slurm/cpu_job.sbatch scripts/21_portfolio_study.py --study config/study_robustness.yaml \
    --reference reversal --out reports/study_robustness

# 11. Final test 2022-2025, run once with config/frozen.yaml (see docs/SPEC.md).
bash scripts/slurm/final_test.sh

# 12. Tables and figure data of the paper, then compilation.
python paper/make_tables.py && python paper/figures/export_data.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

A 10,000-step fine-tuning run takes about 48 minutes on one NVIDIA RTX A4500; the whole TimesFM part needs roughly 50 GPU-hours. `scripts/20_compare_models.py` compares any set of models with paired bootstrap tests.

## Project rules

- **Data:** WRDS data and everything derived from it at the stock level stay in `data/` and are never committed or redistributed.
- **Trials:** every run is appended to `experiments/registry.jsonl` with a hash of the code, because it counts in the Deflated Sharpe correction.
- **Final test:** the dates from 2022 to 2025 were used once, with the frozen configuration; the code refuses them unless explicitly unlocked.
- **Leakage tests:** `tests/test_no_future_leakage.py` must pass before any run on real data.

## License

The code is under the [MIT License](LICENSE). The TimesFM 3 weights are not included and are under Google's `timesfm-non-commercial-license-v1.0`, which excludes revenue-generating use, including trading on the model's predictions; fine-tuned weights may not be distributed. WRDS data are subject to their own license.
