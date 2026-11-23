#!/usr/bin/env python
"""Write the paper's tables as LaTeX fragments, straight from the saved results.

Development figures use the 70 trials of experiments/registry.jsonl for the Deflated
Sharpe ratio. The daily series in data/predictions are needed, so this runs after the experiments. Final-test figures come from reports/final, written by scripts/50_final_report.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qef.metrics import block_bootstrap_ci, deflated_sharpe  # noqa: E402
from qef.walkforward import trial_sharpe_variance  # noqa: E402

PRED, REP = ROOT / "data" / "predictions", ROOT / "reports"
OUT = Path(__file__).resolve().parent / "tables"
SETTINGS = yaml.safe_load((ROOT / "config" / "spec.yaml").read_text())["evaluation"]
N_DEV_TRIALS = 70

# label, daily series path, summary path
DEV_ROWS = [
    ("Simple signals", None, None),
    ("Momentum (12--1)", "momentum", "baselines/momentum"),
    ("Short-term reversal (5 d)", "reversal", "baselines/reversal"),
    ("Earnings surprise (SUE)", "sue", "baselines/sue"),
    ("Analyst revisions (3 m)", "revision", "baselines/revision"),
    ("Tabular models, market features only", None, None),
    ("LightGBM, selected", "lightgbm", "baselines/lightgbm"),
    ("LightGBM, grid average", "lightgbm_avg", "baselines/lightgbm_avg"),
    ("Tabular models with accounting (Acc) and analyst (Ana) features", None, None),
    ("LightGBM + Acc + Ana, selected", "lightgbm_fund_ana", "baselines/lightgbm_fund_ana"),
    ("LightGBM + Acc, grid average", "lightgbm_fund_avg", "baselines/lightgbm_fund_avg"),
    ("LightGBM + Ana, grid average", "lightgbm_ana_avg", "baselines/lightgbm_ana_avg"),
    ("LightGBM + Acc + Ana, grid average", "lightgbm_fund_ana_avg", "baselines/lightgbm_fund_ana_avg"),
    ("\\quad same, 3 seeds", "study/avg3_s1_x40", "study_avg/avg3_s1_x40"),
    ("\\quad + 4 simple signals (\\textbf{primary model})", "study/avg3_signals_s1_x40", "study_avg/avg3_signals_s1_x40"),
    ("TimesFM 3", None, None),
    ("Zero-shot, returns input", "timesfm_returns_trade", "baselines/timesfm_returns_trade"),
    ("LightGBM + zero-shot features, grid average", "lightgbm_tfm_avg_returns_train", "baselines/lightgbm_tfm_avg_returns_train"),
    ("LightGBM + Acc + Ana + zero-shot features, grid avg.", "lightgbm_all_avg_returns_train", "baselines/lightgbm_all_avg_returns_train"),
    ("Fine-tuned, 1k steps", "tfm_finetuned", "baselines/tfm_finetuned"),
    ("Fine-tuned, 10k steps", "tfm_finetuned_10k", "baselines/tfm_finetuned_10k"),
    ("Fine-tuned 10k + reversal", "study/ft10k_reversal_s1_x40", "study_finetune_10k/ft10k_reversal_s1_x40"),
    ("Primary model + fine-tuned 10k", "study/avg3_signals_ft10k_s1_x40", "study_finetune_10k/avg3_signals_ft10k_s1_x40"),
]

DEV_GAPS = [
    ("Grid average vs.\\ selection, market features", "lightgbm_avg", "lightgbm"),
    ("Grid average vs.\\ selection, + Acc + Ana", "lightgbm_fund_ana_avg", "lightgbm_fund_ana"),
    ("+ Acc + Ana (grid average)", "lightgbm_fund_ana_avg", "lightgbm_avg"),
    ("+ 4 simple signals (3 seeds)", "study/avg3_signals_s1_x40", "study/avg3_s1_x40"),
    ("Primary model vs.\\ reversal", "study/avg3_signals_s1_x40", "reversal"),
    ("+ zero-shot features, with Acc + Ana", "lightgbm_all_avg_returns_train", "lightgbm_fund_ana_avg"),
    ("Fine-tuned 10k vs.\\ zero-shot", "tfm_finetuned_10k", "timesfm_returns_trade"),
    ("Fine-tuned 10k vs.\\ 1k", "tfm_finetuned_10k", "tfm_finetuned"),
    ("Fine-tuned 10k vs.\\ primary model", "tfm_finetuned_10k", "study/avg3_signals_s1_x40"),
    ("Fine-tuned 10k + reversal vs.\\ primary model", "study/ft10k_reversal_s1_x40", "study/avg3_signals_s1_x40"),
    ("Primary model + fine-tuned 10k vs.\\ primary model", "study/avg3_signals_ft10k_s1_x40", "study/avg3_signals_s1_x40"),
]

FINAL_LABELS = {
    "avg3_signals_s1_x40": "Primary model",
    "momentum": "Momentum (12--1)",
    "tfm_finetuned_10k": "TimesFM fine-tuned 10k",
    "ft10k_reversal_s1_x40": "TimesFM fine-tuned 10k + reversal",
    "lightgbm": "LightGBM (market features)",
    "reversal": "Short-term reversal (5 d)",
    "timesfm_returns_trade": "TimesFM zero-shot",
}

ROBUST_LABELS = {
    "no_reversal_s1_x40": "Without reversal",
    "no_momentum_s1_x40": "Without momentum",
    "no_revision_s1_x40": "Without revisions",
    "no_sue_s1_x40": "Without SUE",
    "seed0_signals_s1_x40": "LightGBM seed 0 only",
    "seed1_signals_s1_x40": "LightGBM seed 1 only",
    "seed2_signals_s1_x40": "LightGBM seed 2 only",
    "avg3_signals_s1_x30": "Exit rank 30",
    "avg3_signals_s1_x50": "Exit rank 50",
    "avg3_signals_s1_x30_n15": "15 positions, exit 30, cap 10\\%",
    "avg3_signals_s1_x60_n30": "30 positions, exit 60, cap 5\\%",
}


def signed(value: float, digits: int) -> str:
    text = f"{value:+.{digits}f}"
    if float(text) == 0:
        return f"${0:.{digits}f}$"
    return f"${text}$".replace("+", "{+}")


def plain(value: float, digits: int) -> str:
    return f"${value:.{digits}f}$"


def thousands(value: float) -> str:
    return f"{value:,.0f}".replace(",", "{,}")


def active(path: str, column: str = "net_central") -> pd.Series:
    frame = pd.read_parquet(PRED / f"{path}_daily.parquet")
    return (frame[column] - frame["benchmark"]).dropna()


def gap_cells(a: str, b: str, prefix: Path = PRED) -> list[str]:
    cells = []
    for column, scale, digits in (("rank_ic", 1, 4), ("net_central", 252 * 100, 1)):
        left = pd.read_parquet(prefix / f"{a}_daily.parquet")[column]
        right = pd.read_parquet(prefix / f"{b}_daily.parquet")[column]
        gap = (left - right).dropna()
        low, high = block_bootstrap_ci(gap, block=SETTINGS["bootstrap_block"], reps=SETTINGS["bootstrap_reps"], seed=SETTINGS["bootstrap_seed"])
        strong = low > 0 or high < 0
        text = f"{signed(gap.mean() * scale, digits)} [{signed(low * scale, digits)}, {signed(high * scale, digits)}]"
        cells.append(f"{{\\boldmath\\bfseries {text}}}" if strong else text)
    return cells


def write_dev_rows(path: Path, rows: list[tuple[str | None, str]]) -> None:
    """Write (series, line) rows; a None series marks a group heading."""
    lines = []
    for series, line in rows:
        if series is None:
            # A file read inside a tabular must not start with a rule, so the first group has none.
            line = ("\\midrule" if lines else "") + line
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")


def development() -> None:
    variance = trial_sharpe_variance([PRED, PRED / "study"], "central")
    rows = []
    for label, series, summary_path in DEV_ROWS:
        if series is None:
            rows.append((None, f"\\multicolumn{{8}}{{l}}{{\\textit{{{label}}}}} \\\\"))
            continue
        summary = json.loads((REP / f"{summary_path}.json").read_text())
        book, prudent = summary["portfolio"]["central"], summary["portfolio"]["prudent"]
        dsr = deflated_sharpe(active(series), N_DEV_TRIALS, variance)
        cells = [
            label,
            plain(summary["signal"]["rank_ic"], 4),
            plain(summary["signal"]["rank_ic_t"], 1),
            signed(book["ann_active"] * 100, 1),
            signed(book["information_ratio"], 2),
            signed(prudent["information_ratio"], 2),
            thousands(book["annual_turnover"] * 100),
            plain(dsr, 2),
        ]
        rows.append((series, " & ".join(cells) + " \\\\"))
    write_dev_rows(OUT / "dev_results.tex", rows)

    lines = [" & ".join([label, *gap_cells(a, b)]) + " \\\\" for label, a, b in DEV_GAPS]
    (OUT / "dev_gaps.tex").write_text("\n".join(lines) + "\n")
    print(f"development: trial variance {variance:.6f}")


def final() -> None:
    table = pd.read_csv(REP / "final" / "summary.tsv", sep="\t", index_col=0)
    lines = []
    for name, label in FINAL_LABELS.items():
        row = table.loc[name]
        cells = [
            label,
            f"{plain(row['rank_ic'], 4)} ({plain(row['ic_t'], 1)})",
            plain(row["return"] * 100, 1),
            signed(row["active"] * 100, 1),
            f"{signed(row['IR'], 2)} ({plain(row['active_t'], 1)})",
            signed(row["IR_10bp"], 2),
            plain(row["P(IR>0)"], 2),
            signed(row["dev_IR"], 2),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    (OUT / "final_results.tex").write_text("\n".join(lines) + "\n")

    gaps = pd.read_csv(REP / "final" / "gaps.tsv", sep="\t")
    lines = []
    for (a, b), rows in gaps.groupby(["a", "b"], sort=False):
        cells = [f"{FINAL_LABELS[a]} vs.\\ {FINAL_LABELS[b]}"]
        for measure, scale, digits in (("rank_ic", 1, 4), ("net_return", 100, 1)):
            row = rows[rows["measure"] == measure].iloc[0]
            strong = row["low"] > 0 or row["high"] < 0
            text = f"{signed(row['gap'] * scale, digits)} [{signed(row['low'] * scale, digits)}, {signed(row['high'] * scale, digits)}]"
            cells.append(f"{{\\boldmath\\bfseries {text}}}" if strong else text)
        lines.append(" & ".join(cells) + " \\\\")
    (OUT / "final_gaps.tex").write_text("\n".join(lines) + "\n")

    years = pd.read_csv(REP / "final" / "active_by_year.tsv", sep="\t", index_col=0)
    lines = []
    for name, label in FINAL_LABELS.items():
        lines.append(" & ".join([label, *[signed(years.loc[y, name] * 100, 1) for y in years.index]]) + " \\\\")
    (OUT / "final_by_year.tex").write_text("\n".join(lines) + "\n")


def robustness() -> None:
    table = pd.read_csv(REP / "study_robustness" / "summary.tsv", sep="\t", index_col=0)
    candidate = json.loads((REP / "study_avg" / "avg3_signals_s1_x40.json").read_text())
    book = candidate["portfolio"]["central"]
    lines = [" & ".join([
        "Primary (20 positions, exit 40)", plain(candidate["signal"]["rank_ic"], 4), signed(book["ann_active"] * 100, 1),
        signed(book["information_ratio"], 2), signed(candidate["portfolio"]["prudent"]["information_ratio"], 2), thousands(book["annual_turnover"] * 100),
    ]) + " \\\\", "\\midrule"]
    for name, label in ROBUST_LABELS.items():
        row = table.loc[name]
        lines.append(" & ".join([
            label, plain(row["rank_ic"], 4), signed(row["active"] * 100, 1), signed(row["IR"], 2), signed(row["IR_prudent"], 2), thousands(row["turnover"] * 100),
        ]) + " \\\\")
    (OUT / "robustness.tex").write_text("\n".join(lines) + "\n")
    print(f"robustness: median IR {table.loc[list(ROBUST_LABELS), 'IR'].median():.3f}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    development()
    final()
    robustness()
    for path in sorted(OUT.glob("*.tex")):
        print(f"--- {path.name}\n{path.read_text()}")


if __name__ == "__main__":
    main()
