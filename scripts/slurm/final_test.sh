#!/bin/bash
# Final test 2022-2025, run once with the configuration of config/frozen.yaml.
# Run from the repository root of a Slurm cluster: bash scripts/slurm/final_test.sh
# GPU_OPTS sets the GPU request; the reported run used
#   --gres=gpu:rtx_a4500:1 --cpus-per-task=4 --mem=28G
# and excluded one node whose GPU corrupted the model weights.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

GPU="${GPU_OPTS:---gres=gpu:1 --cpus-per-task=4 --mem=28G}"
BASE="scripts/10_run_baselines.py --final-test --panel data/clean/panel.parquet --raw data/raw --threads 32"

zero=$(sbatch --parsable --job-name=qef-final-tfm0 $GPU --time=03:00:00 scripts/slurm/gpu_job.sbatch \
    scripts/30_timesfm_predict.py --series returns --universe trade --first-year 2022 --last-year 2025 --allow-holdout \
    --out data/predictions/timesfm_final)
ft=""
for s in 0 1 2 3 4; do
    id=$(sbatch --parsable --job-name=qef-final-ft$s $GPU --time=06:00:00 scripts/slurm/gpu_job.sbatch \
        scripts/41_finetune_walkforward.py --final-test --steps 10000 --seeds $s \
        --out data/predictions/timesfm/finetuned_10k_final --report reports/final/finetune_10k/seed$s.json)
    ft="$ft:$id"
done
tab=$(sbatch --parsable --job-name=qef-final-tab scripts/slurm/cpu_job.sbatch $BASE --fundamentals data/raw \
    --models reversal,momentum,revision,sue,lightgbm,lightgbm_fund_ana_avg)
seed1=$(sbatch --parsable --job-name=qef-final-s1 scripts/slurm/cpu_job.sbatch $BASE --fundamentals data/raw \
    --models lightgbm_fund_ana_avg --seed 1)
seed2=$(sbatch --parsable --job-name=qef-final-s2 scripts/slurm/cpu_job.sbatch $BASE --fundamentals data/raw \
    --models lightgbm_fund_ana_avg --seed 2)
tfm0=$(sbatch --parsable --job-name=qef-final-tfm0e --dependency=afterok:$zero scripts/slurm/cpu_job.sbatch $BASE \
    --timesfm data/predictions/timesfm_final/returns_trade --models timesfm)
reports=""; for s in 0 1 2 3 4; do reports="$reports reports/final/finetune_10k/seed$s.json"; done
fte=$(sbatch --parsable --job-name=qef-final-fte --dependency=afterok$ft scripts/slurm/cpu_job.sbatch \
    scripts/42_finetune_evaluate.py --final-test --scores data/predictions/timesfm/finetuned_10k_final \
    --report $reports --seed-prefix tfm_ft10k --average-name tfm_finetuned_10k)
study=$(sbatch --parsable --job-name=qef-final-ens --dependency=afterok:$tab:$seed1:$seed2:$fte scripts/slurm/cpu_job.sbatch \
    scripts/21_portfolio_study.py --study config/final_test.yaml --reference reversal --scores data/predictions/final \
    --series-out data/predictions/final/study --out reports/final/study --registry experiments/final_test.jsonl)
report=$(sbatch --parsable --job-name=qef-final-report --dependency=afterok:$study:$tfm0 scripts/slurm/cpu_job.sbatch \
    scripts/50_final_report.py)
echo "zero=$zero ft=$ft tab=$tab seed1=$seed1 seed2=$seed2 tfm0=$tfm0 fte=$fte study=$study report=$report"
