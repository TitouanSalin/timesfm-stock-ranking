#!/bin/bash
# Run one project command on a CPU node. Submit from the repository root:
#   mkdir -p logs && sbatch scripts/slurm/cpu_job.sbatch scripts/02_clean_panel.py
# QEF_PYTHON selects the interpreter of the main environment (default: python).
# Adapt the partition and resources to your cluster.
#SBATCH --job-name=qef
#SBATCH --partition=cpunodes
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
echo "Node $(hostname), started $(date -Is)"
"${QEF_PYTHON:-python}" "$@"
echo "Finished $(date -Is)"
