#!/bin/bash
# Run one project command on a GPU node. Submit from the repository root:
#   mkdir -p logs && sbatch scripts/slurm/gpu_job.sbatch scripts/30_timesfm_predict.py --series returns
# QEF_TFM_PYTHON selects an interpreter whose environment also has PyTorch and TimesFM
# (default: python). Set HF_HOME to keep the model weights on a large disk.
# Adapt the partition, GPU type and resources to your cluster; the reported runs used
# one NVIDIA RTX A4500 (20 GB) with 4 CPUs and 28 GB of memory.
#SBATCH --job-name=qef-gpu
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
echo "Node $(hostname), started $(date -Is)"
nvidia-smi -L
"${QEF_TFM_PYTHON:-python}" "$@"
echo "Finished $(date -Is)"
