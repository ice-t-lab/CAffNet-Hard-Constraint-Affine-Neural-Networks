#!/bin/bash
#SBATCH --job-name=pwc
#SBATCH --partition=gpu               # batch GPU partition (interactive is gpu-interactive)
#SBATCH --gres=gpu:v100-sxm2:1         # or gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=1-8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

SCENARIO="${1:-pwc}"
DIR_NAME="${2:-${SCENARIO}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-test}}}"

SEED="${SLURM_ARRAY_TASK_ID:-1}"

# Make sure logs/ exists for %x %A %a output paths
mkdir -p logs

echo "[INFO] SCENARIO=$SCENARIO  DIR_NAME=$DIR_NAME  SEED=$SEED"
echo "[INFO] PWD=$(pwd)"

# Run inside container; bind current project directory
apptainer exec --nv \
  --bind "$(pwd):/workspace" \
  --pwd /workspace \
  env.sif \
  bash -lc "./run.sh \"$SCENARIO\" \"$DIR_NAME\" \"$SEED\""
