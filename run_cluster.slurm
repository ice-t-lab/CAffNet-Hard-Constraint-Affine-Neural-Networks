#!/bin/bash
#SBATCH --job-name=CAffNet
#SBATCH --partition=gpu               # batch GPU partition (interactive is gpu-interactive)
#SBATCH --gres=gpu:v100-sxm2:1         # or gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=1-8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: sbatch [options] $0 <scenario> <dir_name> <method> [method ...]" >&2
  exit 2
fi

SCENARIO=$1
DIR_NAME=$2
METHODS=("${@:3}")

for method in "${METHODS[@]}"; do
  if [[ -z "$method" ]]; then
    echo "Method names cannot be empty." >&2
    exit 2
  fi
done

SEED="${SLURM_ARRAY_TASK_ID:-1}"

# Make sure logs/ exists for %x %A %a output paths
mkdir -p logs

echo "[INFO] SCENARIO=$SCENARIO  DIR_NAME=$DIR_NAME  SEED=$SEED  METHODS=${METHODS[*]}"
echo "[INFO] PWD=$(pwd)"

# Run inside container; bind current project directory
run_args=("$SCENARIO" "$DIR_NAME" "$SEED" "${METHODS[@]}")

apptainer exec --nv \
  --bind "$(pwd):/workspace" \
  --pwd /workspace \
  env.sif \
  ./run.sh "${run_args[@]}"
