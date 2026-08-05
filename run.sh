#!/bin/bash
set -euo pipefail

case=$1
dir_name=$2
seed=$3

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# python3 -m "src.scenarios.${case}.main" --dir_name "${dir_name}" --seed "${seed}" --method "Optimizer (IPOPT)"
python3 -m "src.scenarios.${case}.main" --dir_name "${dir_name}" --seed "${seed}" --method NN
# python3 -m "src.scenarios.${case}.main" --dir_name "${dir_name}" --seed "${seed}" --method HardNet
# python3 -m "src.scenarios.${case}.main" --dir_name "${dir_name}" --seed "${seed}" --method CAffNet-FF
# python3 -m "src.scenarios.${case}.main" --dir_name "${dir_name}" --seed "${seed}" --method "CAffNet-FF (Lite)"
# python3 -m "src.scenarios.${case}.main" --dir_name "${dir_name}" --seed "${seed}" --method CAffNet-TF
