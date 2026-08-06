#!/bin/bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <scenario> <dir_name> <seed> <method> [method ...]" >&2
  exit 2
fi

case=$1
dir_name=$2
seed=$3
methods=("${@:4}")

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

for method in "${methods[@]}"; do
  if [[ -z "$method" ]]; then
    echo "Method names cannot be empty." >&2
    exit 2
  fi
done

for method in "${methods[@]}"; do
  python3 -m "src.scenarios.${case}.main" --dir_name "$dir_name" --seed "$seed" --method "$method"
done
