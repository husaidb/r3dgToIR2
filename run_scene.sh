#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python script/run_pipeline.py \
  --dataset_root /data/mwir/scenes --work_root /data/mwir/output \
  -d "$1" --temperature 30 --auto_mask "${@:2}"
