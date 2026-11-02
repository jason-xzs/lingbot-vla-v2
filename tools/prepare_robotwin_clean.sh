#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

PYTHON=${TRAIN_PYTHON:-python}
REVISION=3dc3b798668feb99ac61cc9086d84cbcc3d79186
RAW_ROOT=${ROBOTWIN_RAW_ROOT:-"$REPO_ROOT/data/raw_clean"}
DATA_ROOT=${CLEAN_DATA_ROOT:-"$REPO_ROOT/data/lerobot_clean"}
LOG_ROOT=data/lerobot_clean_logs
MANIFEST=assets/training_data/robotwin_clean.txt
NORM_STATS=assets/norm_stats/robotwin_clean.json
WORKERS=${ROBOTWIN_CONVERT_WORKERS:-4}

export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export SVT_LOG=1

mkdir -p "$LOG_ROOT"

printf 'STEP 1/3: Download pinned official clean archives only.\n'
"$PYTHON" -B -u scripts/download_hf_data.py \
    --repo_id TianxingChen/RoboTwin2.0 \
    --local_dir "$RAW_ROOT" \
    --allow_patterns 'dataset/*/aloha-agilex_clean_50.zip' \
    --revision "$REVISION" 2>&1 | tee "$LOG_ROOT/download.log"

printf 'STEP 2/3: Verify official SHA-256 and create LeRobot datasets.\n'
"$PYTHON" -B -u scripts/prepare_robotwin_clean.py \
    --input-root "$RAW_ROOT" \
    --output-root "$DATA_ROOT" \
    --revision "$REVISION" \
    --workers "$WORKERS" \
    --manifest "$MANIFEST" 2>&1 | tee "$LOG_ROOT/convert.log"

printf 'STEP 3/3: Compute clean-only normalization statistics.\n'
"$PYTHON" -B -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=1 \
    scripts/compute_norm_stats.py configs/vla/norm_compute/post_data.yaml \
    --data.data_name multi \
    --data.robot_name robotwin \
    --data.train_path "$MANIFEST" \
    --data.norm_path "$NORM_STATS" \
    --data.data_ratio_for_norm_compute 1.0 \
    --data.num_workers 8 \
    --train.micro_batch_size 48 2>&1 | tee "$LOG_ROOT/norm_stats.log"

printf 'CLEAN PREPARATION COMPLETE\nManifest: %s\nNorm stats: %s\n' "$MANIFEST" "$NORM_STATS"
