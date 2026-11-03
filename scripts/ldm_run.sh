#!/usr/bin/env bash
set -e

WORLD_SIZE=2
CUDA_DEVICES="0,3"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# AUTO_RESUME=false
# RESUME_TIMESTAMP=""
# RESUME_PATH=""

# if [[ "$AUTO_RESUME" == "true" ]]; then
#     TIMESTAMP="$RESUME_TIMESTAMP"
# fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
torchrun --standalone --nproc_per_node="$WORLD_SIZE" \
    --module run.train.ldm_train \
    --schema configs/diffusionmodels/schema.yaml \
    --config configs/diffusionmodels/dit_tiny4.yaml \
    timestamp="'$TIMESTAMP'" \
    dist.world_size="$WORLD_SIZE"