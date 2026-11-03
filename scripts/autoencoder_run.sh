#!/usr/bin/env bash
set -e

WORLD_SIZE=3
CUDA_DEVICES="2,3,4"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# AUTO_RESUME=false
# RESUME_TIMESTAMP=""
# RESUME_PATH=""

# if [[ "$AUTO_RESUME" == "true" ]]; then
#     TIMESTAMP="$RESUME_TIMESTAMP"
# fi


CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
torchrun --standalone --nproc_per_node="$WORLD_SIZE" \
    --module run.train.autoencoder_train \
    --schema configs/autoencoders/schema.yaml \
    --config configs/autoencoders/vae_mmchq_f8c4_betae-1.yaml \
    timestamp="'$TIMESTAMP'" \
    dist.world_size="$WORLD_SIZE"