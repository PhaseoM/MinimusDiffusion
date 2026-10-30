#!/usr/bin/env bash
set -e

WORLD_SIZE=2
CUDA_DEVICES="3,4"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# AUTO_RESUME=false
# RESUME_TIMESTAMP=""
# RESUME_PATH=""

# if [[ "$AUTO_RESUME" == "true" ]]; then
#     TIMESTAMP="$RESUME_TIMESTAMP"
# fi

# CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
# torchrun --standalone --nproc_per_node="$WORLD_SIZE" main.py \
#     --schema ./configs/autoencoders/schema.yaml \
#     --config ./configs/autoencoders/vae_mmchq_z32_init64.yaml \
#     timestamp="'$TIMESTAMP'" \
#     dist.world_size="$WORLD_SIZE"

# CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
# torchrun --standalone --nproc_per_node="$WORLD_SIZE" main.py \
#     --schema ./configs/autoencoders/schema.yaml \
#     --config ./configs/autoencoders/vae_mmchq_f8c8.yaml \
#     timestamp="'$TIMESTAMP'" \
#     dist.world_size="$WORLD_SIZE"

# TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
# torchrun --standalone --nproc_per_node="$WORLD_SIZE" main.py \
#     --schema ./configs/autoencoders/schema.yaml \
#     --config ./configs/autoencoders/vae_mmchq_f8c8_betae-2.yaml \
#     timestamp="'$TIMESTAMP'" \
#     dist.world_size="$WORLD_SIZE"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
torchrun --standalone --nproc_per_node="$WORLD_SIZE" main.py \
    --schema ./configs/autoencoders/schema.yaml \
    --config ./configs/autoencoders/vae_mmchq_f8c8_betae-1.yaml \
    timestamp="'$TIMESTAMP'" \
    dist.world_size="$WORLD_SIZE"