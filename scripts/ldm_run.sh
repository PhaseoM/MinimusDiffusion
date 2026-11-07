#!/usr/bin/env bash
set -e

WORLD_SIZE=1
CUDA_DEVICES="5"

# test
# CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
# torchrun --standalone --nproc_per_node="$WORLD_SIZE" \
#     --module run.train.ldm_train \
#     --config configs/test.yaml

# resume test
# CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
# torchrun --standalone --nproc_per_node="$WORLD_SIZE" \
#     --module run.train.ldm_train \
#     --config configs/test.yaml \
#     auto_resume=true \
#     resume_path="./results/TEST_20261107_230103/ckpts/ckpt_epoch_0004.pth" \
#     exp_name="TEST_20261107_230103"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
torchrun --standalone --nproc_per_node="$WORLD_SIZE" \
    --module run.train.ldm_train \
    --config configs/dit_base4_vae_f8c8.yaml

