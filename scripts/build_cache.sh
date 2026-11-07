CUDA_DEVICES="2"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
python -m data.build_cache.build_ldm_cache \
    --config configs/config.yaml
