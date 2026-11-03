CUDA_DEVICES="2"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
python -m data.build_cache.build_ldm_cache \
    --schema configs/diffusionmodels/schema.yaml \
    --config configs/diffusionmodels/dit_medium4.yaml
