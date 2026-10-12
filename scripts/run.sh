# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun --standalone --nproc_per_node=4 main.py \
#   --schema ./configs/autoencoders/schema.yaml \
#   --config ./configs/autoencoders/vae_default.yaml

CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nproc_per_node=1 main.py \
    --schema ./configs/autoencoders/schema.yaml \
    --config ./configs/autoencoders/vae_mnist_1gputest.yaml

# CUDA_VISIBLE_DEVICES=0,1, \
# torchrun --standalone --nproc_per_node=2 main.py \
#     --schema ./configs/autoencoders/schema.yaml \
#     --config ./configs/autoencoders/vae_mnist_z32_init64.yaml