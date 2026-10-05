CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
main.py --config unet_layer6_w512_c4.yaml \
DATA.WORLD_SIZE=4
