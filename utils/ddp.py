import os
import torch
import torch.distributed as dist


def new_ddp(conf):
    if not torch.cuda.is_available():
        raise RuntimeError("DDP without CUDA")

    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    if world_size != conf.dist.world_size:
        raise RuntimeError(f"need {conf.dist.world_size} GPU,but world_size={world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return rank, local_rank, world_size, device


def delete_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()
