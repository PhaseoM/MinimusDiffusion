import torch
import torchvision
from typing import Optional, List, Type, Tuple, Dict
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Sampler, random_split
from torchvision.transforms import v2, InterpolationMode
from torch.utils.data.distributed import DistributedSampler
from data.build_datasets import MMCelebAHQDataset, Cache_MMCelebAHQDataset


def _build_dataloaders_from_dataset(config, rank: int, dataset: Dataset):
    is_persistent = config.dist.num_workers > 0
    all_size = len(dataset)
    val_size = int(all_size * config.val.dataset_ratio)
    test_size = int(all_size * config.test.dataset_ratio)
    train_size = all_size - val_size - test_size
    train_data, val_data, test_data = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    train_sampler = DistributedSampler(
        train_data,
        rank=rank,
        num_replicas=config.dist.world_size,
        shuffle=True,
    )
    val_sampler = DistributedSampler(
        val_data,
        rank=rank,
        num_replicas=config.dist.world_size,
        shuffle=False,
    )
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=config.train.batch_size,
        sampler=train_sampler,
        num_workers=config.dist.num_workers,
        pin_memory=True,
        persistent_workers=is_persistent,
    )
    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=config.val.batch_size,
        sampler=val_sampler,
        num_workers=config.dist.num_workers,
        pin_memory=True,
        persistent_workers=is_persistent,
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.dist.num_workers,
        pin_memory=True,
        persistent_workers=is_persistent,
    )
    return train_dataloader, val_dataloader, test_dataloader, train_sampler


def _build_mmcelebahq_dataset(config, rank: int, caption_mode: str = "random"):
    img_size = tuple(config.data.img_size)

    img_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(img_size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    if not config.data.use_cache:
        dataset = MMCelebAHQDataset(
            root=config.path.dataset_path,
            img_transform=img_transform,
            txt_transform=None,
            caption_mode=caption_mode,
        )
    else:
        dataset = Cache_MMCelebAHQDataset(
            root=config.path.cache_path,
            caption_mode=caption_mode,
        )

    return dataset


def build_dataloaders(config, rank: int, caption_mode: str = "random"):
    dataset_name = config.data.dataset_name

    if dataset_name == "mm-celeba-hq":
        dataset = _build_mmcelebahq_dataset(config, rank, caption_mode)
    else:
        raise FileNotFoundError("Do not exist that dataset_name {dataset_name}")

    return _build_dataloaders_from_dataset(config, rank, dataset)
