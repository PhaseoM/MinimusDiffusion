import torch
import torchvision
from typing import Optional, List, Type, Tuple, Dict
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Sampler, random_split
from torchvision.transforms import v2, InterpolationMode
from torch.utils.data.distributed import DistributedSampler
from data.build_datasets import MMCelebAHQDataset, Cache_MMCelebAHQDataset
from utils.config import Config, DataConfig


def _build_dataloaders_from_dataset(data_conf: DataConfig, dataset: Dataset):
    all_size = len(dataset)
    val_size = int(all_size * data_conf.val.dataset_ratio)
    test_size = int(all_size * data_conf.test.dataset_ratio)
    train_size = all_size - val_size - test_size
    train_data, val_data, test_data = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(data_conf.seed),
    )
    train_sampler = DistributedSampler(train_data, shuffle=data_conf.train.shuffle)
    val_sampler = DistributedSampler(val_data, shuffle=data_conf.val.shuffle)

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=data_conf.batch_size,
        sampler=train_sampler,
        num_workers=data_conf.train.num_workers,
        pin_memory=data_conf.train.pin_memory,
        persistent_workers=data_conf.train.num_workers > 0,
    )
    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=data_conf.batch_size,
        sampler=val_sampler,
        num_workers=data_conf.val.num_workers,
        pin_memory=data_conf.val.pin_memory,
        persistent_workers=data_conf.val.num_workers > 0,
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=data_conf.batch_size,
        shuffle=data_conf.test.shuffle,
        num_workers=data_conf.test.num_workers,
        pin_memory=data_conf.test.pin_memory,
        persistent_workers=data_conf.test.num_workers > 0,
    )
    return train_dataloader, val_dataloader, test_dataloader, train_sampler


def _build_mmcelebahq_dataset(data_conf: DataConfig, caption_mode: str = "random"):
    img_size = tuple(data_conf.dataset.img_size)

    img_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(img_size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    if not data_conf.dataset.use_cache:
        dataset = MMCelebAHQDataset(
            root=data_conf.dataset.dataset_path,
            img_transform=img_transform,
            txt_transform=None,
            caption_mode=caption_mode,
        )
    else:
        dataset = Cache_MMCelebAHQDataset(
            root=data_conf.dataset.cache_path,
            caption_mode=caption_mode,
        )

    return dataset


def build_dataloaders(data_conf: DataConfig, caption_mode: str = "random"):
    dataset_name = data_conf.dataset.name

    if dataset_name == "mm-celeba-hq":
        dataset = _build_mmcelebahq_dataset(data_conf, caption_mode)
    else:
        raise FileNotFoundError("Do not exist that dataset_name {dataset_name}")

    return _build_dataloaders_from_dataset(data_conf, dataset)
