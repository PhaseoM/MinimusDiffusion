import torch
import data.datasets as datasets
from config import Config
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2, InterpolationMode
from torch.utils.data.distributed import DistributedSampler


def _mask_to_long(mask):
    return (mask.squeeze(0) > 0).long()


def create_isic2016_dataloader(conf: Config, rank):
    image_size = tuple(conf.DATA.IMAGE_SIZE)
    transform_normal = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Resize(image_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
    ])
    target_transform_normal = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=False),
        v2.Resize(image_size, interpolation=InterpolationMode.NEAREST),
        v2.Lambda(_mask_to_long),
    ])
    transform_augment = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Resize(image_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
    ])
    target_transform_augment = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=False),
        v2.Resize(image_size, interpolation=InterpolationMode.NEAREST),
        v2.Lambda(_mask_to_long),
    ])

    train_data = datasets.ISIC2016Dataset(
        image_dir=conf.PATH.TRAIN_DATA,
        mask_dir=conf.PATH.TRAIN_GT,
        transform=transform_augment,
        target_transform=target_transform_augment,
    )
    test_data = datasets.ISIC2016Dataset(
        image_dir=conf.PATH.TEST_DATA,
        mask_dir=conf.PATH.TEST_GT,
        transform=transform_normal,
        target_transform=target_transform_normal,
    )
    train_sampler = DistributedSampler(
        train_data,
        rank=rank,
        num_replicas=conf.DATA.WORLD_SIZE,
        shuffle=True,
    )
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=conf.DATA.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=conf.DATA.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=conf.DATA.NUM_WORKERS > 0,
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=conf.DATA.BATCH_SIZE,
        # sampler=test_sampler,
        num_workers=conf.DATA.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=conf.DATA.NUM_WORKERS > 0,
    )
    return train_dataloader, test_dataloader, train_sampler


def create_isic2016_dataloader_none(conf: Config):
    image_size = tuple(conf.DATA.IMAGE_SIZE)
    transform_normal = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Resize(image_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
    ])
    target_transform_normal = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=False),
        v2.Resize(image_size, interpolation=InterpolationMode.NEAREST),
        v2.Lambda(_mask_to_long),
    ])
    transform_augment = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Resize(image_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
    ])
    target_transform_augment = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=False),
        v2.Resize(image_size, interpolation=InterpolationMode.NEAREST),
        v2.Lambda(_mask_to_long),
    ])
    train_data = datasets.ISIC2016Dataset(
        image_dir=conf.PATH.TRAIN_DATA,
        mask_dir=conf.PATH.TRAIN_GT,
        transform=transform_augment,
        target_transform=target_transform_augment,
    )
    test_data = datasets.ISIC2016Dataset(
        image_dir=conf.PATH.TEST_DATA,
        mask_dir=conf.PATH.TEST_GT,
        transform=transform_normal,
        target_transform=target_transform_normal,
    )
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=conf.DATA.BATCH_SIZE,
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=conf.DATA.BATCH_SIZE,
    )
    return train_dataloader, test_dataloader
