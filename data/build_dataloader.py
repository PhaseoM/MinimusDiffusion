import torch
import torchvision
import data.datasets as datasets
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2, InterpolationMode
from torch.utils.data.distributed import DistributedSampler


def build_mnist_dataloader(conf, rank):
    img_size = tuple(conf.data.img_size)
    is_persistent = conf.dist.num_workers > 0

    transform_normal = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Resize(img_size, antialias=True),
    ])
    transform_augment = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Resize(img_size, antialias=True),
    ])
    train_data = torchvision.datasets.MNIST(
        root=conf.path.dataset_path,
        train=True,
        download=True,
        transform=transform_augment,
    )
    test_data = torchvision.datasets.MNIST(
        root=conf.path.dataset_path,
        train=False,
        download=True,
        transform=transform_normal,
    )

    val_size = int(len(train_data) * conf.data.val_ratio)
    train_size = len(train_data) - val_size
    train_data, val_data = random_split(
        train_data,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(conf.seed),
    )
    train_sampler = DistributedSampler(
        train_data,
        rank=rank,
        num_replicas=conf.dist.world_size,
        shuffle=True,
    )
    val_sampler = DistributedSampler(
        val_data,
        rank=rank,
        num_replicas=conf.dist.world_size,
        shuffle=False,
    )

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=conf.train.batch_size,
        sampler=train_sampler,
        num_workers=conf.dist.num_workers,
        pin_memory=True,
        persistent_workers=is_persistent,
    )
    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=conf.val.batch_size,
        sampler=val_sampler,
        num_workers=conf.dist.num_workers,
        pin_memory=True,
        persistent_workers=is_persistent,
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=conf.train.batch_size,
        shuffle=False,
        num_workers=conf.dist.num_workers,
        pin_memory=True,
        persistent_workers=is_persistent,
    )
    return train_dataloader, val_dataloader, test_dataloader, train_sampler


def build_mm_celebahq_dataloader(conf, rank):
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
