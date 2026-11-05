import torch
import random
from pathlib import Path
from PIL import Image

import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image


class MMCelebAHQDataset(Dataset):
    """
    Pair images/<id>.jpg with captions/<id>.txt under ``root``.

    Args:
        - caption_mode: "random"/"first" for the sample_mode
    Returns:
        - an image and a caption.
    Captions are read lazily.
    """

    def __init__(self, root, img_transform=None, txt_transform=None, caption_mode="random"):
        if caption_mode not in ("random", "first"):
            raise ValueError(f"caption_mode must be 'random' or 'first', got {caption_mode!r}")

        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.caption_dir = self.root / "captions"
        self.img_transform = img_transform
        self.txt_transform = txt_transform
        self.caption_mode = caption_mode

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.caption_dir.is_dir():
            raise FileNotFoundError(f"Caption directory does not exist: {self.caption_dir}")

        self.images = sorted(
            (path for path in self.image_dir.glob("*.jpg") if path.is_file()),
            key=lambda path: int(path.stem),
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        caption_path = self.caption_dir / f"{image_path.stem}.txt"

        with Image.open(image_path) as image:
            image_entity = image.convert("RGB")

        captions = [line.strip() for line in caption_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        caption = random.choice(captions) if self.caption_mode == "random" else captions[0]

        if self.img_transform is not None:
            image_entity = self.img_transform(image_entity)
        if self.txt_transform is not None:
            caption = self.txt_transform(caption)

        return image_entity, caption


class Cache_MMCelebAHQDataset(Dataset):
    """
    Cache of the pair images/<id>.jpg with captions/<id>.txt under ``root``.

    Args:
        - caption_mode: "random"/"first" for the sample_mode
    Returns:
        - cache/feature of an image and a caption.
    """

    def __init__(self, root, caption_mode="random"):
        if caption_mode not in ("random", "first"):
            raise ValueError(f"caption_mode must be 'random' or 'first', got {caption_mode!r}")
        self.cache_dir = Path(root)
        self.z_mean = np.load(self.cache_dir / "z_mean.npy", mmap_mode="r")
        self.z_logvar = np.load(self.cache_dir / "z_logvar.npy", mmap_mode="r")

        self.text_features = np.load(self.cache_dir / "text_features.npy", mmap_mode="r")
        self.empty_text_feature = np.load(self.cache_dir / "empty_text_feature.npy", mmap_mode="r")

        self.caption_mode = caption_mode

    def __len__(self):
        return len(self.z_mean)

    def __getitem__(self, index):
        j = random.randrange(self.text_features.shape[1])

        z_mean = torch.from_numpy(self.z_mean[index].copy())
        if self.caption_mode == "random":
            text_feature = torch.from_numpy(self.text_features[index, j].copy())
        else:
            text_feature = torch.from_numpy(self.text_features[index, 0].copy())

        return z_mean, text_feature
