import torch
from pathlib import Path
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image


class ISIC2016Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None, joint_transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.images = sorted(self.image_dir.glob("*.jpg"))
        if not self.images:
            raise RuntimeError(f"No jpg image found in {self.images}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image_stem = image_path.stem
        mask_path = self.mask_dir / f"{image_stem}_Segmentation.png"

        image_entity = Image.open(image_path).convert("RGB")
        mask_entity = Image.open(mask_path).convert("L")

        if self.joint_transform is not None:
            image_entity, mask_entity = self.joint_transform(image_entity, mask_entity)

        if self.transform is not None:
            image_entity = self.transform(image_entity)

        if self.target_transform is not None:
            mask_entity = self.target_transform(mask_entity)

        return image_entity, mask_entity
