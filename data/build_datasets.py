import torch
import random
from pathlib import Path
from PIL import Image
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
