"""Build VAE and CLIP caches using the existing training config.

Run from the repository root:
    python -B scripts/build_ldm_cache.py \
        --schema configs/diffusionmodels/schema.yaml \
        --config configs/diffusionmodels/dit_medium4.yaml

Both encoders follow config.dist.amp on CUDA; arrays are stored in FP32.
metadata.json is written only after the cache is complete. Existing output
folders are never overwritten. Sample latent noise and apply prompt dropout
online during training. text_features[i, j] is caption j of dataset.images[i];
images retain the dataset's numeric filename order. Every image must have the
same positive number of captions. Dataset splitting remains a training step.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from data.build_datasets import MMCelebAHQDataset
from models.autoencoders.vae_midattn import VAE
from models.diffusionmodels.dit import PromptEncoder
from utils.config import load_config


def read_captions(dataset):
    captions_by_image = []
    for image in dataset.images:
        caption_path = Path(dataset.caption_dir / f"{image.stem}.txt")
        captions = [s.strip() for s in caption_path.read_text(encoding="utf-8").splitlines() if s.strip()]
        if not captions:
            raise ValueError(f"No captions: {caption_path}")
        if captions_by_image and len(captions) != len(captions_by_image[0]):
            raise ValueError(f"Caption count differs between images: {caption_path}")
        captions_by_image.append(captions)
    return captions_by_image


@torch.no_grad()
def build_latent_cache(config, dataset, output, device):
    # load vae encoder and z_channels
    checkpoint = torch.load(config.path.vae_ckpt_path, map_location="cpu", weights_only=True)
    vae = VAE(**checkpoint["model_kwargs"])
    vae.load_state_dict(checkpoint["model"], strict=True)
    encoder = vae._encoder.eval().requires_grad_(False).to(device)
    channels = checkpoint["model_kwargs"]["z_channels"]
    del vae, checkpoint

    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.dist.num_workers,
        pin_memory=device.type == "cuda",
    )
    shape = (len(dataset), channels, config.model.z_img_size, config.model.z_img_size)
    means = np.lib.format.open_memmap(output / "z_mean.npy", mode="w+", dtype="float32", shape=shape)
    offset = 0
    for images, _ in tqdm(loader, desc="VAE cache"):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, dtype=torch.float16, enabled=config.dist.amp and device.type == "cuda"):
            mean, logvar = encoder(images)
        values = mean.float().cpu().numpy()
        if values.shape[1:] != shape[1:] or not np.isfinite(values).all():
            raise ValueError("Invalid VAE mean shape or non-finite values")
        means[offset : offset + len(images)] = values
        offset += len(images)

    logvar = logvar.float().cpu().numpy()
    if logvar.size != 1 or not np.isfinite(logvar).all():
        raise ValueError("Expected one finite shared z_logvar")
    np.save(output / "z_logvar.npy", logvar.reshape(()), allow_pickle=False)
    means.flush()


@torch.no_grad()
def build_text_cache(config, captions_by_image, output, device):
    encoder = PromptEncoder(text_encoder_path=config.path.text_encoder_path).eval().requires_grad_(False).to(device)
    image_count, caption_count = len(captions_by_image), len(captions_by_image[0])
    features = np.lib.format.open_memmap(
        output / "text_features.npy",
        mode="w+",
        dtype="float32",
        shape=(image_count, caption_count, encoder.output_dim),
    )
    # A flat view allows batches to cross image boundaries without extra mappings.
    flat_features = features.reshape(-1, encoder.output_dim)
    texts = [caption for captions in captions_by_image for caption in captions]
    batch_size = config.train.batch_size
    for start in tqdm(range(0, len(texts), batch_size), desc="CLIP cache"):
        with torch.amp.autocast(device.type, dtype=torch.float16, enabled=config.dist.amp and device.type == "cuda"):
            values = encoder(texts[start : start + batch_size]).float().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Non-finite CLIP features")
        flat_features[start : start + len(values)] = values
    features.flush()
    with torch.amp.autocast(device.type, dtype=torch.float16, enabled=config.dist.amp and device.type == "cuda"):
        empty_feature = encoder([""])[0].float().cpu().numpy()
    if not np.isfinite(empty_feature).all():
        raise ValueError("Non-finite empty text feature")
    np.save(output / "empty_text_feature.npy", empty_feature, allow_pickle=False)


def main(config):
    output = Path(config.path.cache_path)
    if output.exists():
        raise FileExistsError(f"Cache directory already exists: {output}")
    if not Path(config.path.vae_ckpt_path).is_file():
        raise FileNotFoundError(config.path.vae_ckpt_path)
    if not Path(config.path.text_encoder_path).is_dir():
        raise FileNotFoundError(config.path.text_encoder_path)

    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(tuple(config.data.img_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    dataset = MMCelebAHQDataset(config.path.dataset_path, img_transform=transform, caption_mode="first")
    if not len(dataset):
        raise ValueError("Dataset is empty")
    captions_by_image = read_captions(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output.mkdir(parents=True, exist_ok=False)
    build_latent_cache(config, dataset, output, device)
    build_text_cache(config, captions_by_image, output, device)

    metadata = {
        "format_version": 3,
        "status": "complete",
        "text_layout": ["image", "caption", "feature"],
        "image_order": "numeric filename stem, ascending",
        "dataset_root": str(Path(config.path.dataset_path).resolve()),
        "vae_checkpoint": str(Path(config.path.vae_ckpt_path).resolve()),
        "text_encoder_path": str(Path(config.path.text_encoder_path).resolve()),
        "img_size": list(config.data.img_size),
        "compute_precision": "autocast_fp16" if config.dist.amp and device.type == "cuda" else "fp32",
    }
    temporary = output / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output / "metadata.json")
    print(f"Cache complete: {output}")


if __name__ == "__main__":
    main(load_config())
