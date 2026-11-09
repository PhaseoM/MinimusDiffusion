"""Build VAE and CLIP caches using the existing training config.

Run from the repository root:
    python -B -m data.build_cache.build_ldm_cache \
        --config configs/config.yaml

Both encoders follow config.optim.amp on CUDA; arrays are stored in FP32.
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
from models.diffusions.dit import PromptEncoder
from utils.config import Config, load_config


def get_vae_kwargs(checkpoint):
    if "model_kwargs" in checkpoint:
        return dict(checkpoint["model_kwargs"])
    config = checkpoint["config"]
    model_config = config["model"]
    vae_config = model_config.get("autoencoder", model_config)
    # beta belongs to the loss, not the VAE constructor.
    keys = (
        "in_channels",
        "init_channels",
        "z_channels",
        "x_channels",
        "ch_mult",
        "num_groups",
        "shift_factor",
        "scale_factor",
    )
    kwargs = {key: vae_config[key] for key in keys if key in vae_config}
    if "in_channels" not in kwargs:
        data_config = config["data"]
        kwargs["in_channels"] = data_config.get("dataset", data_config)["img_channels"]
    return kwargs


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
def build_latent_cache(config: Config, dataset, output, device):
    if not len(dataset):
        raise ValueError("Dataset is empty")
    # load vae encoder and z_channels
    checkpoint = torch.load(config.model.autoencoder_path, map_location="cpu", weights_only=True)
    vae_kwargs = get_vae_kwargs(checkpoint)
    channels = vae_kwargs["z_channels"]
    if channels != config.model.diffusion.channels:
        raise ValueError("VAE z_channels does not match model.diffusion.channels")
    vae = VAE(**vae_kwargs)
    vae.load_state_dict(checkpoint["model"], strict=True)
    encoder = vae._encoder.eval().requires_grad_(False).to(device)
    del vae, checkpoint

    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.train.num_workers,
        pin_memory=config.data.train.pin_memory and device.type == "cuda",
    )
    size = config.model.diffusion.img_size
    shape = (len(dataset), channels, size, size)
    means = np.lib.format.open_memmap(output / "z_mean.npy", mode="w+", dtype="float32", shape=shape)
    offset = 0
    for images, _ in tqdm(loader, desc="VAE cache"):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(
            device.type,
            dtype=getattr(torch, config.optim.amp.autocast_dtype),
            enabled=config.optim.amp.enabled and device.type == "cuda",
        ):
            mean, logvar = encoder(images)
        values = mean.float().cpu().numpy()
        expected_shape = (len(images), *shape[1:])
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"Expected finite VAE means of shape {expected_shape}, got {values.shape}")
        means[offset : offset + len(images)] = values
        offset += len(images)

    if offset != len(dataset):
        raise ValueError(f"Expected {len(dataset)} latent samples, wrote {offset}")
    logvar = logvar.float().cpu().numpy()
    if logvar.size != 1 or not np.isfinite(logvar).all():
        raise ValueError("Expected one finite shared z_logvar")
    np.save(output / "z_logvar.npy", logvar.reshape(()), allow_pickle=False)
    means.flush()


@torch.no_grad()
def build_text_cache(config: Config, captions_by_image, output, device):
    encoder = (
        PromptEncoder(text_encoder_path=config.model.diffusion.text_encoder_path).eval().requires_grad_(False).to(device)
    )
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
    batch_size = config.data.batch_size
    for start in tqdm(range(0, len(texts), batch_size), desc="CLIP cache"):
        with torch.amp.autocast(
            device.type,
            dtype=getattr(torch, config.optim.amp.autocast_dtype),
            enabled=config.optim.amp.enabled and device.type == "cuda",
        ):
            values = encoder(texts[start : start + batch_size]).float().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Non-finite CLIP features")
        flat_features[start : start + len(values)] = values
    features.flush()
    with torch.amp.autocast(
        device.type,
        dtype=getattr(torch, config.optim.amp.autocast_dtype),
        enabled=config.optim.amp.enabled and device.type == "cuda",
    ):
        empty_feature = encoder([""])[0].float().cpu().numpy()
    if not np.isfinite(empty_feature).all():
        raise ValueError("Non-finite empty text feature")
    np.save(output / "empty_text_feature.npy", empty_feature, allow_pickle=False)


def main(config: Config):
    output = Path(config.data.dataset.cache_path)
    if output.exists():
        raise FileExistsError(f"Cache directory already exists: {output}")
    if not Path(config.model.autoencoder_path).is_file():
        raise FileNotFoundError(config.model.autoencoder_path)
    if not Path(config.model.diffusion.text_encoder_path).is_dir():
        raise FileNotFoundError(config.model.diffusion.text_encoder_path)

    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(tuple(config.data.dataset.img_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    dataset = MMCelebAHQDataset(config.data.dataset.dataset_path, img_transform=transform, caption_mode="first")
    if not len(dataset):
        raise ValueError("Dataset is empty")
    captions_by_image = read_captions(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output.mkdir(parents=True, exist_ok=False)
    build_latent_cache(config, dataset, output, device)
    build_text_cache(config, captions_by_image, output, device)

    precision = {"float16": "fp16", "bfloat16": "bf16"}.get(
        config.optim.amp.autocast_dtype, config.optim.amp.autocast_dtype
    )
    metadata = {
        "format_version": 3,
        "status": "complete",
        "text_layout": ["image", "caption", "feature"],
        "image_order": "numeric filename stem, ascending",
        "dataset_root": str(Path(config.data.dataset.dataset_path).resolve()),
        "vae_checkpoint": str(Path(config.model.autoencoder_path).resolve()),
        "text_encoder_path": str(Path(config.model.diffusion.text_encoder_path).resolve()),
        "img_size": list(config.data.dataset.img_size),
        "compute_precision": f"autocast_{precision}" if config.optim.amp.enabled and device.type == "cuda" else "fp32",
    }
    temporary = output / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output / "metadata.json")
    print(f"Cache complete: {output}")


if __name__ == "__main__":
    main(load_config())
