"""VAE evaluation using the same config, dataset and loss as main.py.

Run with the existing configuration interface::

    python vae_eval_0.py \
        --schema configs/autoencoders/schema.yaml \
        --config configs/autoencoders/vae_mmchq_z32_init64.yaml \
        mode.eval=true \
        path.resume_path=results/<run>/ckpts/ckpt_epoch_0020.pth

Optional image paths can be passed to main(config, image_paths=[...]) or added
to the image_paths list at the bottom of this file. Both functions follow
VAE.forward(), including its posterior sampling, just like vae_engine.test().
"""

import pickle
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

from data.build_dataloader import build_mmcelebahq_dataloader
from metrics.losses import vae_loss
from models.autoencoders.vae_midattn import VAE
from utils import unis
from utils.config import load_config
from utils.logger import create_logger


@torch.no_grad()
def evaluate_vae(config, model, device, dataloader):
    """Evaluate test loss and image metrics, averaging over actual samples.

    Test loss uses the existing vae_loss on the raw model outputs. MSE, MAE
    and PSNR use image values in [0, 1], with reconstructions clipped to that
    range. PSNR is averaged per image with a 1e-12 MSE floor.
    """
    model.eval()
    totals = {"test_loss": 0.0, "mse": 0.0, "mae": 0.0, "psnr": 0.0}
    sample_count = 0

    for X, _ in tqdm(dataloader, desc="VAE evaluation"):
        X = X.to(device, non_blocking=True)
        z_mean, z_logvar, x_mean, x_logvar = model(X)
        loss = vae_loss(X, z_mean, z_logvar, x_mean, x_logvar, config.model.beta)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite VAE test loss; check the checkpoint and model outputs.")

        image = ((X + 1) / 2).clamp(0, 1)
        reconstruction = ((x_mean + 1) / 2).clamp(0, 1)
        difference = image - reconstruction
        mse = difference.square().flatten(1).mean(1)
        mae = difference.abs().flatten(1).mean(1)
        psnr = -10 * mse.clamp_min(1e-12).log10()
        batch_size = X.size(0)

        totals["test_loss"] += loss.item() * batch_size
        totals["mse"] += mse.sum().item()
        totals["mae"] += mae.sum().item()
        totals["psnr"] += psnr.sum().item()
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("The test dataloader contains no images.")
    metrics = {name: total / sample_count for name, total in totals.items()}
    metrics["num_samples"] = sample_count
    return metrics


@torch.no_grad()
def infer_images(config, model, device, image_paths, img_transform):
    """Reuse the dataset transform and save input/reconstruction image pairs.

    Each saved PNG shows the resized input on the left and reconstruction on
    the right. img_transform is taken from the existing test dataset.
    """
    model.eval()
    if isinstance(image_paths, (str, Path)):
        image_paths = [image_paths]
    output_dir = Path(config.path.output_dir) / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for index, image_path in enumerate(image_paths):
        image_path = Path(image_path)
        with Image.open(image_path) as image:
            X = img_transform(image.convert("RGB")).unsqueeze(0).to(device)

        _, _, x_mean, _ = model(X)
        if not torch.isfinite(x_mean).all():
            raise FloatingPointError(f"Non-finite reconstruction for {image_path}.")
        image = ((X[0] + 1) / 2).clamp(0, 1)
        reconstruction = ((x_mean[0] + 1) / 2).clamp(0, 1)
        comparison = torch.cat((image, reconstruction), dim=-1)
        save_path = output_dir / f"{index:04d}_{image_path.stem}.png"
        save_image(comparison, save_path)
        saved_paths.append(save_path)

    return saved_paths


def main(config, image_paths=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.seed_everything:
        unis.seed_everything(config.seed)

    logger = create_logger(config.path.logs_dir, 0, name=config.model.name)
    unis.results_mkdirs(config, logger)

    model = VAE(
        in_channels=config.data.img_channels,
        init_channels=config.model.init_channels,
        z_channels=config.model.z_channels,
        x_channels=config.model.x_channels,
        ch_mult=list(config.model.ch_mult),
        num_groups=config.model.num_groups,
        shift_factor=config.model.shift_factor,
        scale_factor=config.model.scale_factor,
    ).to(device)
    checkpoint = torch.load(config.path.resume_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    del state_dict, checkpoint
    logger.info(f"Loaded VAE weights from {config.path.resume_path}")

    # The existing test loader is not sharded and needs no DDP process group.
    _, _, test_dataloader, _ = build_mmcelebahq_dataloader(config, rank=0)
    metrics = evaluate_vae(config, model, device, test_dataloader)
    logger.info(f"VAE test metrics: {metrics}")

    data_dump_file = Path(config.path.data_dump_dir) / "eval_info.pkl"
    with data_dump_file.open("wb") as file:
        pickle.dump({"config": config, "metrics": metrics}, file)
    logger.info(f"Saved evaluation data to {data_dump_file}")

    if image_paths:
        img_transform = test_dataloader.dataset.dataset.img_transform
        saved_paths = infer_images(config, model, device, image_paths, img_transform)
        for path in saved_paths:
            logger.info(f"Saved input/reconstruction comparison to {path}")

    return metrics


if __name__ == "__main__":
    config = load_config()
    image_paths = []  # Optional, e.g. ["datasets/mm-celeba-hq/images/0.jpg"]

    main(config, image_paths=image_paths)
