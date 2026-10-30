"""Evaluate a VAE checkpoint and reconstruct selected images.

Examples (run from the project root)::

    python vae_eval.py --checkpoint results/<run>/ckpts/ckpt_epoch_0020.pth
    python vae_eval.py --checkpoint <ckpt> --mode infer --images photo.jpg
    python vae_eval.py --checkpoint <ckpt> --images photo.jpg another.png
    python vae_eval.py --checkpoint <weights.pth> --config configs/autoencoders/vae_mmchq_z32_init64.yaml

``evaluate_vae`` evaluates a dataloader; ``infer_images`` saves individual
reconstructions. Both use the posterior mean by default. Pass
``--sample-posterior`` to sample latents as in VAE.forward(). Image metrics use
RGB/grayscale values clipped to [0, 1]; the VAE loss uses unclipped outputs and
the same formula as metrics/losses.py, without the Gaussian constant or 1/2.
This script runs in one process and does not require torchrun or TensorFlow.
"""

import argparse
import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2
from torchvision.utils import save_image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
METRIC_NAMES = ("vae_loss", "recon_loss", "kl_loss", "mse", "mae", "psnr_db", "ssim")
DEFAULT_CONFIG = {
    "seed": 42,
    "model": {
        "init_channels": 64, "z_channels": 32, "x_channels": 3,
        "ch_mult": [1, 2, 4, 4], "num_groups": 1,
        "shift_factor": 0.0, "scale_factor": 1.0, "beta": 0.5,
    },
    "data": {"img_channels": 3, "img_size": [256, 256], "val_ratio": 0.1},
    "path": {"dataset_dir": "./datasets", "dataset_path": "${.dataset_dir}/mm-celeba-hq"},
    "test": {"batch_size": 16},
}


def _image_transform(image_size: Sequence[int], channels: int = 3):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(tuple(image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5] * channels, std=[0.5] * channels),
    ])


def _read_image(path: Path, transform, channels: int) -> Tensor:
    with Image.open(path) as image:
        return transform(image.convert("RGB" if channels == 3 else "L")).as_subclass(Tensor)


def _ssim(target: Tensor, prediction: Tensor) -> Tensor:
    """Per-image SSIM: Gaussian window, sigma=1.5, population covariance."""
    size = min(11, target.shape[-2], target.shape[-1])
    size -= 1 - size % 2
    coordinates = torch.arange(size, device=target.device, dtype=target.dtype) - size // 2
    weights = torch.exp(-coordinates.square() / (2 * 1.5**2))
    weights /= weights.sum()
    window = torch.outer(weights, weights).expand(target.shape[1], 1, size, size)

    def average(value):
        return F.conv2d(value, window, groups=target.shape[1])

    mean_target, mean_prediction = average(target), average(prediction)
    variance_target = (average(target.square()) - mean_target.square()).clamp_min(0)
    variance_prediction = (average(prediction.square()) - mean_prediction.square()).clamp_min(0)
    covariance = average(target * prediction) - mean_target * mean_prediction
    numerator = (2 * mean_target * mean_prediction + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (mean_target.square() + mean_prediction.square() + 0.01**2) * (
        variance_target + variance_prediction + 0.03**2
    )
    return (numerator / denominator).flatten(1).mean(1)


def _metrics(images: Tensor, outputs: tuple[Tensor, Tensor, Tensor, Tensor], beta: float):
    z_mean, z_logvar, reconstruction, x_logvar = (value.float() for value in outputs)
    images = images.float()
    recon_loss = ((images - reconstruction).square() / x_logvar.exp() + x_logvar).flatten(1).sum(1)
    kl_loss = (z_mean.square() + z_logvar.exp() - z_logvar - 1).flatten(1).sum(1)
    target = ((images + 1) / 2).clamp(0, 1)
    prediction = ((reconstruction + 1) / 2).clamp(0, 1)
    difference = target - prediction
    mse = difference.square().flatten(1).mean(1)
    values = {
        "vae_loss": recon_loss + beta * kl_loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
        "mse": mse,
        "mae": difference.abs().flatten(1).mean(1),
        # A 1e-12 floor keeps perfect reconstruction finite (120 dB).
        "psnr_db": -10 * mse.clamp_min(1e-12).log10(),
        "ssim": _ssim(target, prediction),
    }
    for name, value in values.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite {name}; check the checkpoint and model outputs.")
    return values, target, prediction


def _reconstruct(model: nn.Module, images: Tensor, sample_posterior: bool, generator):
    z_mean, z_logvar = model.encode(images)
    latent = z_mean
    if sample_posterior:
        noise = torch.randn(z_mean.shape, dtype=z_mean.dtype, device=z_mean.device, generator=generator)
        latent = z_mean + (0.5 * z_logvar).exp() * noise
    reconstruction, x_logvar = model.decode(latent)
    if reconstruction.shape != images.shape:
        raise ValueError(f"Reconstruction shape {tuple(reconstruction.shape)} != input {tuple(images.shape)}")
    return z_mean, z_logvar, reconstruction, x_logvar


@torch.inference_mode()
def evaluate_vae(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device = "cpu",
    *,
    beta: float = 0.5,
    sample_posterior: bool = False,
    seed: int = 42,
    max_samples: int | None = None,
    output_dir: str | Path | None = None,
    num_preview: int = 8,
    progress: bool = True,
) -> dict:
    """Return sample-averaged reconstruction metrics and optionally save a grid.

    Inputs must use the training normalization [-1, 1]. Each grid row contains
    the input on the left and reconstruction on the right. ``vae_loss`` uses
    the selected latent mode; mean mode is deterministic, not a Monte Carlo
    estimate of the expected reconstruction term in the ELBO.
    """
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    device = torch.device(device)
    model.to(device).eval()
    generator = torch.Generator(device=device).manual_seed(seed) if sample_posterior else None
    sample_limit = len(dataloader.dataset)
    if max_samples is not None:
        sample_limit = min(sample_limit, max_samples)
    totals = torch.zeros(len(METRIC_NAMES), dtype=torch.float64, device=device)
    count, preview_count = 0, 0
    previews = []
    with tqdm(total=sample_limit, desc="VAE evaluation", unit="image", disable=not progress) as bar:
        for batch in dataloader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images[:sample_limit - count].to(device, non_blocking=True)
            if images.shape[0] == 0:
                break
            values, target, prediction = _metrics(
                images, _reconstruct(model, images, sample_posterior, generator), beta
            )
            totals += torch.stack([values[name].double().sum() for name in METRIC_NAMES])
            take = min(max(0, num_preview - preview_count), images.shape[0])
            if output_dir is not None and take:
                previews.append(torch.stack((target[:take], prediction[:take]), dim=1).flatten(0, 1).cpu())
                preview_count += take
            count += images.shape[0]
            bar.update(images.shape[0])
            if count >= sample_limit:
                break
    if count == 0:
        raise ValueError("The evaluation split contains no images")
    result = dict(zip(METRIC_NAMES, (totals / count).cpu().tolist()))
    result.update(num_samples=count, posterior_mode="sample" if sample_posterior else "mean", beta=beta)
    if previews:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        comparison_path = destination / "reconstructions.png"
        save_image(torch.cat(previews), comparison_path, nrow=2, padding=2, pad_value=1)
        result["comparison_path"] = str(comparison_path)
    return result


@torch.inference_mode()
def infer_images(
    model: nn.Module,
    image_paths: str | Path | Sequence[str | Path],
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    *,
    image_size: Sequence[int] = (256, 256),
    image_channels: int = 3,
    beta: float = 0.5,
    sample_posterior: bool = False,
    seed: int = 42,
    progress: bool = True,
) -> list[dict]:
    """Save input/reconstruction/comparison PNGs and return per-image metrics.

    Files and directories are accepted. Inputs are resized exactly as for
    training; comparison images show the resized input on the left.
    """
    if isinstance(image_paths, (str, Path)):
        image_paths = [image_paths]
    paths = []
    for entry in image_paths:
        entry = Path(entry)
        if entry.is_dir():
            paths.extend(sorted(path for path in entry.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES))
        elif entry.is_file():
            paths.append(entry)
        else:
            raise FileNotFoundError(entry)
    if not paths:
        raise ValueError("No inference images were provided")
    if image_channels not in (1, 3):
        raise ValueError("Image inference supports one or three channels")
    device = torch.device(device)
    model.to(device).eval()
    transform = _image_transform(image_size, image_channels)
    generator = torch.Generator(device=device).manual_seed(seed) if sample_posterior else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, path in enumerate(tqdm(paths, desc="VAE inference", disable=not progress)):
        images = _read_image(path, transform, image_channels).unsqueeze(0).to(device)
        values, target, prediction = _metrics(
            images, _reconstruct(model, images, sample_posterior, generator), beta
        )
        prefix = output_dir / f"{index:04d}_{path.stem}"
        input_path = Path(f"{prefix}_input.png")
        reconstruction_path = Path(f"{prefix}_reconstruction.png")
        comparison_path = Path(f"{prefix}_comparison.png")
        save_image(target[0], input_path)
        save_image(prediction[0], reconstruction_path)
        save_image(torch.cat((target[0], prediction[0]), dim=-1), comparison_path)
        results.append({
            "image": str(path.resolve()),
            "input_path": str(input_path),
            "reconstruction_path": str(reconstruction_path),
            "comparison_path": str(comparison_path),
            **{name: value.item() for name, value in values.items()},
        })
    return results


class _ImageDataset(Dataset):
    """Read images only, preserving MMCelebAHQDataset's numeric JPG ordering."""

    def __init__(self, root: Path, transform, channels: int):
        image_dir = root / "images"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        self.paths = sorted((path for path in image_dir.glob("*.jpg") if path.is_file()), key=lambda path: int(path.stem))
        self.transform, self.channels = transform, channels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        return _read_image(path, self.transform, self.channels), str(path)


def _load_vae(checkpoint_path: Path, model_name: str, config_path: Path | None, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected a training checkpoint or a model state_dict")
    full_checkpoint = "model" in checkpoint
    config = OmegaConf.create(DEFAULT_CONFIG)
    if full_checkpoint and checkpoint.get("config") is not None:
        config = OmegaConf.merge(config, checkpoint["config"])
    if config_path is not None:
        config = OmegaConf.merge(config, OmegaConf.load(config_path))

    model_kwargs = {
        "in_channels": config.data.img_channels,
        **{key: config.model[key] for key in (
            "init_channels", "z_channels", "x_channels", "num_groups", "shift_factor", "scale_factor"
        )},
        "ch_mult": list(config.model.ch_mult),
    }
    # Checkpoint architecture metadata takes precedence over evaluation settings.
    if full_checkpoint and checkpoint.get("model_kwargs") is not None:
        model_kwargs.update(OmegaConf.to_container(OmegaConf.create(checkpoint["model_kwargs"]), resolve=True))
    if model_kwargs["in_channels"] not in (1, 3) or model_kwargs["x_channels"] != model_kwargs["in_channels"]:
        raise ValueError("Reconstruction evaluation requires matching one- or three-channel input/output")

    state = checkpoint["model"] if full_checkpoint else checkpoint
    if not isinstance(state, Mapping) or not state or not all(isinstance(value, Tensor) for value in state.values()):
        raise ValueError("The model state_dict must contain tensor weights")
    state = dict(state)
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    for name, value in state.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f"Checkpoint parameter {name!r} contains NaN/Inf")
    model = importlib.import_module(f"models.autoencoders.{model_name}").VAE(**model_kwargs)
    model.load_state_dict(state, strict=True)
    epoch = checkpoint.get("epoch") if full_checkpoint else None
    return model.to(device).eval(), config, model_kwargs, epoch


def _evaluation_loader(config, dataset_root: Path, image_size, channels: int, split: str, batch_size: int, workers: int, device):
    dataset = _ImageDataset(dataset_root, _image_transform(image_size, channels), channels)
    if split != "all":
        ratio = float(config.data.val_ratio)
        if not 0 <= ratio < 0.5:
            raise ValueError("data.val_ratio must be in [0, 0.5)")
        held_out = int(len(dataset) * ratio)
        splits = random_split(
            dataset, [len(dataset) - 2 * held_out, held_out, held_out],
            generator=torch.Generator().manual_seed(int(config.seed)),
        )
        dataset = splits[("train", "val", "test").index(split)]
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", "--ckpt", type=Path, required=True)
    parser.add_argument("--model", choices=("vae_midattn", "vae_interleave"), default="vae_midattn")
    parser.add_argument("--config", type=Path, help="Optional YAML; also supplies architecture for bare weights")
    parser.add_argument("--mode", choices=("eval", "infer", "both"), help="Default: eval, or both when --images is supplied")
    parser.add_argument("--images", "--image", nargs="+", type=Path, help="Image files or directories")
    parser.add_argument("--dataset-root", type=Path, help="MM-CelebA-HQ root containing images/")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--output-dir", type=Path, help="Default: <run>/vae_eval/<checkpoint stem>/")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--max-samples", type=int, help="Evaluate only the first N images of the chosen split")
    parser.add_argument("--num-preview", type=int, default=8, help="Number of evaluation reconstruction pairs to save")
    parser.add_argument("--seed", type=int, help="Sampling seed; the split seed remains the training config's seed")
    parser.add_argument("--sample-posterior", action="store_true", help="Sample z instead of decoding its posterior mean")
    parser.add_argument("--quiet", action="store_true", help="Hide progress bars")
    args = parser.parse_args()
    args.mode = args.mode or ("both" if args.images else "eval")
    if args.mode in ("infer", "both") and not args.images:
        parser.error("--images is required for infer/both mode")
    if args.num_workers < 0 or args.num_preview < 0:
        parser.error("--num-workers and --num-preview must be nonnegative")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    return args


def main():
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    model, config, model_kwargs, epoch = _load_vae(args.checkpoint, args.model, args.config, device)
    image_size = tuple(args.image_size or config.data.img_size)
    factor = 2 ** (len(model_kwargs["ch_mult"]) - 1)
    if len(image_size) != 2 or any(size <= 0 or size % factor for size in image_size):
        raise ValueError(f"Image height and width must be positive multiples of {factor}")
    channels = model_kwargs["in_channels"]
    seed = int(config.seed if args.seed is None else args.seed)
    run_dir = args.checkpoint.parent.parent if args.checkpoint.parent.name == "ckpts" else args.checkpoint.parent
    output_dir = args.output_dir or run_dir / "vae_eval" / args.checkpoint.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": epoch,
        "model": args.model, "model_kwargs": model_kwargs, "device": str(device),
        "image_size": list(image_size), "seed": seed,
        "posterior_mode": "sample" if args.sample_posterior else "mean",
        "image_metrics_range": "[0, 1], reconstructions clipped",
        "loss_convention": "Unclipped outputs; same formula as metrics/losses.py",
    }
    if args.mode in ("eval", "both"):
        dataset_root = args.dataset_root
        if dataset_root is None:
            dataset_root = Path(config.path.dataset_path)
            if not dataset_root.is_absolute():
                dataset_root = PROJECT_ROOT / dataset_root
        loader = _evaluation_loader(
            config, dataset_root, image_size, channels, args.split,
            args.batch_size or int(config.test.batch_size), args.num_workers, device,
        )
        report.update(dataset_root=str(dataset_root.resolve()), split=args.split, split_seed=int(config.seed))
        report["evaluation"] = evaluate_vae(
            model, loader, device, beta=float(config.model.beta), sample_posterior=args.sample_posterior,
            seed=seed, max_samples=args.max_samples, output_dir=output_dir,
            num_preview=args.num_preview, progress=not args.quiet,
        )
    if args.mode in ("infer", "both"):
        report["inference"] = infer_images(
            model, args.images, output_dir / "images", device,
            image_size=image_size, image_channels=channels, beta=float(config.model.beta),
            sample_posterior=args.sample_posterior, seed=seed, progress=not args.quiet,
        )
    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Saved evaluation results to {report_path}")


if __name__ == "__main__":
    main()
