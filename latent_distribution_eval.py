"""Evaluate a VAE's aggregate posterior and save numerical/visual diagnostics.

Examples (from the project root)::

    python latent_distribution_eval.py --run-dir results/VAE_20261029_1131 --device cuda:1
    python latent_distribution_eval.py --checkpoint results/<run>/ckpts/ckpt_epoch_0100.pth
    python latent_distribution_eval.py --checkpoint <weights.pth> --config <config.yaml>

Default output: <run>/latent_distribution_eval/<checkpoint>/<split>_<timestamp>/
The script only encodes images, runs in one process, and never changes training
files. It evaluates raw encode() latents, before any diffusion scaling. Current
dataset support follows vae_eval.py: MM-CelebA-HQ, with the training split/resize/
normalization reproduced. Every invocation creates a new output directory.

The aggregate q(z) is a mixture over images. E_x KL(q(z|x)||N(0,I)) is NOT
KL(q(z)||N(0,I)). Moment-matched Gaussian KLs below are explicitly proxies;
histograms, correlations and random projections provide complementary checks.
"""

import argparse
import base64
import csv
import hashlib
import html
import json
import math
import os
import shlex
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
_plot_cache = tempfile.TemporaryDirectory(prefix="latent_distribution_plot_")
os.environ["MPLCONFIGDIR"] = str(Path(_plot_cache.name) / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(Path(_plot_cache.name) / "cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import special, stats
from tqdm import tqdm

from vae_eval import PROJECT_ROOT, _evaluation_loader, _load_vae


def _normal_quantiles(count):
    return special.ndtri((np.arange(count, dtype=np.float64) + 0.5) / count)


def _marginal_distances(values):
    """Descriptive W2/KS distances; no IID-based p-values for spatial tokens."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    count = len(ordered)
    cdf = special.ndtr(ordered)
    return {
        "w2_to_standard_normal": float(np.sqrt(np.mean((ordered - _normal_quantiles(count)) ** 2))),
        "ks_distance_to_standard_normal": float(max(
            np.max(np.arange(1, count + 1) / count - cdf),
            np.max(cdf - np.arange(count) / count))),
    }


def _mixture_moments(mu, variance, axes):
    """Exact moments of the empirical diagonal-Gaussian mixture, not MC moments."""
    mean = mu.mean(dim=axes, keepdim=True)
    centered = mu - mean
    mean_variance = variance.mean(dim=axes)
    variance_mu = centered.square().mean(dim=axes)
    total_variance = variance_mu + mean_variance
    third = (centered.pow(3) + 3 * centered * variance).mean(dim=axes)
    fourth = (centered.pow(4) + 6 * centered.square() * variance + 3 * variance.square()).mean(dim=axes)
    return {
        "mean": mean.squeeze(), "variance_mu": variance_mu,
        "mean_posterior_variance": mean_variance, "variance": total_variance,
        "std": total_variance.sqrt(), "skewness": third / total_variance.pow(1.5),
        "excess_kurtosis": fourth / total_variance.square() - 3,
    }


def _covariance(mu_rows, variance_rows):
    """Cov_q = Cov_x(mu) + E_x diag(variance), using population denominator N."""
    centered = mu_rows - mu_rows.mean(0)
    covariance_mu = centered.T @ centered / len(centered)
    return covariance_mu, covariance_mu + torch.diag(variance_rows.mean(0))


def _correlation(covariance):
    std = covariance.diag().clamp_min(0).sqrt()
    denominator = std[:, None] * std[None, :]
    # Zero-variance mean channels have undefined correlation; mark them NaN in NPZ/plots.
    return torch.where(denominator > 0, covariance / denominator.clamp_min(1e-300), torch.nan)


def _off_diagonal_summary(matrix):
    values = matrix[~torch.eye(len(matrix), dtype=torch.bool)].abs()
    values = values[torch.isfinite(values)]
    return {"mean_abs": float(values.mean()) if len(values) else None,
            "max_abs": float(values.max()) if len(values) else None}


@torch.inference_mode()
def collect_latents(model, dataloader, device, max_samples=512, seed=42, progress=True):
    """Collect mu, broadcast logvar and one fixed posterior draw per image on CPU."""
    model.to(device).eval()
    means, logvars, samples, identifiers = [], [], [], []
    limit = min(max_samples, len(dataloader.dataset))
    with tqdm(total=limit, desc="Encode latents", unit="image", disable=not progress) as bar:
        count = 0
        for batch in dataloader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            take = min(len(images), limit - count)
            if take <= 0:
                break
            mu, logvar = model.encode(images[:take].to(device))
            mu, logvar = mu.float(), logvar.float()
            if mu.ndim != 4:
                raise ValueError(f"Expected BCHW latents, got {tuple(mu.shape)}")
            logvar = torch.broadcast_to(logvar, mu.shape)
            if not torch.isfinite(mu).all() or not torch.isfinite(logvar).all():
                raise FloatingPointError("Encoder returned NaN/Inf")
            mu, logvar = mu.cpu(), logvar.cpu()
            noise = torch.stack([
                torch.randn(mu.shape[1:], generator=torch.Generator().manual_seed(seed + count + i))
                for i in range(take)
            ])
            z = mu + (0.5 * logvar).exp() * noise
            if not torch.isfinite(z).all():
                raise FloatingPointError("Non-finite posterior sample")
            means.append(mu)
            logvars.append(logvar)
            samples.append(z)
            paths = batch[1] if isinstance(batch, (tuple, list)) and len(batch) > 1 else None
            identifiers.extend([str(p) for p in paths[:take]] if paths is not None else [str(count + i) for i in range(take)])
            count += take
            bar.update(take)
            if count == limit:
                break
    if count < 2:
        raise ValueError("At least two images are required to assess aggregate variation")
    return torch.cat(means), torch.cat(logvars), torch.cat(samples), identifiers


def analyze_latents(mu, logvar, samples, *, seed=42, num_projections=64,
                    covariance_dims=128, marginal_samples=100_000, active_threshold=0.01):
    """Return JSON-compatible metrics plus arrays for plots/NPZ. All inputs are CPU BCHW."""
    if mu.shape != samples.shape or mu.ndim != 4 or len(mu) < 2:
        raise ValueError("mu and samples must have identical BCHW shapes with B >= 2")
    if min(num_projections, covariance_dims, marginal_samples) <= 0 or active_threshold < 0:
        raise ValueError("Diagnostic sample counts must be positive and active threshold nonnegative")
    mu = mu.detach().cpu().double()
    logvar = torch.broadcast_to(torch.as_tensor(logvar).detach().cpu().double(), mu.shape)
    samples = samples.detach().cpu().float()
    variance = logvar.exp()
    if not all(torch.isfinite(x).all() for x in (mu, logvar, samples, variance)) or not (variance > 0).all():
        raise FloatingPointError("Latents must be finite with positive finite posterior variance")
    n, channels, height, width = mu.shape
    dimensions = channels * height * width
    pooled = _mixture_moments(mu, variance, (0, 1, 2, 3))
    channel = _mixture_moments(mu, variance, (0, 2, 3))
    coordinate_mean = mu.mean(0)
    coordinate_mu_variance = mu.var(0, correction=0)
    coordinate_noise_variance = variance.mean(0)
    coordinate_variance = coordinate_mu_variance + coordinate_noise_variance
    conditional_kl = 0.5 * (mu.square() + variance - logvar - 1)
    kl_image = conditional_kl.flatten(1).sum(1)
    diagonal_proxy = 0.5 * (coordinate_mean.square() + coordinate_variance - coordinate_variance.log() - 1)
    active = coordinate_mu_variance > active_threshold
    rng = np.random.default_rng(seed)
    # Same scalar-coordinate subsample for mu and sampled z; positions are not IID images.
    indices = rng.choice(samples.numel(), size=min(marginal_samples, samples.numel()), replace=False)
    pooled_sample = samples.numpy().reshape(-1)[indices]
    pooled_mu = mu.numpy().reshape(-1)[indices]
    channel_rows, channel_samples = [], []
    for c in range(channels):
        values = samples[:, c].numpy().reshape(-1)
        subset = values[rng.choice(len(values), size=min(marginal_samples, len(values)), replace=False)]
        channel_samples.append(subset)
        channel_rows.append({"channel": c, **{key: float(value.reshape(-1)[c]) for key, value in channel.items()},
                             "conditional_kl_per_element": float(conditional_kl[:, c].mean()),
                             "active_coordinate_fraction": float(active[c].double().mean()),
                             **_marginal_distances(subset)})

    # Channel covariance pools (image, spatial position) as rows; it is NOT full latent covariance.
    mu_tokens = mu.permute(0, 2, 3, 1).reshape(-1, channels)
    variance_tokens = variance.permute(0, 2, 3, 1).reshape(-1, channels)
    channel_cov_mu, channel_cov = _covariance(mu_tokens, variance_tokens)
    channel_corr_mu, channel_corr = _correlation(channel_cov_mu), _correlation(channel_cov)
    chosen = np.sort(rng.choice(dimensions, size=min(covariance_dims, dimensions), replace=False))
    flat_mu, flat_variance = mu.flatten(1), variance.flatten(1)
    _, sub_cov = _covariance(flat_mu[:, chosen], flat_variance[:, chosen])
    eigenvalues, eigenvectors = torch.linalg.eigh(sub_cov)
    if eigenvalues.min() <= 0:
        raise FloatingPointError("Selected aggregate covariance is not numerically positive definite")
    sub_mean = flat_mu[:, chosen].mean(0)
    sub_kl = 0.5 * (sub_mean.square().sum() + eigenvalues.sum() - eigenvalues.log().sum() - len(chosen))

    # Full image-latent random projections, not projections of pooled C-dimensional tokens.
    generator = torch.Generator().manual_seed(seed + 100_003)
    directions = torch.randn(dimensions, num_projections, generator=generator)
    directions /= directions.norm(dim=0, keepdim=True)
    projected = (samples.flatten(1) @ directions).double().numpy()
    quantiles = _normal_quantiles(n)
    projection_w2 = np.sqrt(np.mean((np.sort(projected, axis=0) - quantiles[:, None]) ** 2, axis=0))
    baseline = rng.standard_normal((n, num_projections))
    baseline_w2 = np.sqrt(np.mean((np.sort(baseline, axis=0) - quantiles[:, None]) ** 2, axis=0))
    spatial = {}
    for name, first, second in (("horizontal", (..., slice(None), slice(None, -1)), (..., slice(None), slice(1, None))),
                                ("vertical", (..., slice(None, -1), slice(None)), (..., slice(1, None), slice(None)))):
        if (width if name == "horizontal" else height) < 2:
            spatial[name] = None
            continue
        a, b = mu[first], mu[second]
        ca, cb = a - a.mean((0, 2, 3), keepdim=True), b - b.mean((0, 2, 3), keepdim=True)
        va = ca.square().mean((0, 2, 3)) + variance[first].mean((0, 2, 3))
        vb = cb.square().mean((0, 2, 3)) + variance[second].mean((0, 2, 3))
        spatial[name] = ((ca * cb).mean((0, 2, 3)) / (va * vb).sqrt()).numpy()

    metrics = {
        "num_images": n, "latent_shape_chw": [channels, height, width], "dimensions_per_image": dimensions,
        "aggregate_pooled_analytic": {key: float(value) for key, value in pooled.items()},
        "aggregate_pooled_sampled": {"mean": float(samples.double().mean()), "std": float(samples.double().std(correction=0)),
                                     "scalar_subsample_count": len(pooled_sample), **_marginal_distances(pooled_sample)},
        "conditional_kl": {"mean_nats_per_image": float(kl_image.mean()),
                           "mean_nats_per_element": float(kl_image.mean() / dimensions),
                           "note": "Standard KL includes 1/2. This is E_x KL(q(z|x)||p), not KL(q(z)||p)."},
        "aggregate_gaussian_moment_proxies": {
            "diagonal_kl_nats": float(diagonal_proxy.sum()), "diagonal_kl_nats_per_element": float(diagonal_proxy.mean()),
            "selected_subspace_dimensions": len(chosen), "selected_subspace_full_covariance_kl_nats": float(sub_kl),
            "selected_subspace_kl_nats_per_element": float(sub_kl / len(chosen)),
            "selected_subspace_covariance_min_eigenvalue": float(eigenvalues.min()),
            "selected_subspace_covariance_max_eigenvalue": float(eigenvalues.max()),
            "note": "KL of moment-matched Gaussians only; not an exact aggregate-mixture KL. Subspace uses fixed CHW coordinates across images."},
        "active_units": {"variance_mu_threshold": active_threshold, "active_coordinate_count": int(active.sum()),
                         "coordinate_count": dimensions, "active_fraction": float(active.double().mean()),
                         "note": "Var_x(mu_j) > threshold; threshold is descriptive, not a universal collapse criterion."},
        "channel_correlation_aggregate": _off_diagonal_summary(channel_corr),
        "channel_correlation_mu": _off_diagonal_summary(channel_corr_mu),
        "selected_coordinate_correlation_aggregate": _off_diagonal_summary(_correlation(sub_cov)),
        "sliced_w2_full_latent": {"rms_w2": float(np.sqrt(np.mean(projection_w2 ** 2))),
                                 "num_random_unit_projections": num_projections, "images_per_projection": n,
                                 "prior_reference_rms_w2": float(np.sqrt(np.mean(baseline_w2 ** 2))),
                                 "max_projection_w2": float(projection_w2.max()),
                                 "note": "Sorted projections vs midpoint N(0,1) quantiles. Finite-sample approximation; prior_reference uses the same N/projection count."},
        "spatial_neighbor_correlation_aggregate": {key: value.tolist() if value is not None else None for key, value in spatial.items()},
        "per_channel": channel_rows,
    }
    arrays = {"coordinate_mean": coordinate_mean.numpy(), "coordinate_mu_variance": coordinate_mu_variance.numpy(),
              "coordinate_mean_posterior_variance": coordinate_noise_variance.numpy(),
              "coordinate_aggregate_variance": coordinate_variance.numpy(), "coordinate_conditional_kl": conditional_kl.mean(0).numpy(),
              "coordinate_active": active.numpy(), "channel_covariance_mu": channel_cov_mu.numpy(),
              "channel_covariance_aggregate": channel_cov.numpy(), "channel_correlation_mu": channel_corr_mu.numpy(),
              "channel_correlation_aggregate": channel_corr.numpy(), "selected_flat_coordinate_indices": chosen,
              "selected_covariance_aggregate": sub_cov.numpy(), "selected_correlation_aggregate": _correlation(sub_cov).numpy(),
              "selected_covariance_eigenvalues": eigenvalues.numpy(), "projection_w2": projection_w2,
              "prior_projection_w2": baseline_w2, "projected_z": projected,
              "pca_selected_coordinates_z": (samples.flatten(1)[:, chosen].double() @ eigenvectors[:, -2:]).numpy(),
              "pooled_sample_z": pooled_sample, "pooled_sample_mu": pooled_mu,
              "per_image_conditional_kl": kl_image.numpy(),
              "per_image_sample_norm2_per_dim": samples.double().square().flatten(1).mean(1).numpy(),
              "per_image_expected_norm2_per_dim": (mu.square() + variance).flatten(1).mean(1).numpy()}
    # Validate serialization now, rather than after plotting/writing partial reports.
    json.dumps(metrics, allow_nan=False)
    return metrics, arrays


def _save_figure(fig, output):
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_plots(metrics, arrays, output_dir):
    rows = metrics["per_channel"]
    channel_ids = np.arange(len(rows))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    ax = axes[0, 0]
    low, high = np.quantile(arrays["pooled_sample_z"], [0.001, 0.999])
    bins = np.linspace(min(-4, low), max(4, high), 100)
    for name, label in (("pooled_sample_mu", "mu(x)"), ("pooled_sample_z", "sampled q(z)")):
        # Include out-of-range mass in the normalization; tails are not renormalized away.
        value = arrays[name]
        counts, _ = np.histogram(value, bins=bins)
        ax.stairs(counts / (len(value) * np.diff(bins)), bins, label=label, linewidth=1.5)
    x = np.linspace(bins[0], bins[-1], 500)
    ax.plot(x, stats.norm.pdf(x), "k--", label="N(0,1)")
    ax.set(title="Pooled scalar marginals (not joint density)", xlabel="Latent value", ylabel="Density")
    ax.legend()
    ax = axes[0, 1]
    ordered = np.sort(arrays["pooled_sample_z"])
    keep = np.linspace(0, len(ordered) - 1, min(1000, len(ordered))).astype(int)
    q = _normal_quantiles(len(ordered))[keep]
    ax.plot(q, ordered[keep], ".", ms=2, label="sampled q(z)")
    ax.plot(q, q, "k--", label="N(0,1)")
    moment = metrics["aggregate_pooled_analytic"]
    ax.plot(q, moment["mean"] + moment["std"] * q, ":", color="tab:red", label="matched mean/std Gaussian")
    ax.set(title="Pooled Q-Q plot", xlabel="Standard normal quantile", ylabel="Empirical quantile")
    ax.legend(fontsize=8)
    ax = axes[0, 2]
    ax.plot(channel_ids, [r["mean"] for r in rows], "o-")
    ax.axhline(0, ls="--", color="k")
    ax.set(title="Aggregate mean per channel", xlabel="Channel", ylabel="Target = 0")
    ax = axes[1, 0]
    for key, label in (("std", "aggregate std"), ("variance_mu", "std of mu"), ("mean_posterior_variance", "RMS conditional std")):
        values = np.array([r[key] for r in rows])
        ax.plot(channel_ids, np.sqrt(values) if key != "std" else values, "o-", ms=3, label=label)
    ax.axhline(1, ls="--", color="k", label="prior std = 1")
    ax.set(title="Variance decomposition per channel", xlabel="Channel", ylabel="Standard deviation")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    for key in ("skewness", "excess_kurtosis"):
        ax.plot(channel_ids, [r[key] for r in rows], "o-", ms=3, label=key)
    ax.axhline(0, ls="--", color="k")
    ax.set(title="Analytic Gaussian-mixture moments", xlabel="Channel", ylabel="Gaussian target = 0")
    ax.legend()
    ax = axes[1, 2]
    ax.bar(channel_ids, [r["w2_to_standard_normal"] for r in rows])
    ax.set(title="Marginal W2 to N(0,1)", xlabel="Channel", ylabel="Distance (lower is closer)")
    for ax in axes.flat:
        ax.grid(alpha=0.15)
    _save_figure(fig, output_dir / "marginal_diagnostics.png")

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for ax, key, title in zip(axes[0], ("channel_correlation_mu", "channel_correlation_aggregate", "selected_correlation_aggregate"),
                              ("mu channel correlation (tokens pooled)", "q(z) channel correlation (analytic)", "q(z) fixed-coordinate correlation")):
        matrix = np.ma.masked_invalid(arrays[key])
        im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
        ax.set(title=title, xlabel="Channel" if "channel" in key else "Selected coordinate", ylabel="Same index")
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[1, 0]
    ax.plot(arrays["selected_covariance_eigenvalues"][::-1], ".-")
    ax.axhline(1, ls="--", color="k", label="prior eigenvalue = 1")
    ax.set(title="Selected-coordinate covariance spectrum", xlabel="Eigenvalue rank", ylabel="Eigenvalue", yscale="log")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    for label, value in metrics["spatial_neighbor_correlation_aggregate"].items():
        if value is not None:
            ax.plot(channel_ids, value, "o-", ms=3, label=label)
    ax.axhline(0, ls="--", color="k")
    ax.set(title="q(z) spatial neighbor correlation", xlabel="Channel", ylabel="Lag-1 correlation")
    if ax.lines and len(ax.lines) > 1:
        ax.legend()
    ax = axes[1, 2]
    values = arrays["coordinate_mu_variance"].reshape(-1)
    ax.hist(np.log10(np.maximum(values, 1e-15)), bins=70)
    threshold = metrics["active_units"]["variance_mu_threshold"]
    if threshold > 0:
        ax.axvline(math.log10(threshold), color="tab:red", ls="--", label=f"active threshold {threshold:g}")
        ax.legend()
    ax.set(title="Across-image variance of each mu coordinate", xlabel="log10 Var_x(mu_j), floor 1e-15", ylabel="Coordinates")
    _save_figure(fig, output_dir / "dependence_diagnostics.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    ax = axes[0, 0]
    projected = np.sort(arrays["projected_z"], axis=0)
    quantiles = _normal_quantiles(len(projected))
    for index in range(min(8, projected.shape[1])):
        ax.plot(quantiles, projected[:, index], alpha=0.7)
    ax.plot(quantiles, quantiles, "k--", label="N(0,1)")
    ax.set(title="Full-latent random-projection Q-Q (first 8)", xlabel="Standard normal quantile", ylabel="Projected z")
    ax.legend()
    ax = axes[0, 1]
    ax.plot(arrays["projection_w2"], ".-", label="q(z)")
    ax.plot(arrays["prior_projection_w2"], ".-", alpha=0.6, label="finite-N N(0,1) reference")
    ax.set(title="Projection W2 distances", xlabel="Random unit direction", ylabel="W2 to normal quantiles")
    ax.legend()
    ax = axes[1, 0]
    points = arrays["pca_selected_coordinates_z"]
    if points.shape[1] >= 2:
        reference = np.random.default_rng(2026).standard_normal(points.shape)
        ax.scatter(reference[:, 0], reference[:, 1], s=9, alpha=0.2, label="N(0,I) reference")
        ax.scatter(points[:, 0], points[:, 1], s=9, alpha=0.4, label="q(z)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Requires at least 2 latent dimensions", ha="center", transform=ax.transAxes)
    ax.set(title="Top covariance axes in selected-coordinate subspace", xlabel="Axis 1 (uncentered)", ylabel="Axis 2 (uncentered)")
    ax.set_aspect("equal", adjustable="datalim")
    ax = axes[1, 1]
    norms = arrays["per_image_sample_norm2_per_dim"]
    ax.hist(norms, bins=35, density=True, alpha=0.65, label="q(z), one draw / image")
    dims = metrics["dimensions_per_image"]
    low, high = stats.chi2.ppf([0.0001, 0.9999], dims) / dims
    x = np.linspace(low, high, 500)
    ax.plot(x, stats.chi2.pdf(x * dims, dims) * dims, "k--", label="N(0,I): chi-square(D) / D")
    ax.set(title="Full-latent squared radius / dimension", xlabel="||z||^2 / D", ylabel="Density")
    ax.legend(fontsize=8)
    _save_figure(fig, output_dir / "joint_diagnostics.png")

    columns = min(6, len(rows))
    fig, axes = plt.subplots(3, columns, figsize=(2.7 * columns, 8), squeeze=False, constrained_layout=True)
    for row, (key, label) in enumerate((("coordinate_mean", "E_x[mu]"),
                                       ("coordinate_aggregate_variance", "Std_q(z)"),
                                       ("coordinate_mu_variance", "Var_x(mu)"))):
        maps = arrays[key][:columns]
        if row == 1:
            maps = np.sqrt(maps)
        limit = max(float(np.abs(maps).max()), 1e-12)
        for col in range(columns):
            im = axes[row, col].imshow(maps[col], cmap="coolwarm" if row == 0 else "viridis",
                                       vmin=-limit if row == 0 else 0, vmax=limit)
            axes[row, col].set_title(f"{label}, channel {col}", fontsize=9)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        fig.colorbar(im, ax=list(axes[row]), shrink=0.75)
    _save_figure(fig, output_dir / "latent_statistic_maps.png")


def save_results(output_dir, metrics, arrays, metadata, identifiers):
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"metadata": metadata, "metrics": metrics}
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output_dir / "per_channel.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics["per_channel"][0]))
        writer.writeheader()
        writer.writerows(metrics["per_channel"])
    with (output_dir / "per_image.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "conditional_kl_nats", "sample_norm2_per_dim", "expected_norm2_per_dim"])
        writer.writerows(zip(identifiers, arrays["per_image_conditional_kl"], arrays["per_image_sample_norm2_per_dim"], arrays["per_image_expected_norm2_per_dim"]))
    np.savez_compressed(output_dir / "diagnostic_arrays.npz", **arrays)
    save_plots(metrics, arrays, output_dir)
    pooled = metrics["aggregate_pooled_analytic"]
    kl = metrics["conditional_kl"]
    sw = metrics["sliced_w2_full_latent"]
    proxy = metrics["aggregate_gaussian_moment_proxies"]
    summary_rows = [
        ("聚合后验 pooled mean / std", f"{pooled['mean']:.6g} / {pooled['std']:.6g}", "标准正态目标 0 / 1；整体接近不代表各通道或联合分布匹配"),
        ("pooled skewness / excess kurtosis", f"{pooled['skewness']:.6g} / {pooled['excess_kurtosis']:.6g}", "高斯目标均为 0；检查 Q–Q 图的尾部偏离"),
        ("平均条件 KL / latent 元素", f"{kl['mean_nats_per_element']:.6g} nats", "含 1/2；不是聚合后验 KL，不能仅据此判断 prior 匹配"),
        ("对角高斯矩匹配 KL / latent 元素", f"{proxy['diagonal_kl_nats_per_element']:.6g} nats", "只检查每个固定坐标的前两阶矩；忽略相关性与非高斯性"),
        ("选定坐标子空间高斯矩匹配 KL / 元素", f"{proxy['selected_subspace_kl_nats_per_element']:.6g} nats", f"{proxy['selected_subspace_dimensions']} 个固定 CHW 坐标，保留这些坐标的协方差；仍是高斯代理"),
        ("完整 latent 随机投影 SW2 近似", f"{sw['rms_w2']:.6g}", f"同样本量标准正态参考 {sw['prior_reference_rms_w2']:.6g}；越接近参考越好，无通用合格阈值"),
        ("聚合后验通道相关系数绝对值均值 / 最大值", f"{metrics['channel_correlation_aggregate']['mean_abs']} / {metrics['channel_correlation_aggregate']['max_abs']}", "标准独立正态的非对角相关为 0；这里将图片和空间位置合并为 token"),
        ("活跃坐标比例", f"{metrics['active_units']['active_fraction']:.2%}", f"Var_x(mu_j) > {metrics['active_units']['variance_mu_threshold']:g}；仅作描述，不应单独设为优化目标"),
    ]
    table = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td><td>{html.escape(note)}</td></tr>" for k, v, note in summary_rows)
    images = ""
    for name, title in (("marginal_diagnostics.png", "边缘分布与通道统计"), ("dependence_diagnostics.png", "相关性、协方差与活跃坐标"),
                         ("joint_diagnostics.png", "完整 latent 的投影与半径诊断"), ("latent_statistic_maps.png", "固定坐标的空间统计图（前 6 个通道）")):
        encoded = base64.b64encode((output_dir / name).read_bytes()).decode("ascii")
        images += f'<h2>{title}</h2><img src="data:image/png;base64,{encoded}" alt="{title}">'
    document = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>VAE 潜空间分布评估</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1250px;margin:auto;padding:28px;color:#172334}}
img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #dce3eb;padding:10px;text-align:left}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f8;padding:16px}}h2{{margin-top:2em}}.scroll{{overflow-x:auto}}</style>
<h1>VAE 潜空间分布评估</h1><p>{html.escape(metadata['checkpoint'])}</p>
<p>epoch={metadata['epoch']}，split={metadata['split']}，{metrics['num_images']} 张图片，latent CHW={metrics['latent_shape_chw']}。
评估 encode() 的原始潜变量；未应用 shift_factor / scale_factor。每张图片采样一次，随机种子固定。</p>
<div class="scroll"><table><tr><th>指标</th><th>结果</th><th>解释</th></tr>{table}</table></div>
<h2>判断方法</h2><p>先检查逐通道 mean≈0、std≈1，以及直方图与 Q–Q 图是否偏离标准正态；再看通道/空间相关性、固定坐标协方差谱和完整 latent 随机投影。
若 pooled 直方图接近高斯，但相关性较强或投影距离远高于同样本量 prior 参考，就不能称联合分布匹配 N(0,I)。
若方差主要来自后验噪声、mu 跨图片变化与条件 KL 都接近零，需结合重建质量排查后验坍塌。</p>
<pre>q(z) = (1/N) sum_i N(mu_i, diag(sigma_i²))
E_q[z_j] = E_x[mu_j]
Var_q(z_j) = Var_x(mu_j) + E_x[sigma_j²]
Cov_q(z) = Cov_x(mu) + E_x[diag(sigma²)]
E_x KL(q(z|x) || p(z)) = I_q(x;z) + KL(q(z) || p(z))</pre>
<p>前两阶矩、偏度、峰度和相关性由这组图片对应的高斯混合解析计算；W2/KS、投影与半径图使用采样值。
对角和子空间 Gaussian KL 是矩匹配代理，没有把真实聚合后验当作高斯，也没有直接估计高维混合密度的 KL。
SW2 使用完整 C×H×W latent 的随机单位方向，按排序后的投影与标准正态中点分位数计算，提供有限样本 prior 参考。
通道相关性将不同空间位置合并，固定坐标相关性则在图片之间比较同一个 CHW 坐标，二者含义不同。</p>
<p>空间 token 并非独立样本，因此本报告不输出依赖 IID 假设的正态性检验 p 值；不把 2D 图当作高维分布匹配的证明。
样本数和随机方向数有限，局部通过不能保证整体匹配；单次 checkpoint 也不能证明训练收敛。LDM 的有效表示允许弱 KL 下保留结构，分布更接近标准正态不等于重建或后续扩散效果一定更好。</p>
{images}
<h2>输出与复现</h2><p>同目录包含 metrics.json、per_channel.csv、per_image.csv、diagnostic_arrays.npz 和四张 PNG。
NPZ 中包括选定坐标索引、相关矩阵、逐坐标统计、投影数据与逐图 KL。空间图使用按样本计算的总体方差（分母 N）。</p>
<pre>{html.escape(metadata['command'])}</pre><p>训练划分 seed={metadata['split_seed']}，噪声/诊断 seed={metadata['seed']}，device={html.escape(metadata['device'])}，FP32 编码、FP64 解析统计。
每图噪声 seed=seed+该图片在所选划分中的序号，固定数据顺序下不受 batch size 影响。
当前只支持与 vae_eval.py 相同的 MM-CelebA-HQ 图片预处理。</p>
<p>方法参考：<a href="https://approximateinference.org/archives/2016/accepted/HoffmanJohnson2016.pdf">Hoffman &amp; Johnson, ELBO surgery (2016)</a>；
<a href="https://openaccess.thecvf.com/content_cvpr_2018/html/Deshpande_Generative_Modeling_Using_CVPR_2018_paper.html">Deshpande et al., Generative Modeling Using the Sliced Wasserstein Distance (CVPR 2018)</a>。</p>
</html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", "--ckpt", type=Path)
    source.add_argument("--run-dir", type=Path, help="Use the highest-epoch ckpt_epoch_*.pth in this run")
    parser.add_argument("--config", type=Path, help="Optional config override; required for nondefault bare weights")
    parser.add_argument("--model", choices=("vae_midattn", "vae_interleave"), default="vae_midattn")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=512, help="Fixed first N images of the chosen split")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42, help="Posterior/diagnostic seed; does not change the training split")
    parser.add_argument("--num-projections", type=int, default=64)
    parser.add_argument("--covariance-dims", type=int, default=128, help="Fixed CHW coordinates used for full-covariance diagnostics")
    parser.add_argument("--marginal-samples", type=int, default=100_000, help="Maximum sampled scalars per marginal diagnostic")
    parser.add_argument("--active-threshold", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, help="New directory only; default is inside the checkpoint's run")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if min(args.max_samples, args.batch_size, args.threads, args.num_projections, args.covariance_dims, args.marginal_samples) <= 0:
        parser.error("sample counts, batch size, threads and projection/covariance sizes must be positive")
    if args.max_samples < 2 or args.num_workers < 0 or args.seed < 0 or not math.isfinite(args.active_threshold) or args.active_threshold < 0:
        parser.error("Need at least two samples, nonnegative workers/seed, and a finite nonnegative active threshold")
    return args


def main():
    args = _parse_args()
    torch.set_num_threads(args.threads)
    if args.run_dir is not None:
        paths = list((args.run_dir / "ckpts").glob("ckpt_epoch_*.pth"))
        if not paths:
            raise FileNotFoundError(f"No checkpoints in {args.run_dir / 'ckpts'}")
        args.checkpoint = max(paths, key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    checkpoint = args.checkpoint.resolve()
    run_dir = checkpoint.parent.parent if checkpoint.parent.name == "ckpts" else checkpoint.parent
    output_dir = args.output_dir or run_dir / "latent_distribution_eval" / checkpoint.stem / f"{args.split}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose an accessible GPU or explicitly use --device cpu")
    model, config, model_kwargs, epoch = _load_vae(checkpoint, args.model, args.config, device)
    if str(config.data.get("dataset_name", "MM-CelebA-HQ")).lower() != "mm-celeba-hq":
        raise ValueError("This evaluator currently reproduces the MM-CelebA-HQ preprocessing/split only")
    image_size = tuple(config.data.img_size)
    factor = 2 ** (len(model_kwargs["ch_mult"]) - 1)
    if len(image_size) != 2 or any(size <= 0 or size % factor for size in image_size):
        raise ValueError(f"Image height/width must be positive multiples of {factor}")
    dataset_root = args.dataset_root or Path(config.path.dataset_path)
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root
    loader = _evaluation_loader(config, dataset_root, image_size, model_kwargs["in_channels"], args.split,
                                args.batch_size, args.num_workers, device)
    print(f"Checkpoint: {checkpoint}\nSplit: {args.split}; device: {device}; max images: {args.max_samples}", flush=True)
    mu, logvar, samples, identifiers = collect_latents(model, loader, device, args.max_samples, args.seed, not args.quiet)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Analyzing {len(mu)} latents of shape {tuple(mu.shape[1:])}...", flush=True)
    metrics, arrays = analyze_latents(mu, logvar, samples, seed=args.seed, num_projections=args.num_projections,
                                     covariance_dims=args.covariance_dims, marginal_samples=args.marginal_samples,
                                     active_threshold=args.active_threshold)
    metadata = {"created_at": datetime.now().astimezone().isoformat(), "checkpoint": str(checkpoint),
                "epoch": int(epoch) + 1 if epoch is not None else None, "model": args.model, "model_kwargs": model_kwargs,
                "dataset_root": str(dataset_root.resolve()), "split": args.split, "split_seed": int(config.seed),
                "seed": args.seed, "device": str(device), "torch_version": torch.__version__, "image_size": list(image_size),
                "space": "raw encode() output, before shift/scale", "normal_reference": "N(0,I)",
                "posterior_samples_per_image": 1, "split_size": len(loader.dataset),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "command": shlex.join([sys.executable, *sys.argv])}
    save_results(output_dir, metrics, arrays, metadata, identifiers)
    print(json.dumps({key: value for key, value in metrics.items() if key not in ("per_channel", "spatial_neighbor_correlation_aggregate")}, indent=2, allow_nan=False))
    print(f"Saved report: {(output_dir / 'report.html').resolve()}", flush=True)


if __name__ == "__main__":
    main()
