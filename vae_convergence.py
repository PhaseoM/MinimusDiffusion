"""Read existing VAE checkpoints and write ONE self-contained convergence report.

No training, config, checkpoint or log file is changed. The destination must not
already exist. Intended for this project's MM-CelebA-HQ / vae_midattn model.

Example:
    python vae_convergence.py --run-dir results/VAE_20261029_1131 --device cuda:1

The HTML embeds charts, reconstruction previews, downloadable CSV/JSON and the
evaluation protocol. Original epoch histories and retrospective fixed-subset
measurements are explicitly distinguished.
"""

import argparse
import base64
import csv
import hashlib
import html
import importlib.util
import io
import json
import math
import os
import shlex
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
# Keep plotting caches out of the project and the user's existing config files.
_plot_cache = tempfile.TemporaryDirectory(prefix="vae_convergence_plot_")
os.environ["MPLCONFIGDIR"] = str(Path(_plot_cache.name) / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(Path(_plot_cache.name) / "cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from models.autoencoders.vae_midattn import VAE


ROOT = Path(__file__).resolve().parent
# metrics/__init__.py imports the TensorFlow evaluator; load only the loss file.
_loss_spec = importlib.util.spec_from_file_location("_vae_training_loss", ROOT / "metrics" / "losses.py")
_loss_module = importlib.util.module_from_spec(_loss_spec)
_loss_spec.loader.exec_module(_loss_module)
vae_loss = _loss_module.vae_loss


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def fixed_subsets(config, max_samples):
    """Reproduce build_mmcelebahq_dataloader's numeric order and seeded split."""
    if config["data"]["dataset_name"] != "MM-CelebA-HQ":
        raise ValueError("This report implements the MM-CelebA-HQ split only")
    data_root = Path(config["path"]["dataset_path"])
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    paths = sorted((p for p in (data_root / "images").glob("*.jpg") if p.is_file()),
                   key=lambda p: int(p.stem))
    val_size = int(len(paths) * config["data"]["val_ratio"])
    train_size = len(paths) - 2 * val_size
    if min(val_size, train_size) <= 0:
        raise ValueError("Empty train/validation split")
    permutation = torch.randperm(len(paths), generator=torch.Generator().manual_seed(config["seed"]))
    selections = {"train": permutation[:train_size],
                  "val": permutation[train_size:train_size + val_size]}
    transform = v2.Compose([
        v2.ToImage(), v2.Resize(tuple(config["data"]["img_size"]), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    subsets = {}
    for split, indices in selections.items():
        selected = [paths[i] for i in indices[:max_samples].tolist()]
        tensors = []
        for path in selected:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")).as_subclass(torch.Tensor))
        subsets[split] = (torch.stack(tensors), [int(p.stem) for p in selected])
    return subsets, {"total": len(paths), "train": train_size, "val": val_size, "test": val_size}


def loss_components(images, z_mean, z_logvar, reconstruction, x_logvar, beta):
    """Per-image terms, in the repository's sum-over-elements convention."""
    if x_logvar.numel() != 1 or z_logvar.numel() != 1:
        raise ValueError("This diagnostic expects scalar x_logvar and z_logvar")
    if reconstruction.shape != images.shape:
        raise ValueError("Reconstruction shape does not match input")
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    pixel_count = images[0].numel()
    latent_count = z_mean[0].numel()
    mse = (images - reconstruction).square().flatten(1).mean(1)
    kl_code = (z_mean.square() + z_logvar.exp() - z_logvar - 1).flatten(1).sum(1)
    positive_recon = pixel_count * mse / x_logvar.exp()
    logvar_term = torch.ones_like(mse) * pixel_count * x_logvar
    values = {
        "mse_raw": mse,
        "kl_standard": 0.5 * kl_code,
        "kl_standard_per_latent": 0.5 * kl_code / latent_count,
        "kl_weighted_code": beta * kl_code,
        "positive_recon_code": positive_recon,
        "logvar_term_code": logvar_term,
        "recon_code": positive_recon + logvar_term,
        "total_code": positive_recon + logvar_term + beta * kl_code,
        "mu_squared_per_latent": z_mean.square().flatten(1).mean(1),
    }
    for name, value in values.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite metric: {name}")
    return values


@torch.inference_mode()
def evaluate(model, images, image_ids, beta, device, batch_size, noise_seed):
    per_image = {}
    previews = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device)
        mu, z_logvar = model.encode(batch)
        # Per-image CPU seeds make the noise independent of batching and checkpoint.
        noise = torch.stack([
            torch.randn(mu.shape[1:], generator=torch.Generator().manual_seed(noise_seed + image_id))
            for image_id in image_ids[start:start + len(batch)]
        ]).to(device)
        reconstruction, x_logvar = model.decode(mu + (0.5 * z_logvar).exp() * noise)
        values = loss_components(batch, mu, z_logvar, reconstruction, x_logvar, beta)
        if start == 0:
            # Verify the decomposition against the actual training loss function.
            reference = vae_loss(batch, mu, z_logvar, reconstruction, x_logvar, beta)
            torch.testing.assert_close(values["total_code"].mean(), reference, rtol=2e-5, atol=0.02)
        for name, value in values.items():
            per_image.setdefault(name, []).extend(value.cpu().double().tolist())
        if start < 4:
            previews.append(reconstruction[:4 - start].cpu())
    result = {name: float(np.mean(value)) for name, value in per_image.items()}
    result["num_samples"] = len(images)
    result["mse_sem"] = float(np.std(per_image["mse_raw"], ddof=1) / math.sqrt(len(images))) if len(images) > 1 else 0.0
    result["psnr_from_mean_mse_db"] = 10 * math.log10(4 / max(result["mse_raw"], 1e-30))
    result["calibration_ratio"] = result["mse_raw"] / float(x_logvar.exp())
    return result, per_image, torch.cat(previews)


def figure_bytes(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def make_curves(records, history, pixels):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    epochs = [r["epoch"] for r in records]
    colors = {"train": "#3178a8", "val": "#d05b31"}

    def series(axis, key, split, label=None, **kwargs):
        axis.plot(epochs, [r[split][key] for r in records], "o-", markersize=4,
                  color=colors[split], label=label or f"fixed {split}", **kwargs)

    ax = axes[0, 0]
    for split in ("train", "val"):
        values = history[split]
        ax.plot(range(1, len(values) + 1), np.array(values) / pixels,
                color=colors[split], label=f"logged {split}")
    ax.plot(epochs, [r["x_logvar"] for r in records], "k.--", label="x_logvar")
    ax.set(title="A  Original total loss / pixel", ylabel="Code loss / pixel")

    ax = axes[0, 1]
    for split in ("train", "val"):
        series(ax, "mse_raw", split)
    ax.set(title="B  Raw reconstruction MSE", ylabel="Unclipped, input scale [-1, 1]")

    ax = axes[0, 2]
    for split in ("train", "val"):
        series(ax, "psnr_from_mean_mse_db", split)
    ax.set(title="C  PSNR from aggregate MSE", ylabel="10 log10(4 / MSE), dB")

    ax = axes[1, 0]
    for split in ("train", "val"):
        series(ax, "kl_standard_per_latent", split)
    ax.set(title="D  Standard KL / latent element", ylabel="Nats (includes factor 1/2)")

    ax = axes[1, 1]
    ax.plot(epochs, [r["x_variance"] for r in records], "k.--", label="exp(x_logvar)")
    for split in ("train", "val"):
        series(ax, "mse_raw", split)
    ax.set(title="E  Variance versus MSE", ylabel="Log scale", yscale="log")

    ax = axes[1, 2]
    for split in ("train", "val"):
        series(ax, "calibration_ratio", split)
    ax.axhline(1, color="#777", ls="--", label="variance optimum (train expectation)")
    ax.set(title="F  Variance calibration", ylabel="MSE / exp(x_logvar)", ylim=(0, None))

    ax = axes[2, 0]
    for key, label in (("positive_recon_code", "SSE / variance / pixel"),
                       ("kl_weighted_code", "beta * code KL / pixel")):
        ax.plot(epochs, [r["val"][key] / pixels for r in records], "o-", label=label)
    ax.set(title="G  Positive loss terms, fixed val", ylabel="Per input element (logvar excluded)")

    ax = axes[2, 1]
    ax.plot(epochs, [r["z_variance"] for r in records], "o-", label="exp(z_logvar)")
    ax.plot(epochs, [r["val"]["mu_squared_per_latent"] for r in records], "o-", label="E[mu^2], fixed val")
    ax.set(title="H  Posterior statistics", ylabel="Per latent element")

    ax = axes[2, 2]
    ax.plot(epochs, [max(r["learning_rate"], 1e-30) for r in records], "o-", color="#6646a1", label="checkpoint optimizer LR")
    ax.set(title="I  Learning rate at checkpoint", ylabel="Log scale", yscale="log")

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("VAE convergence diagnostics | dots = evaluated checkpoints; lines only guide the eye", fontsize=14)
    return figure_bytes(fig)


def make_previews(originals, predictions):
    count = min(4, len(originals))
    columns = [("Input", originals), *[(f"Epoch {epoch}", images) for epoch, images in predictions.items()]]
    fig, axes = plt.subplots(count, len(columns), figsize=(3 * len(columns), 3 * count), squeeze=False)
    for col, (label, images) in enumerate(columns):
        for row in range(count):
            axes[row, col].imshow(((images[row] + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy())
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(label)
    fig.suptitle("Fixed validation images + fixed posterior noise (display clipped only)")
    fig.tight_layout()
    return figure_bytes(fig)


def data_url(content, mime):
    if isinstance(content, str):
        content = content.encode("utf-8")
    return f"data:{mime};base64," + base64.b64encode(content).decode("ascii")


def write_report(output, payload, curves, previews):
    records, history = payload["checkpoints"], payload["original_history"]
    final = records[-1]
    reference = records[max(0, len(records) - 3)]
    val, train = final["val"], final["train"]
    delta = np.array(final["per_image"]["val"]["mse_raw"]) - np.array(reference["per_image"]["val"]["mse_raw"])
    delta_sem = float(delta.std(ddof=1) / math.sqrt(len(delta))) if len(delta) > 1 else 0.0
    improvement = 100 * (reference["val"]["mse_raw"] - val["mse_raw"]) / reference["val"]["mse_raw"]
    beta = final["beta"]
    pixels = payload["pixel_count"]
    flat_records = []
    for record in records:
        for split in ("train", "val"):
            flat_records.append({**{k: v for k, v in record.items() if k not in ("train", "val", "per_image")},
                                 "split": split, **record[split]})
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=list(flat_records[0]))
    writer.writeheader()
    writer.writerows(flat_records)
    raw_json = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
    rows = "".join(
        f"<tr><td>{r['epoch']}</td><td>{r['train']['mse_raw']:.6g}</td><td>{r['val']['mse_raw']:.6g}</td>"
        f"<td>{r['val']['psnr_from_mean_mse_db']:.3f}</td><td>{r['val']['kl_standard_per_latent']:.5f}</td>"
        f"<td>{r['x_logvar']:.5f}</td><td>{r['train']['calibration_ratio']:.4f}</td>"
        f"<td>{r['val']['calibration_ratio']:.4f}</td><td>{r['learning_rate']:.3g}</td></tr>" for r in records)
    lr_ratio = final["learning_rate"] / max(payload["config"]["optim"]["base_lr"], 1e-30)
    lr_note = ("末期学习率已接近零，曲线变平可能主要来自参数更新停止；不能据此宣称优化已经充分收敛。"
               if lr_ratio < 1e-3 else "末期学习率尚未接近零，可结合验证指标是否持续改善判断是否进入平台期。")
    calibration_note = ("当前训练子集的方差校准比例明显偏离 1；这提示 x_logvar 仍存在校准偏差，需结合更大训练样本确认。"
                        if not 0.8 <= train["calibration_ratio"] <= 1.2 else
                        "训练子集的方差校准比例接近 1，但这本身不能证明重建质量或整体优化已收敛。")
    document = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VAE 收敛诊断 — {html.escape(payload['run_dir'])}</title>
<style>
body{{font:16px/1.7 system-ui,sans-serif;max-width:1250px;margin:auto;padding:28px;color:#172334;background:#fafbfd}}
h1,h2{{line-height:1.3}} h2{{margin-top:2em}} .card{{padding:18px 24px;background:white;border:1px solid #dce3eb;border-radius:12px;margin:20px 0}}
img{{width:100%;height:auto}} table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}}
th,td{{padding:8px;border-bottom:1px solid #dce3eb;text-align:right}} th:first-child,td:first-child{{text-align:left}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#eff3f7;padding:14px;border-radius:6px}} a{{color:#216ba5}}
.muted{{color:#526479}} .scroll{{overflow-x:auto}} code{{background:#eff3f7;padding:2px 4px}}
</style>
<h1>VAE 收敛诊断</h1>
<p class="muted">{html.escape(payload['run_dir'])} · 生成时间 {html.escape(payload['created_at'])} · β = {beta:g}</p>
<div class="card"><p><strong>先看重建误差与 KL，再用方差校准和学习率解释平台期。总 loss 单独变平不足以证明收敛。</strong></p>
<p>最后一个 checkpoint 为 epoch {final['epoch']}：验证 MSE = <strong>{val['mse_raw']:.6g}</strong>，
PSNR = {val['psnr_from_mean_mse_db']:.3f} dB，标准 KL / latent element = {val['kl_standard_per_latent']:.5f}。
epoch {reference['epoch']} → {final['epoch']} 的验证 MSE 改善为 {improvement:.3f}%（负值表示变差）。</p>
<p>末期 LR = {final['learning_rate']:.4g}（初始 base_lr 的 {lr_ratio:.4g} 倍）。{lr_note}</p>
<p>exp(x_logvar) = {final['x_variance']:.6g}，训练/验证 MSE 与该方差之比分别为
{train['calibration_ratio']:.4f} / {val['calibration_ratio']:.4f}。{calibration_note}
这不等于需要删除 x_logvar 或立刻修改 β。</p></div>

<h2>九条诊断曲线</h2>
<p>A 中 train/val 来自原 checkpoint 保存的逐 epoch 历史；A 的 x_logvar 及其余面板只在现有 checkpoint 上有测量值。
每个点固定使用 {train['num_samples']} 张训练图片和 {val['num_samples']} 张验证图片。
连线仅辅助观察，没有补造缺失 epoch 的 MSE/KL。</p>
<img alt="九条诊断曲线：总损失、MSE、PSNR、KL、方差、校准比例、损失分量、后验统计、学习率" src="{data_url(curves, 'image/png')}">
<p><a download="vae_convergence_curves.png" href="{data_url(curves, 'image/png')}">下载曲线 PNG</a> ·
<a download="vae_convergence_metrics.csv" href="{data_url(csv_buffer.getvalue(), 'text/csv')}">下载指标 CSV</a> ·
<a download="vae_convergence_data.json" href="{data_url(raw_json, 'application/json')}">下载完整数据 JSON</a></p>

<h2>如何判断</h2>
<div class="scroll"><table><thead><tr><th>观察</th><th>含义 / 判断方法</th></tr></thead><tbody>
<tr><td>B/C：验证 MSE / PSNR</td><td>主要重建质量指标。固定样本上连续多个 checkpoint 改善很小，才支持重建进入平台期；查看下方图片确认细节。</td></tr>
<tr><td>D：标准 KL / latent</td><td>观察其与 MSE 的折中是否稳定。KL 不需要接近 0；KL 接近 0 且重建差，才需要进一步排查后验坍塌。</td></tr>
<tr><td>E/F：方差校准</td><td>训练期望下 MSE / exp(x_logvar) = 1 是共享方差的无正则最优条件。远小于 1 表示方差大于重建误差；验证集不必恰好为 1。</td></tr>
<tr><td>A/B/D：总 loss 与分量</td><td>若总 loss 继续降低，但 MSE/KL 基本不变、x_logvar 仍下降，主要变化可能来自方差项，而非重建质量。</td></tr>
<tr><td>B：train 与 val</td><td>训练 MSE 持续下降而验证 MSE 上升，提示过拟合。子集的绝对间距受图片难度影响，重点比较各自趋势。</td></tr>
<tr><td>I：学习率</td><td>LR 接近 0 时所有曲线容易变平。此时只能说本轮调度结束，不能单凭平台期认定已经充分优化。</td></tr>
</tbody></table></div>
<p>可操作的规则：选择固定验证集，用 3–5 个 checkpoint 的 MSE/PSNR 与 KL 趋势、图像细节和仍有实际更新的学习率共同判断。
例如把 MSE 相对改善 &lt; 1% 作为你任务的工程阈值，但它不是通用收敛定理。
若要选择生成模型用的 VAE，还应另查感知质量与后续扩散模型效果；这份报告只诊断当前 VAE 的目标与重建。</p>
<p>本报告使用同一批图片和相同后验噪声做配对比较：epoch {reference['epoch']} → {final['epoch']}
的逐图 MSE 差（后减前）平均为 {float(delta.mean()):.6g}，近似 95% 区间为
[{float(delta.mean()) - 1.96 * delta_sem:.6g}, {float(delta.mean()) + 1.96 * delta_sem:.6g}]。
该区间只描述这组样本的配对变化，不涵盖多随机种子训练或重复后验采样的不确定性。</p>

<h2>指标与当前损失的对应关系</h2>
<pre>D = {pixels}（每张图片的 C×H×W），L = {payload['latent_count']}（每张图片的 latent 元素数）
M = mean((x - x_hat)²)，s = x_logvar，v = exp(s)
K_code = sum(mu² + exp(z_logvar) - z_logvar - 1) = 2 × KL_standard
loss_code / D = M / v + s + β × K_code / D
∂(loss_code / D) / ∂s = 1 - M / v（固定网络输出、不计 weight decay 时）</pre>
<p>原代码的重建项与 KL 项都省略了 1/2；这里保留原损失尺度，同时单独报告含 1/2 的标准 KL。
连续高斯密度对应的负对数似然可以为负，因此 loss 的绝对值越来越大不等于发散，要看它的符号与具体分量。
原始 MSE 使用未裁剪的 [-1,1] 输入尺度；它是常见 [0,1] 尺度未裁剪 MSE 的 4 倍。
PSNR 由整体 MSE 计算，并非逐图 PSNR 的平均值。图像展示才裁剪到 [0,1]。</p>

<h2>Checkpoint 指标表</h2>
<div class="scroll"><table><thead><tr><th>Epoch</th><th>Train MSE</th><th>Val MSE</th><th>Val PSNR</th>
<th>Val KL / latent</th><th>x_logvar</th><th>Train MSE/v</th><th>Val MSE/v</th><th>LR</th></tr></thead>
<tbody>{rows}</tbody></table></div>

<h2>固定验证图片的重建变化</h2>
<img alt="同一组验证图片在不同 checkpoint 的采样重建" src="{data_url(previews, 'image/png')}">
<p>这是从 q(z|x) 采样后的重建，符合训练目标；并非使用 z_mean 的确定性重建。所有 checkpoint 对同一图片使用相同 ε。</p>

<h2>复现与限制</h2>
<ul><li>按训练代码的数字文件名排序、seed={payload['config']['seed']}、val_ratio={payload['config']['data']['val_ratio']} 重建数据划分，取各划分固定前 N 张图片；未使用 test 集挑选模型。</li>
<li>后验噪声使用 CPU generator，逐图 seed = {payload['noise_seed']} + image_id。每图一次采样；推理使用 FP32，训练日志可能来自 AMP，因此二者不应逐点完全一致。</li>
<li>原始 history 是整轮训练/验证记录；补算的 train 指标是在冻结 checkpoint 上评估的训练子集，不能当作该 epoch 的在线训练平均。</li>
<li>这次没有改训练设计或重新训练。只在当前代码与 checkpoint 架构一致、数据内容和排序未变的前提下，划分可复现。</li>
<li>checkpoint 未记录逐步梯度范数、clip 命中率或全部中间 epoch 的分量，无法事后精确恢复。本报告不编造这些数据。</li>
<li>上述方差等式直接由当前损失求导；实际 Adam、梯度裁剪、weight decay 和学习率衰减会影响其逼近速度。</li></ul>
<pre>{html.escape(payload['command'])}</pre>
<p class="muted">此 HTML 包含全部图像与可下载数据，不依赖外部网页或 CDN。输出路径必须不存在，脚本不会覆盖旧报告。</p>
<script id="diagnostic-data" type="application/json">{raw_json.replace('<', chr(92) + 'u003c')}</script>
</html>"""
    with output.open("x", encoding="utf-8") as handle:
        handle.write(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New HTML file (existing files are refused)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=128, help="Number of fixed images PER split")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--noise-seed", type=int, default=20261029)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.max_samples, args.batch_size, args.threads) <= 0 or args.noise_seed < 0:
        parser.error("sample count, batch size and threads must be positive; seed must be nonnegative")
    run_dir = args.run_dir.resolve()
    output = args.output or run_dir / f"vae_convergence_{datetime.now():%Y%m%d_%H%M%S}.html"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output parent does not exist: {output.parent}")
    paths = sorted((run_dir / "ckpts").glob("ckpt_epoch_*.pth"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {run_dir / 'ckpts'}")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose an accessible device or explicitly use --device cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    last = load_checkpoint(paths[-1])
    config, model_kwargs, history = last["config"], last["model_kwargs"], last["history"]
    del last
    subsets, sizes = fixed_subsets(config, args.max_samples)
    print(f"Fixed samples: train={len(subsets['train'][0])}, val={len(subsets['val'][0])}; device={device}", flush=True)
    model = VAE(**model_kwargs).to(device).eval()
    preview_indices = {0, (len(paths) - 1) // 2, len(paths) - 1}
    records, previews = [], {}
    for index, path in enumerate(paths):
        checkpoint = load_checkpoint(path)
        if checkpoint["model_kwargs"] != model_kwargs:
            raise ValueError(f"Model architecture differs: {path}")
        for key in ("data", "seed"):
            if checkpoint["config"][key] != config[key]:
                raise ValueError(f"Data protocol differs: {path}: {key}")
        if checkpoint["config"]["path"]["dataset_path"] != config["path"]["dataset_path"]:
            raise ValueError(f"Dataset path differs: {path}")
        if checkpoint["config"]["model"]["beta"] != config["model"]["beta"]:
            raise ValueError(f"beta differs: {path}; this report expects a fixed objective")
        model.load_state_dict(checkpoint["model"], strict=True)
        scalar = lambda suffix: float(next(v for k, v in checkpoint["model"].items() if k.endswith(suffix)))
        s, zs = scalar("x_logvar"), scalar("z_logvar")
        beta = float(config["model"]["beta"])
        record = {"epoch": int(checkpoint["epoch"]) + 1, "checkpoint": str(path),
                  "checkpoint_bytes": path.stat().st_size, "checkpoint_mtime_ns": path.stat().st_mtime_ns,
                  "beta": beta, "x_logvar": s, "x_variance": math.exp(s),
                  "z_logvar": zs, "z_variance": math.exp(zs),
                  "beta_times_x_variance": beta * math.exp(s),
                  "learning_rate": float(checkpoint["optimizer"]["param_groups"][0]["lr"]), "per_image": {}}
        del checkpoint
        for split, (images, image_ids) in subsets.items():
            stats, values, samples = evaluate(model, images, image_ids, beta, device, args.batch_size, args.noise_seed)
            record[split], record["per_image"][split] = stats, values
            if split == "val" and index in preview_indices:
                previews[record["epoch"]] = samples
        records.append(record)
        print(f"epoch {record['epoch']:3d}: train MSE={record['train']['mse_raw']:.7f}, "
              f"val MSE={record['val']['mse_raw']:.7f}, val KL/latent={record['val']['kl_standard_per_latent']:.5f}, "
              f"train MSE/var={record['train']['calibration_ratio']:.4f}, LR={record['learning_rate']:.4g}", flush=True)
    with torch.inference_mode():
        latent_count = model.encode(subsets["val"][0][:1].to(device))[0][0].numel()
    pixels = subsets["val"][0][0].numel()
    payload = {"created_at": datetime.now().astimezone().isoformat(), "run_dir": str(run_dir),
               "command": shlex.join([sys.executable, *sys.argv]), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "torch_version": torch.__version__, "device": str(device), "dtype": "float32",
               "config": config, "model_kwargs": model_kwargs, "split_sizes": sizes,
               "image_ids": {split: ids for split, (_, ids) in subsets.items()},
               "noise_seed": args.noise_seed, "posterior_samples_per_image": 1,
               "pixel_count": pixels, "latent_count": latent_count,
               "original_history": history, "checkpoints": records}
    write_report(output, payload, make_curves(records, history, pixels), make_previews(subsets["val"][0], previews))
    print(f"Report: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
