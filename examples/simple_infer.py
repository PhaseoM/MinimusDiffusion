import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm


CHECKPOINT = "./results/DiT_20261104_211157/ckpts/ckpt_epoch_0300.pth"
VAE_CHECKPOINT = None  # None：使用 LDM 检查点记录的 VAE；手动指定时须与训练匹配
TEXT_ENCODER_PATH = None  # None：使用 LDM 检查点记录的文本编码器路径

PROMPT = "This man appears to be . The person is with mouth slightly open, bags under eyes, big nose, eyeglasses, pale skin, pointy nose and wears ."
NEGATIVE_PROMPT = ""
GUIDANCE_SCALE = 3
NUM_STEPS = 300
SAMPLER = "euler"
NUM_IMAGES = 4
SEED = 42
DEVICE = "cuda:0"
USE_FP16 = False


ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def get_vae_checkpoint(checkpoint):
    if VAE_CHECKPOINT:
        return VAE_CHECKPOINT
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    # Current config, intermediate migration, then the original config.
    for path in (
        model_config.get("autoencoder_path"),
        model_config.get("diffusion", {}).get("autoencoder_path"),
        config.get("path", {}).get("vae_ckpt_path"),
    ):
        if path:
            return path
    raise ValueError("LDM 检查点未记录 VAE 路径，请设置 VAE_CHECKPOINT")


def get_diffusion_kwargs(checkpoint):
    if "model_kwargs" in checkpoint:
        kwargs = dict(checkpoint["model_kwargs"])
    else:
        kwargs = dict(checkpoint["config"]["model"]["diffusion"])
    for key in ("_target_", "_recursive_", "_convert_", "_partial_"):
        kwargs.pop(key, None)
    if "c" in kwargs:
        channels = kwargs.pop("c")
        if "channels" in kwargs and kwargs["channels"] != channels:
            raise ValueError("LDM 检查点的 c 与 channels 不一致")
        kwargs["channels"] = channels
    return kwargs


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


@torch.inference_mode()
def main():
    # 避免给项目模块写入 __pycache__，运行缓存也统一放在 results/。
    sys.dont_write_bytecode = True
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    cache = ROOT / "results" / "simple_infer" / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for key in ("HF_HOME", "TORCH_HOME", "CUDA_CACHE_PATH", "TRITON_CACHE_DIR"):
        os.environ[key] = str(cache / key.lower())

    from models.autoencoders.vae_midattn import VAE
    from models.diffusions.dit import DiffusionTransformerWithFlowModel

    if NUM_STEPS <= 0 or NUM_IMAGES <= 0:
        raise ValueError("NUM_STEPS 和 NUM_IMAGES 必须大于 0")
    if SAMPLER not in ("euler", "heun"):
        raise ValueError('SAMPLER 只能是 "euler" 或 "heun"')
    if not math.isfinite(GUIDANCE_SCALE) or GUIDANCE_SCALE < 0:
        raise ValueError("GUIDANCE_SCALE 必须是非负有限数")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用，请把 DEVICE 改成 "cpu"')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.manual_seed(SEED)

    print(f"加载 LDM：{CHECKPOINT}；设备：{device}", flush=True)
    checkpoint = torch.load(project_path(CHECKPOINT), map_location="cpu", weights_only=True, mmap=True)
    model_kwargs = get_diffusion_kwargs(checkpoint)
    text_encoder_path = TEXT_ENCODER_PATH or model_kwargs.get("text_encoder_path", "pretrained/clip-vit-large-patch14")
    model_kwargs["text_encoder_path"] = str(project_path(text_encoder_path))
    model = DiffusionTransformerWithFlowModel(**model_kwargs)
    model.load_state_dict(checkpoint["model"], strict=True)
    vae_path = get_vae_checkpoint(checkpoint)
    del checkpoint
    model = model.to(device).eval().requires_grad_(False)

    print(f"加载 VAE：{vae_path}", flush=True)
    checkpoint = torch.load(project_path(vae_path), map_location="cpu", weights_only=True, mmap=True)
    vae_kwargs = get_vae_kwargs(checkpoint)
    if model_kwargs["channels"] != vae_kwargs["z_channels"]:
        raise ValueError("LDM 潜通道数与 VAE 不匹配，请检查 VAE_CHECKPOINT")
    vae = VAE(**vae_kwargs)
    vae.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    vae = vae.to(device).eval().requires_grad_(False)

    generator = torch.Generator(device=device).manual_seed(SEED)
    size = int(model_kwargs["img_size"])
    z = torch.randn((NUM_IMAGES, model_kwargs["channels"], size, size), device=device, generator=generator)
    times = torch.linspace(0, 1, NUM_STEPS + 1, device=device)

    with torch.autocast(device.type, dtype=torch.float16, enabled=USE_FP16 and device.type == "cuda"):
        # 文本只编码一次；下面等价于模型 forward 中的文本条件和时间条件相加。
        text = model.prompt_proj(model.prompt_embedder([PROMPT, NEGATIVE_PROMPT]))

        def velocity(state, time):
            patches = model.patchifier(state)
            time_embedding = model.time_embedder(time.expand(NUM_IMAGES))

            def predict(embedding):
                return model.depatchifier(model.dit(patches, time_embedding + embedding)).float()

            if GUIDANCE_SCALE == 0:
                return predict(text[1:2])
            conditional = predict(text[0:1])
            if GUIDANCE_SCALE == 1:
                return conditional
            unconditional = predict(text[1:2])
            return unconditional + GUIDANCE_SCALE * (conditional - unconditional)

        for step in tqdm(range(NUM_STEPS), desc=f"采样 ({SAMPLER}, CFG={GUIDANCE_SCALE:g})"):
            dt = times[step + 1] - times[step]
            first = velocity(z, times[step])
            predicted = z + dt * first
            z = predicted if SAMPLER == "euler" else z + dt * (first + velocity(predicted, times[step + 1])) / 2
        if not torch.isfinite(z).all():
            raise FloatingPointError("采样产生 NaN/Inf，请检查模型或降低 GUIDANCE_SCALE")
        # 训练中直接使用原始 VAE 潜变量，解码时也不添加额外的 shift/scale。
        decoded, _ = vae.decode(z)

    if not torch.isfinite(decoded).all():
        raise FloatingPointError("VAE 解码产生 NaN/Inf")
    images = ((decoded.float() + 1) / 2).clamp(0, 1).cpu()
    output = ROOT / "results" / "simple_infer" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    for index, image in enumerate(images):
        save_image(image, output / f"image_{index:03d}.png")
    save_image(images, output / "grid.png", nrow=math.ceil(math.sqrt(NUM_IMAGES)))
    settings = {
        "checkpoint": str(project_path(CHECKPOINT)),
        "vae_checkpoint": str(project_path(vae_path)),
        "text_encoder_path": str(project_path(text_encoder_path)),
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "guidance_scale": GUIDANCE_SCALE,
        "num_steps": NUM_STEPS,
        "sampler": SAMPLER,
        "num_images": NUM_IMAGES,
        "seed": SEED,
        "device": str(device),
        "fp16": USE_FP16 and device.type == "cuda",
        "image_size": list(images.shape[-2:]),
    }
    (output / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已保存 {NUM_IMAGES} 张图片、grid.png 和 settings.json：{output}")
    return output


if __name__ == "__main__":
    main()
