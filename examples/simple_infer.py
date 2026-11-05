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
TEXT_ENCODER_PATH = "pretrained/clip-vit-large-patch14"

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


@torch.inference_mode()
def main():
    # 避免给项目模块写入 __pycache__，运行缓存也统一放在 results/。
    sys.dont_write_bytecode = True
    cache = ROOT / "results" / "simple_infer" / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for key in ("HF_HOME", "TORCH_HOME", "CUDA_CACHE_PATH", "TRITON_CACHE_DIR"):
        os.environ[key] = str(cache / key.lower())

    from models.autoencoders.vae_midattn import VAE
    from models.diffusionmodels.dit import DiffusionTransformerWithFlowModel

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
    model_kwargs = dict(checkpoint["model_kwargs"])
    model_kwargs["text_encoder_path"] = str(project_path(TEXT_ENCODER_PATH))
    model = DiffusionTransformerWithFlowModel(**model_kwargs)
    model.load_state_dict(checkpoint["model"], strict=True)
    vae_path = VAE_CHECKPOINT or checkpoint["config"]["path"]["vae_ckpt_path"]
    del checkpoint
    model = model.to(device).eval().requires_grad_(False)

    print(f"加载 VAE：{vae_path}", flush=True)
    checkpoint = torch.load(project_path(vae_path), map_location="cpu", weights_only=True, mmap=True)
    vae_kwargs = checkpoint["model_kwargs"]
    if model_kwargs["c"] != vae_kwargs["z_channels"]:
        raise ValueError("LDM 潜通道数与 VAE 不匹配，请检查 VAE_CHECKPOINT")
    vae = VAE(**vae_kwargs)
    vae.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    vae = vae.to(device).eval().requires_grad_(False)

    generator = torch.Generator(device=device).manual_seed(SEED)
    size = int(model_kwargs["img_size"])
    z = torch.randn((NUM_IMAGES, model_kwargs["c"], size, size), device=device, generator=generator)
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
        "text_encoder_path": str(project_path(TEXT_ENCODER_PATH)),
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
