import os
import psutil
import argparse
import torch.distributed as dist
from pathlib import Path
from datetime import datetime
from typing import cast, Any, Optional
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose


@dataclass
class AutoEncoderConfig:
    beta: float = 0.1
    init_channels: int = 64
    z_channels: int = 8
    x_channels: int = 3
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    shift_factor: float = 0.0
    scale_factor: float = 1.0
    num_groups: int = 1


@dataclass
class DiffusionConfig:
    _target_: str = "models.diffusions.dit.DiffusionTransformerWithFlowModel"
    img_size: int = 32
    channels: int = 8
    patch_size: int = 4
    num_layers: int = 12
    d_model: int = 768
    num_heads: int = 8
    ff_hidden_factor: int = 4
    final_dim: int = 16
    text_encoder_name: str = "clip-vit-large-patch14"
    text_encoder_path: str = "./pretrained/${.text_encoder_name}"


@dataclass
class ProbPathConfig:
    _target_: str = "models.probpaths.GaussianLinearPath"
    p_simple_shape: list[int] = field(default_factory=lambda: [8, 32, 32])


@dataclass
class ModelConfig:
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    autoencoder: AutoEncoderConfig = field(default_factory=AutoEncoderConfig)
    probpath: ProbPathConfig = field(default_factory=ProbPathConfig)
    time_eps: float = 0.001
    use_cache: bool = False
    autoencoder_path: str = "./results/VAE_mmchq_f8c8_betae-1/ckpts/ckpt_epoch_0100.pth"
    diffusion_path: str = "./results/DiT_d512_f8c8_l8p4/ckpts/ckpt_epoch_0400.pth"


@dataclass
class DatasetConfig:
    name: str = "mm-celeba-hq"
    use_cache: bool = False
    prompt_drop_rate: float = 0.1
    img_size: list[int] = field(default_factory=lambda: [256, 256])
    img_channels: int = 3
    dataset_dir: str = "./datasets"
    dataset_path: str = "${.dataset_dir}/${.name}"
    cache_path: str = "${.dataset_path}/caches"


@dataclass
class TrainConfig:
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ValConfig:
    dataset_ratio: float = 0.1
    val_epoch_freq: int = 1
    shuffle: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    training: bool = False


@dataclass
class TestConfig:
    dataset_ratio: float = 0.1
    shuffle: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    num_samples: int = 3000
    num_timesteps: int = 100
    guidance_scales: list[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 3.0, 5.0])


@dataclass
class DataConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    batch_size: int = 16
    seed: int = 42
    use_cache: bool = False
    train: TrainConfig = field(default_factory=TrainConfig)
    val: ValConfig = field(default_factory=ValConfig)
    test: TestConfig = field(default_factory=TestConfig)


@dataclass
class OptimizerConfig:
    _target_: str = "torch.optim.AdamW"
    lr: float = 1e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    weight_decay: float = 0.0


@dataclass
class SchedulerConfig:
    _target_: str = "utils.lr_scheduler.CosineLRScheduler"
    warmup_steps: int = 300
    warmup_start_ratio: float = 1e-2
    min_lr_ratio: float = 0.0
    last_epoch: int = -1


@dataclass
class AmpConfig:
    enabled: bool = True
    autocast_dtype: str = "float16"


@dataclass
class OptimConfig:
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    amp: AmpConfig = field(default_factory=AmpConfig)


@dataclass
class DistributedConfig:
    backend: str = "nccl"
    main_rank: int = 0


@dataclass
class CheckpointConfig:
    save_freq: int = 5
    ckpt_dir: str = "${result_dir}/ckpts"


@dataclass
class LoggingConfig:
    log_dir: str = "${result_dir}/logs"
    log_freq: int = 1
    log_level_primary: str = "INFO"
    log_level_secondary: str = "WARNING"
    all_ranks: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    name: str = "LDM"
    timestamp: str = "${now:%Y-%m-%d_%H-%M-%S}"
    exp_name: str = "${name}_${timestamp}"
    result_dir: str = "results/${exp_name}"
    seed: int = 42
    seed_everything: bool = True
    start_epoch: int = 0
    max_epochs: int = 100
    auto_resume: bool = False
    resume_path: Optional[str] = None
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# sync timestamp wth the ppid start_time
def get_timestamp() -> str:
    if "LOCAL_RANK" in os.environ:
        timestamp = psutil.Process(os.getppid()).create_time()
    else:
        timestamp = datetime.now().timestamp()

    return datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args, overrides = parser.parse_known_args()

    return args, overrides


def load_config(is_readonly=False) -> Config:
    args, overrides = parse_option()

    config_file = args.config.expanduser().resolve(strict=True)
    config_path = str(config_file.parent)
    config_name = config_file.name

    with initialize_config_dir(version_base=None, config_dir=config_path):
        config = compose(config_name=config_name, overrides=overrides)

    config.timestamp = get_timestamp()

    # check for missing required fields
    missing = OmegaConf.missing_keys(config)
    if missing:
        raise ValueError(f"Missing config values: {missing}")

    # resolve interpolations.
    OmegaConf.resolve(config)

    OmegaConf.set_readonly(config, is_readonly)

    return cast(Config, config)
