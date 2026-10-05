import argparse
from omegaconf import OmegaConf
from pathlib import Path
from typing import cast
from dataclasses import dataclass, field


@dataclass
class MODELConfig:
    NAME: str = "unet"
    RESUME: str = ""
    NUM_LAYERS: int = 4
    NUM_CLASSES: int = 2
    BASE_CHANNELS: int = 64
    SKIP_PROCESS: str = "NULL"  # NULL, ONE_RES, DUAL_RES
    INCLUDE_BG: bool = False
    DROP_RATE: float = 0.0
    DROP_PATH_RATE: float = 0.1
    LABEL_SMOOTHING: float = 0.1


@dataclass
class PATHConfig:
    ROOT: str = "."
    PARENT: str = ".."
    CONFIGS: str = "${.ROOT}/configs"
    CURRENT_CONFIG: str = "${.CONFIGS}/default.yaml"
    DATABASE: str = "${.PARENT}/datasets"
    DATASET: str = "${.DATABASE}/ISIC2016"
    TRAIN_DATA: str = "${.DATASET}/ISBI2016_ISIC_Part1_Training_Data"
    TRAIN_GT: str = "${.DATASET}/ISBI2016_ISIC_Part1_Training_GroundTruth"
    TEST_DATA: str = "${.DATASET}/ISBI2016_ISIC_Part1_Test_Data"
    TEST_GT: str = "${.DATASET}/ISBI2016_ISIC_Part1_Test_GroundTruth"
    DATA: str = "${.ROOT}/data"
    DATA_DUMP: str = "${.DATA}/data_dump"
    DATA_EVAL: str = "${.DATA}/data_eval"
    LOG_INFO: str = "${.DATA}/log_info"
    CHECKPOINTS: str = "${.DATA}/checkpoints"


@dataclass
class AUGConfig:
    AUTO_AUGMENT: bool = False


@dataclass
class DATAConfig:
    NAME: str = "ISIC2016"
    BATCH_SIZE: int = 16
    IMAGE_SIZE: list = field(default_factory=lambda: [256, 256])
    MAIN_RANK: int = 0
    AMP: bool = True
    WORLD_SIZE: int = 4
    NUM_WORKERS: int = 4
    IMAGE_CHANNELS: int = 3
    AUG: AUGConfig = field(default_factory=AUGConfig)


@dataclass
class SCHEDULERConfig:
    NAME: str = "multistep"


@dataclass
class OPTIMConfig:
    NAME: str = "sgd"
    BASE_LR: float = 8e-2
    WARMUP_LR: float = 8e-4
    WARMUP_STEP: int = 100
    MOMENTUM: float = 0.9
    BETAS: list = field(default_factory=lambda: [0.9, 0.999])
    WEIGHT_DECAY: float = 1e-4
    DROPOUT: float = 0.1
    EPS: float = 1e-8
    SCHEDULER: SCHEDULERConfig = field(default_factory=SCHEDULERConfig)


@dataclass
class TRAINConfig:
    START_EPOCH: int = 0
    EPOCHS: int = 100
    SAVE_FREQ: int = 5


@dataclass
class TESTConfig:
    PARALLEL: bool = False


@dataclass
class Config:
    MODEL: MODELConfig = field(default_factory=MODELConfig)
    PATH: PATHConfig = field(default_factory=PATHConfig)
    DATA: DATAConfig = field(default_factory=DATAConfig)
    OPTIM: OPTIMConfig = field(default_factory=OPTIMConfig)
    TRAIN: TRAINConfig = field(default_factory=TRAINConfig)
    TEST: TESTConfig = field(default_factory=TESTConfig)
    SEED: int = 42
    TRAIN_MODE: bool = True
    EVAL_MODE: bool = False
    PROCESS_MODE: bool = False


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/default.yaml")
    args, unknow = parser.parse_known_args()

    return args, unknow


def load_config(is_resolve=True, is_struct=True, is_readonly=False) -> Config:
    args, unknown = parse_option()

    schema = OmegaConf.structured(Config)
    config_path = args.config
    if not ("/" in args.config or "\\" in args.config):
        config_path = schema.PATH.CONFIGS + "/" + args.config
    file_config = OmegaConf.load(config_path)
    cli_config = OmegaConf.from_cli(unknown)

    config = OmegaConf.merge(schema, file_config, cli_config)
    config.PATH.CURRENT_CONFIG = args.config

    missing = OmegaConf.missing_keys(config)
    if missing:
        raise ValueError(f"Missing config values: {missing}")

    if is_resolve:
        OmegaConf.resolve(config)
    OmegaConf.set_struct(config, is_struct)
    OmegaConf.set_readonly(config, is_readonly)

    return config
