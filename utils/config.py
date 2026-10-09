import argparse
from omegaconf import OmegaConf
from pathlib import Path


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True)
    parser.add_argument("--config", required=True)
    # parser.add_argument("--dataset", required=True)
    args, unknow = parser.parse_known_args()

    return args, unknow


def load_config(is_resolve=True, is_struct=True, is_readonly=False):
    args, unknown = parse_option()

    schema_path = args.schema
    config_path = args.config

    schema = OmegaConf.load(schema_path)
    file_config = OmegaConf.load(config_path)
    cli_config = OmegaConf.from_cli(unknown)

    # structure the schema and check for invalid args in unknown
    OmegaConf.set_struct(schema, is_struct)

    # merge all the args
    config = OmegaConf.merge(schema, file_config, cli_config)

    config.PATH.CURRENT_CONFIG = args.config

    # check for missing required fields
    missing = OmegaConf.missing_keys(config)
    if missing:
        raise ValueError(f"Missing config values: {missing}")

    # resolve interpolations.
    if is_resolve:
        OmegaConf.resolve(config)

    OmegaConf.set_readonly(config, is_readonly)

    return config
