import os
import sys
import numpy as np
import tomllib
import argparse
import random
import functools
from torch import nn
from pathlib import Path
from functools import wraps
from datetime import datetime
from utils.config import Config


def seed_everything(seed: int = 42, is_deterministic: bool = False):
    import torch

    # os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    if is_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text)

    def flush(self):
        for f in self.files:
            f.flush()


# feature: output > Terminal & @filename
def tee_output(filename, mode="w"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with open(filename, mode=mode) as f:
                sys.stdout = Tee(sys.__stdout__, f)
                res = func(*args, **kwargs)
                sys.stdout = sys.__stdout__
            return res

        return wrapper

    return decorator


def trapz(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return (0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])).sum()


def store_model_params(model, checkpoint_path, model_name):
    import torch

    checkpoint_path = Path(checkpoint_path)
    model_name = Path(model_name)
    model_dump_path = checkpoint_path / model_name
    model_dump_path.mkdir(parents=True, exist_ok=True)

    random_token = "".join(random.choices("0123456789abcdefABCDEF", k=7))
    time_token = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_file = model_dump_path / f"{time_token}.pth"
    torch.save(model.state_dict(), str(model_file))


def model_size_b(model: nn.Module, logger, MiB: int = 1024**2) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    Args:
        - model: self-explanatory
    Returns:
        - size: model size in bytes
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()

    logger.info(f"Training model with size: {size / MiB:.3f} MiB")
    return size
