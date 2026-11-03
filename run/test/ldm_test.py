from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.func import vmap, jacrev
from einops import rearrange
from einops.layers.torch import Rearrange
from models.modules.neural_ode import CFGVectorFieldODE, EulerSimulator, EulerMaruyamaSimulator


def ldm_eval(
    config,
    model: nn.Module,
    num_timesteps: int,
):
    pass
