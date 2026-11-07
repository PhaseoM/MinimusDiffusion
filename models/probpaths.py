import torch
import torch.nn as nn
from typing import Optional, List, Type, Tuple, Dict

from .modules.fm_lib import Sampleable
from .modules.fm_lib import LinearAlpha, LinearBeta
from .modules.fm_lib import GaussianConditionalProbabilityPath


class GaussianLinearPath(GaussianConditionalProbabilityPath):
    def __init__(self, p_data: Sampleable, p_simple_shape: List[int]):
        super().__init__(p_data, p_simple_shape, LinearAlpha(), LinearBeta())
