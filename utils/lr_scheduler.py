import math
from abc import ABC, abstractmethod
from bisect import bisect_right
from collections.abc import Sequence
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR


class BaseScheduler(LRScheduler, ABC):
    """Base scheduler with linear warmup.

    Args:
        optimizer (Optimizer): Optimizer to schedule.
        total_steps (int): Total planned optimizer updates.
        warmup_steps (int): Number of warmup updates.
        warmup_start_ratio (float): Starting LR relative to base LR.
        min_lr_ratio (float): Minimum LR ratio after warmup.
        last_epoch (int): Scheduler index; -1 for initialization.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        warmup_start_ratio: float = 0.0,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):

        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0 <= warmup_steps < total_steps:
            raise ValueError("Require 0 <= warmup_steps < total_steps")
        if not 0.0 <= warmup_start_ratio <= 1.0:
            raise ValueError("warmup_start_ratio must be in [0, 1]")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.warmup_start_ratio = warmup_start_ratio
        self.min_lr_ratio = min_lr_ratio

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = min(max(self.last_epoch, 0), self.total_steps)

        if step < self.warmup_steps:
            progress = step / self.warmup_steps
            ratio = self.warmup_start_ratio + (1.0 - self.warmup_start_ratio) * progress
        else:
            ratio = self.decay_ratio(step)

        return [base_lr * ratio for base_lr in self.base_lrs]

    # normalize the step after warmup
    def decay_progress(self, step: int) -> float:
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return min(max(progress, 0.0), 1.0)

    @abstractmethod
    def decay_ratio(self, step: int) -> float:
        raise NotImplementedError


class ConstantLRScheduler(BaseScheduler):
    """Linear warmup followed by a constant learning rate.

    Inherits all arguments from BaseScheduler.
    """

    def decay_ratio(self, step: int) -> float:
        return 1.0


class LinearLRScheduler(BaseScheduler):
    """Linear warmup followed by linear decay.

    Inherits all arguments from BaseScheduler.
    """

    def decay_ratio(self, step: int) -> float:
        progress = self.decay_progress(step)
        factor = 1.0 - progress

        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * factor


class CosineLRScheduler(BaseScheduler):
    """Linear warmup followed by cosine decay.

    Inherits all arguments from BaseScheduler.
    """

    def decay_ratio(self, step: int) -> float:
        progress = self.decay_progress(step)
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * factor


class MultiStepLRScheduler(BaseScheduler):
    """Linear warmup followed by decay at specified steps.

    Args:
        milestones (Sequence[int]): Absolute decay steps after warmup.
        gamma (float): LR multiplier at each milestone.

    Other arguments are inherited from BaseScheduler.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        milestones: Sequence[int],
        gamma: float = 0.1,
        warmup_steps: int = 0,
        warmup_start_ratio: float = 0.0,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        self.milestones = tuple(milestones)

        if any(a >= b for a, b in zip(self.milestones, self.milestones[1:])):
            raise ValueError("milestones must be strictly increasing")
        if any(not warmup_steps < s <= total_steps for s in self.milestones):
            raise ValueError("Require warmup_steps < milestone <= total_steps")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")

        self.gamma = gamma

        super().__init__(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            warmup_start_ratio=warmup_start_ratio,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
        )

    def decay_ratio(self, step: int) -> float:
        passed = bisect_right(self.milestones, step)

        return max(self.min_lr_ratio, self.gamma**passed)
