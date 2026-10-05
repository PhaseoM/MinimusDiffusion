import math
from config import Config


def warmup_linear_decay(warmup_steps, total_steps):

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return max(0, (total_steps - step) / (total_steps - warmup_steps))

    return lr_lambda


def warmup_cosine_decay(warmup_steps, total_steps, min_lr_ratio=0.0):

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            progress = min(progress, 1.0)

            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return lr_lambda
