import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 1):
        super().__init__()
        self.project = nn.Identity()
        if in_channels != out_channels:
            self.project = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.model = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.Conv2d(kernel_size=3, in_channels=in_channels, out_channels=out_channels, padding=1),
            nn.SiLU(),
        )

    def forward(self, X):
        identity = self.project(X)
        X = identity + self.model(X)
        return X


class ResidualStack(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        num_groups: int = 1,
    ):
        super().__init__()
        self.model = nn.Sequential(*self._res_stack(num_layers, in_channels, out_channels, num_groups))

    def _res_stack(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        num_groups: int = 1,
    ):
        res_stack = []
        res_stack.append(ResidualBlock(in_channels, out_channels, num_groups))
        for i in range(1, num_layers):
            res_stack.append(ResidualBlock(out_channels, out_channels, num_groups))
        return res_stack

    def forward(self, X):
        return self.model(X)


class DownSampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, X):
        pad = (0, 1, 0, 1)
        X = nn.functional.pad(X, pad, mode="constant", value=0)
        X = self.conv(X)
        return X


class UpSampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        X = nn.functional.interpolate(X, scale_factor=2.0, mode="nearest")
        X = self.conv(X)
        return X
