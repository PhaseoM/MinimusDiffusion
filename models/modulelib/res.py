import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.project = nn.Identity()
        if in_channels != out_channels:
            self.project = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.model = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
                padding=padding,
            ),
            nn.SiLU(),
        )

    def forward(self, X):
        identity = self.project(X)
        X = identity + self.model(X)
        return X


class ResidualStack(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.model = nn.Sequential(*self._res_stack(num_layers, in_channels, out_channels, kernel_size, padding))

    def _res_stack(self, num_layers, in_channels, out_channels, kernel_size=3, padding=1):
        res_stack = []
        res_stack.append(ResidualBlock(in_channels, out_channels, kernel_size, padding))
        for i in range(1, num_layers):
            res_stack.append(ResidualBlock(out_channels, out_channels, kernel_size, padding))
        return res_stack

    def forward(self, X):
        return self.model(X)
