from typing import Optional, List, Type, Tuple, Dict

import torch
from torch import nn, Tensor
from einops import rearrange
from einops.layers.torch import Rearrange


class MLP(nn.Module):
    def __init__(self, dims: List[int], activation: Type[nn.Module] = nn.SiLU, final_init: bool = False):
        super().__init__()
        mlp = []
        for i in range(len(dims) - 1):
            mlp.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mlp.append(activation())
        self.model = torch.nn.Sequential(*mlp)

        if final_init:
            nn.init.zeros_(self.model[-1].weight)
            nn.init.zeros_(self.model[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: (... d)
        Returns:
            - x: (... d)
        """
        return self.model(x)


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


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0

        self.qkv_project = nn.Linear(d_model, d_model * 3)
        self.split_heads = Rearrange("b l (h d) -> b h l d", h=num_heads)
        self.merge_heads = Rearrange("b h l d -> b l (h d)", h=num_heads)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: (b l d)
        Returns:
            - x: (b l d)
        """
        # Compute queries, keys, and values (b l (h d))
        q, k, v = self.qkv_project(x).chunk(3, dim=-1)

        # split heads (b h l d)
        q, k, v = map(self.split_heads, (q, k, v))

        # Compute attention
        attn = nn.functional.scaled_dot_product_attention(q, k, v)

        # merge heads (b l (h d))
        x = self.merge_heads(attn)

        return self.out(x)


class ImageAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, num_groups: int = 1):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.w_q = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.w_k = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.w_v = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.w_o = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, X: Tensor) -> Tensor:
        identity = X
        X = self.norm(X)
        q = self.w_q(X)
        k = self.w_k(X)
        v = self.w_v(X)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        attn = nn.functional.scaled_dot_product_attention(q, k, v)

        attn = self.w_o(rearrange(attn, "b 1 (h w) c -> b c h w", b=b, c=c, h=h, w=w))
        return identity + attn


class SequentialTranspose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, X):
        return X.transpose(self.dim0, self.dim1)


class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean
