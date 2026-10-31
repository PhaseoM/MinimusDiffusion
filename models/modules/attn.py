import torch
from torch import nn, Tensor
from einops import rearrange
from einops.layers.torch import Rearrange


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
