import torch
from torch import nn, Tensor
from einops import rearrange


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
        self.model = nn.Sequential(*self._res_stack(num_layers, in_channels, out_channels))

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


class AttentionBlock(nn.Module):
    def __init__(self, in_channels: int, num_groups: int = 1):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.w_q = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.w_k = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.w_v = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.w_o = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, X: Tensor) -> Tensor:
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
        return attn


class DownSamlping(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, X):
        pad = (0, 1, 0, 1)
        X = nn.functional.pad(X, pad, mode="constant", value=0)
        X = self.conv(X)
        return X


class UpSampling(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        X = nn.functional.interpolate(X, scale_factor=2.0, mode="nearest")
        X = self.conv(X)
        return X


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        downsample_channels: int | None = None,
        num_groups: int = 1,
    ):
        super().__init__()
        self.model = nn.Sequential(
            ResidualBlock(in_channels, in_channels, num_groups),
            ResidualBlock(in_channels, in_channels, num_groups),
            AttentionBlock(in_channels, num_groups),
        )
        self.down = DownSamlping(in_channels, downsample_channels) if downsample_channels is not None else nn.Identity()

    def forward(self, X):
        X = self.model(X)
        X = self.down(X)
        return X


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        init_ch: int,
        ch_mult: list[int],
        z_ch: int,
        num_groups: int = 1,
    ):
        super().__init__()
        # Initial Conv2d
        self.conv_in = nn.Conv2d(in_channels, init_ch, kernel_size=3, stride=1, padding=1)

        # Initialize Model
        num_stacks = len(ch_mult)
        in_ch_mult = ch_mult * init_ch
        out_ch_mult = ch_mult[1:] * init_ch + [None]

        blocks = []
        for ch_in, ch_out in zip(in_ch_mult, out_ch_mult):
            blocks.append(EncoderBlock(ch_in, ch_out, num_groups))
        self.model = nn.ModuleList(blocks)

        # Predict Z_mean
        end_ch = in_ch_mult[-1]
        self.z_mean = nn.Sequential(
            nn.GroupNorm(num_groups, end_ch),
            nn.Conv2d(end_ch, z_ch, kernel_size=1, stride=1, padding=0),
        )

        # Scalar log_var
        self.logvar = nn.Parameter(torch.zeros())

    def forward(self, X):
        X = self.conv_in(X)
        X = self.model(X)
        X = self.z_mean(X)
        return X, self.logvar


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

    def forward(self, X):
        pass


class VAE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()

    def forward(self, X):
        pass
