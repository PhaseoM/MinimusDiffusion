import torch
from torch import nn, Tensor
from einops import rearrange
from ..modules.uni_lib import ResidualBlock, DownSampling, UpSampling, ImageAttentionBlock


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        down_channels: int | None = None,
        num_groups: int = 1,
    ):
        super().__init__()
        self.model = nn.Sequential(
            ResidualBlock(in_channels, in_channels, num_groups),
            ResidualBlock(in_channels, in_channels, num_groups),
        )
        self.down = DownSampling(in_channels, down_channels) if down_channels is not None else nn.Identity()

    def forward(self, X):
        X = self.model(X)
        X = self.down(X)
        return X


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        init_channels: int,
        z_channels: int,
        ch_mult: list[int],
        num_groups: int = 1,
    ):
        super().__init__()
        # Initial Conv2d
        self.conv_in = nn.Conv2d(in_channels, init_channels, kernel_size=3, stride=1, padding=1)

        # Initialize Model
        num_stacks = len(ch_mult)
        in_ch_mult = [mult * init_channels for mult in ch_mult]
        out_ch_mult = in_ch_mult[1:] + [None]

        blocks = []
        for ch_in, ch_out in zip(in_ch_mult, out_ch_mult):
            blocks.append(EncoderBlock(ch_in, ch_out, num_groups))
        self.model = nn.Sequential(*blocks)

        end_ch = in_ch_mult[-1]

        # middle attn
        self.mid_attn = nn.Sequential(
            ResidualBlock(end_ch, end_ch, num_groups),
            ImageAttentionBlock(end_ch, num_groups),
            ResidualBlock(end_ch, end_ch, num_groups),
        )

        # Predict z_mean
        self.z_mean = nn.Sequential(
            nn.GroupNorm(num_groups, end_ch),
            nn.Conv2d(end_ch, z_channels, kernel_size=1, stride=1, padding=0),
        )

        # Scalar log_var of z
        self.z_logvar = nn.Parameter(torch.zeros(()))

    def forward(self, X):
        X = self.conv_in(X)
        X = self.model(X)
        X = self.mid_attn(X)
        X = self.z_mean(X)
        return X, self.z_logvar


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        up_channels: int | None = None,
        num_groups: int = 1,
    ):
        super().__init__()
        self.model = nn.Sequential(
            ResidualBlock(in_channels, in_channels, num_groups),
            ResidualBlock(in_channels, in_channels, num_groups),
        )
        self.up = UpSampling(in_channels, up_channels) if up_channels is not None else nn.Identity()

    def forward(self, X):
        X = self.model(X)
        X = self.up(X)
        return X


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        init_channels: int,
        x_channels: int,
        ch_mult: list[int],
        num_groups: int = 1,
    ):
        super().__init__()

        # compute reverse channels
        init_ch = init_channels * ch_mult[-1]

        in_ch_mult = ch_mult[::-1]
        in_ch_mult = [mult * init_channels for mult in in_ch_mult]
        out_ch_mult = in_ch_mult[1:] + [None]

        # Restore channels
        self.conv_in = nn.Conv2d(in_channels, init_ch, kernel_size=1, stride=1, padding=0)

        # middle attn
        self.mid_attn = nn.Sequential(
            ResidualBlock(init_ch, init_ch, num_groups),
            ImageAttentionBlock(init_ch, num_groups),
            ResidualBlock(init_ch, init_ch, num_groups),
        )

        # Initialize Model
        blocks = []
        for ch_in, ch_out in zip(in_ch_mult, out_ch_mult):
            blocks.append(DecoderBlock(ch_in, ch_out, num_groups))
        self.model = nn.Sequential(*blocks)

        # Predict x_mean
        end_ch = init_channels
        self.x_mean = nn.Sequential(
            nn.GroupNorm(num_groups, end_ch),
            nn.Conv2d(end_ch, x_channels, kernel_size=1, stride=1, padding=0),
        )

        # Scalar log_var of x
        self.x_logvar = nn.Parameter(torch.zeros(()))

    def forward(self, X):
        X = self.conv_in(X)
        X = self.mid_attn(X)
        X = self.model(X)
        X = self.x_mean(X)

        return X, self.x_logvar


class VAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        init_channels: int,
        z_channels: int,
        x_channels: int,
        ch_mult: list[int],
        num_groups: int = 1,
        shift_factor: float = 0,
        scale_factor: float = 1,
    ):
        super().__init__()

        self._encoder = Encoder(
            in_channels=in_channels,
            init_channels=init_channels,
            z_channels=z_channels,
            ch_mult=ch_mult,
            num_groups=num_groups,
        )

        self._decoder = Decoder(
            in_channels=z_channels,
            init_channels=init_channels,
            x_channels=x_channels,
            ch_mult=ch_mult,
            num_groups=num_groups,
        )

        self.shift_factor = shift_factor
        self.scale_factor = scale_factor

    def encode(self, X):
        return self._encoder(X)

    def decode(self, X):
        return self._decoder(X)

    def forward(self, X):
        z_mean, z_logvar = self.encode(X)
        z = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)

        # z = (z - self.shift_factor) / self.scale_factor

        x_mean, x_logvar = self.decode(z)

        # x_mean = x_mean / self.scale_factor + self.shift_factor
        return z_mean, z_logvar, x_mean, x_logvar
