import torch
from torch import nn
from models.modules.fm_lib import GaussianConditionalProbabilityPath


def vae_loss(
    x_true: torch.Tensor,
    z_mean: torch.Tensor,
    z_logvar: torch.Tensor,
    x_mean: torch.Tensor,
    x_logvar: torch.Tensor,
    beta: float,
):
    z_mean = z_mean.float()
    z_logvar = z_logvar.float()

    eps = 1e-6
    B = x_true.shape[0]
    # Recon Loss
    mse_term = ((x_true - x_mean).pow(2)) / torch.exp(x_logvar)
    confidence_term = x_logvar
    loss_recon = ((mse_term + confidence_term) / B).sum()

    # KL Loss
    loss_kl = ((z_mean.pow(2) + torch.exp(z_logvar) - z_logvar - 1) / B).sum()

    loss_vae = loss_recon + beta * loss_kl

    return loss_vae


def mseloss(theta: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.square(theta - target).mean()
