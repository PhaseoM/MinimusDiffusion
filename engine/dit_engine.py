import os
import uuid
import random
from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict

import torch
import torch.distributed as dist
from torch import nn
from tqdm import tqdm
from models.modules.neural_ode import GaussianConditionalProbabilityPath, CFGVectorFieldODE, EulerSimulator
from models.autoencoders.vae_midattn import VAE
from metrics.losses import mseloss


def drop_prompts(captions, prompt_drop_rate=0.1):
    if prompt_drop_rate <= 0:
        return list(captions)

    if prompt_drop_rate >= 1:
        return [""] * len(captions)

    return ["" if random.random() < prompt_drop_rate else caption for caption in captions]


def train(
    config,
    model,
    vae: VAE,
    prob_path: GaussianConditionalProbabilityPath,
    device,
    dataloader,
    optimizer,
    scheduler,
    loss_scaler,
    amp_enabled,
):
    model.train()
    train_loss = torch.zeros(1, device=device)
    sample_count = torch.zeros(1, device=device)

    for X, y in dataloader:
        X = X.to(device, non_blocking=True)
        y = drop_prompts(y, config.data.prompt_drop_rate)
        batch_size = X.size(0)

        optimizer.zero_grad(set_to_none=True)

        # sample from the latent space through vae encoder
        with torch.no_grad():
            z_mean, z_logvar = vae.encode(X)
            z_latent = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)

        # time_gen and add noise
        t = torch.rand(batch_size, device=device) * (1 - config.eps)
        x_latent = prob_path.sample_conditional_path(z_latent, t)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            ut_theta = model(x_latent, t, y)
            ut_target = prob_path.conditional_vector_field(x_latent, z_latent, t)
            loss = mseloss(ut_theta, ut_target)

        # loss backward with Gradient Scaling
        loss_scaler.scale(loss).backward()
        loss_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Gradient Clipping
        scale_before = loss_scaler.get_scale()
        loss_scaler.step(optimizer)
        loss_scaler.update()

        if loss_scaler.get_scale() >= scale_before:
            scheduler.step()

        train_loss += loss.detach() * batch_size
        sample_count += batch_size

    metrics = torch.cat([train_loss, sample_count])
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    global_train_loss = (metrics[0] / metrics[-1]).item()
    return global_train_loss


@torch.no_grad()
def validate(
    config,
    model,
    vae: VAE,
    prob_path: GaussianConditionalProbabilityPath,
    device,
    dataloader,
    amp_enabled,
):
    model.eval()

    val_loss = torch.zeros(1, device=device)
    sample_count = torch.zeros(1, device=device)

    for X, y in dataloader:
        X = X.to(device)
        y = list(y)
        batch_size = X.size(0)

        z_mean, z_logvar = vae.encode(X)
        z_latent = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)
        t = torch.rand(batch_size, device=device) * (1 - config.eps)
        x_latent = prob_path.sample_conditional_path(z_latent, t)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            ut_theta = model(x_latent, t, y)
            ut_target = prob_path.conditional_vector_field(x_latent, z_latent, t)
            loss = mseloss(ut_theta, ut_target)

        val_loss += loss.detach() * batch_size
        sample_count += batch_size

    metrics = torch.cat([val_loss, sample_count])
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    global_val_loss = (metrics[0] / metrics[-1]).item()
    return global_val_loss


@torch.no_grad()
def test(
    config,
    model,
    vae: VAE,
    prob_path: GaussianConditionalProbabilityPath,
    device,
    dataloader,
):
    model.eval()

    test_loss = 0
    size_n = len(dataloader.dataset)
    for X, y in dataloader:
        X = X.to(device)
        y = list(y)
        batch_size = X.size(0)

        z_mean, z_logvar = vae.encode(X)
        z_latent = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)
        t = torch.rand(batch_size, device=device) * (1 - config.eps)
        x_latent = prob_path.sample_conditional_path(z_latent, t)

        ut_theta = model(x_latent, t, y)
        ut_target = prob_path.conditional_vector_field(x_latent, z_latent, t)

        loss = mseloss(ut_theta, ut_target)

        test_loss += loss.detach() * batch_size

    test_loss = (test_loss / size_n).item()
    return test_loss
