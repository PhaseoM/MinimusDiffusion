import os
import uuid
import random
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict

import torch
import torch.distributed as dist
from torch import nn
from models.modules.fm_lib import GaussianConditionalProbabilityPath, CFGVectorFieldODE, EulerSimulator
from models.autoencoders.vae_midattn import VAE
from metrics.losses import mseloss


class Trainer(ABC):
    """
    base Trainer for all Trainers
    """

    @abstractmethod
    def train(self):
        pass

    @torch.no_grad()
    @abstractmethod
    def validate(self):
        pass

    @torch.no_grad()
    @abstractmethod
    def test(self):
        pass


class LDMTrainer(Trainer):
    """
    Latent Diffusion Trainer
    """

    def __init__(
        self,
        device: torch.device,
        diffusion_model: nn.Module,
        autoencoder: Optional[nn.Module],
        prob_path: GaussianConditionalProbabilityPath,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        loss_scaler: torch.amp.GradScaler,
        # hyparam in config
        prompt_drop_rate: float = 0.1,
        time_eps: float = 0.001,
        amp_enabled: bool = True,
    ):
        self.device = device
        self.diffusion_model = diffusion_model
        self.autoencoder = autoencoder
        self.prob_path = prob_path
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_scaler = loss_scaler

        self.prompt_drop_rate = prompt_drop_rate
        self.time_eps = time_eps
        self.amp_enabled = amp_enabled

    def _drop_prompts(self, prompts, empty_prompt):
        if isinstance(prompts, torch.Tensor):
            drop_mask = torch.rand(prompts.shape[0], device=self.device) < self.prompt_drop_rate
            droped_prompts = torch.where(drop_mask[:, None], empty_prompt[None, :], prompts)
        else:
            droped_prompts = [empty_prompt if random.random() < self.prompt_drop_rate else prompt for prompt in prompts]
        return droped_prompts

    def train(self, dataloader, use_cache: bool = False):
        self.diffusion_model.train()
        train_loss = torch.zeros(1, device=self.device)
        sample_count = torch.zeros(1, device=self.device)
        if not use_cache:
            empty_prompt = ""
        else:
            cache_dataset = dataloader.dataset.dataset
            z_logvar = torch.tensor(cache_dataset.z_logvar, dtype=torch.float32, device=self.device)
            empty_prompt = torch.tensor(cache_dataset.empty_text_feature, dtype=torch.float32, device=self.device)

        for X, prompt in dataloader:
            X = X.to(self.device, non_blocking=True)
            if use_cache:
                prompt = prompt.to(self.device, non_blocking=True)
            prompt = self._drop_prompts(prompt, empty_prompt)
            batch_size = X.size(0)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp_enabled):
                # sample from the latent space through vae encoder
                with torch.no_grad():
                    if not use_cache:
                        z_mean, z_logvar = self.autoencoder.encode(X)
                    else:
                        z_mean = X
                    z_latent = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)

                # time_gen and add noise
                t = torch.rand(batch_size, device=self.device, dtype=torch.float32) * (1 - self.time_eps)
                x_latent = self.prob_path.sample_conditional_path(z_latent, t)

                ut_theta = self.diffusion_model(x_latent, t, prompt)
                ut_target = self.prob_path.conditional_vector_field(x_latent, z_latent, t)
                loss = mseloss(ut_theta, ut_target)

            # loss backward with Gradient Scaling
            self.loss_scaler.scale(loss).backward()
            self.loss_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), max_norm=1.0)

            # Gradient Clipping
            scale_before = self.loss_scaler.get_scale()
            self.loss_scaler.step(self.optimizer)
            self.loss_scaler.update()

            if self.loss_scaler.get_scale() >= scale_before:
                self.scheduler.step()

            train_loss += loss.detach() * batch_size
            sample_count += batch_size

        metrics = torch.cat([train_loss, sample_count])
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        global_train_loss = (metrics[0] / metrics[-1]).item()
        return global_train_loss

    @torch.no_grad()
    def validate(self, dataloader, use_cache: bool = False):
        self.diffusion_model.eval()

        val_loss = torch.zeros(1, device=self.device)
        sample_count = torch.zeros(1, device=self.device)
        if use_cache:
            cache_dataset = dataloader.dataset.dataset
            z_logvar = torch.tensor(cache_dataset.z_logvar, dtype=torch.float32, device=self.device)

        for X, prompt in dataloader:
            X = X.to(self.device)
            if use_cache:
                prompt = prompt.to(self.device, non_blocking=True)
            batch_size = X.size(0)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp_enabled):
                if not use_cache:
                    z_mean, z_logvar = self.autoencoder.encode(X)
                else:
                    z_mean = X
                z_latent = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)

                t = torch.rand(batch_size, device=self.device, dtype=torch.float32) * (1 - self.time_eps)
                x_latent = self.prob_path.sample_conditional_path(z_latent, t)

                ut_theta = self.diffusion_model(x_latent, t, prompt)
                ut_target = self.prob_path.conditional_vector_field(x_latent, z_latent, t)
                loss = mseloss(ut_theta, ut_target)

            val_loss += loss.detach() * batch_size
            sample_count += batch_size

        metrics = torch.cat([val_loss, sample_count])
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        global_val_loss = (metrics[0] / metrics[-1]).item()
        return global_val_loss

    @torch.no_grad()
    def test(self, dataloader, use_cache: bool = False):
        self.diffusion_model.eval()

        test_loss = 0
        size_n = len(dataloader.dataset)

        if use_cache:
            cache_dataset = dataloader.dataset.dataset
            z_logvar = torch.tensor(cache_dataset.z_logvar, dtype=torch.float32, device=self.device)

        for X, prompt in dataloader:
            X = X.to(self.device)
            if use_cache:
                prompt = prompt.to(self.device, non_blocking=True)
            batch_size = X.size(0)

            if not use_cache:
                z_mean, z_logvar = self.autoencoder.encode(X)
            else:
                z_mean = X
            z_latent = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)
            t = torch.rand(batch_size, device=self.device) * (1 - self.time_eps)
            x_latent = self.prob_path.sample_conditional_path(z_latent, t)

            ut_theta = self.diffusion_model(x_latent, t, prompt)
            ut_target = self.prob_path.conditional_vector_field(x_latent, z_latent, t)

            loss = mseloss(ut_theta, ut_target)

            test_loss += loss.detach() * batch_size

        test_loss = (test_loss / size_n).item()
        return test_loss
