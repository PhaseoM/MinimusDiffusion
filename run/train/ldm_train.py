import torch
import torchvision
import time
import pickle
import torch.distributed as dist
from pathlib import Path
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from data.build_dataloader import build_dataloaders
from data.data_process import data_process
from utils import unis
from utils.ddp import build_ddp, delete_ddp
from utils.config import load_config
from metrics.losses import vae_loss
from utils.logger import create_logger
from utils.checkpoint import save_ckpt, load_ckpt
from utils.lr_scheduler import warmup_cosine_decay
from models.autoencoders.vae_midattn import VAE
from models.diffusionmodels.dit import DiffusionTransformerWithFlowModel
from models.modules.neural_ode import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from ..trainer import LDMTrainer


def main(config):
    rank, local_rank, world_size, device = build_ddp()

    seed = config.seed + rank
    if config.seed_everything:
        unis.seed_everything(seed)

    logger = create_logger(config.path.logs_dir, rank, name=config.model.name)
    unis.results_mkdirs(config, logger)

    train_dataloader, val_dataloader, test_dataloader, train_sampler = build_dataloaders(config, rank)

    # load vae model
    autoencoder_ckpt = torch.load(config.path.vae_ckpt_path, map_location="cpu", weights_only=True)
    autoencoder_model_kwargs = autoencoder_ckpt["model_kwargs"]
    if not config.data.use_cache:
        autoencoder_model = VAE(**autoencoder_model_kwargs).to(device)
        autoencoder_model.load_state_dict(autoencoder_ckpt["model"], strict=True)
        autoencoder_model.requires_grad_(False)
        autoencoder_model.eval()
    else:
        autoencoder_model = None
    del autoencoder_ckpt

    # build Gaussion prob_path
    img_size = config.model.z_img_size
    c = autoencoder_model_kwargs["z_channels"]
    prob_path = GaussianConditionalProbabilityPath(
        p_data=None,
        p_simple_shape=[c, img_size, img_size],
        alpha=LinearAlpha(),
        beta=LinearBeta(),
    ).to(device)

    # Initialize Model with DDP
    dit_model_kwargs = {
        "img_size": img_size,
        "patch_size": config.model.patch_size,
        "num_layers": config.model.num_layers,
        "c": c,
        "d_model": config.model.d_model,
        "num_heads": config.model.num_heads,
        "final_dim": config.model.final_dim,
        "text_encoder_name": config.model.text_encoder_name,
        "text_encoder_path": config.path.text_encoder_path,
    }
    dit_model = DiffusionTransformerWithFlowModel(**dit_model_kwargs).to(device)
    model = DDP(dit_model, device_ids=[local_rank], output_device=local_rank)

    # Initialize optimizer, scheduler, scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.base_lr,
        betas=tuple(config.optim.betas),
        weight_decay=config.optim.weight_decay,
        fused=True,
    )
    total_steps = config.train.epochs * len(train_dataloader)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=warmup_cosine_decay(config.optim.warmup_step, total_steps),
    )
    loss_scaler = torch.amp.GradScaler("cuda", enabled=config.dist.amp)

    start_epoch = config.train.start_epoch
    history = {"train": [], "val": []}
    train_loss_list, val_loss_list = [], []

    # resume from the ckpt
    if config.auto_resume:
        start_epoch, history = load_ckpt(
            config.path.resume_path,
            model,
            optimizer,
            lr_scheduler,
            loss_scaler,
            logger,
        )
    # build LDM Trainer
    trainer = LDMTrainer(
        device=device,
        model=model,
        autoencoder=autoencoder_model,
        prob_path=prob_path,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        loss_scaler=loss_scaler,
        prompt_drop_rate=config.data.prompt_drop_rate,
        time_eps=config.eps,
        amp_enabled=config.dist.amp,
    )
    train_loss_list = history["train"]
    val_loss_list = history["val"]

    logger.info(f"{config.model.name}: {config.path.config_path}")
    logger.info("Start Training")
    for epoch in range(start_epoch, config.train.epochs):
        train_sampler.set_epoch(epoch)

        epoch_start = time.perf_counter()

        train_loss = trainer.train(train_dataloader, config.data.use_cache)
        if config.mode.val:
            val_loss = trainer.validate(val_dataloader, config.data.use_cache)

        epoch_seconds = time.perf_counter() - epoch_start

        if rank == config.dist.main_rank:
            train_loss_list.append(train_loss)
            if config.mode.val:
                val_loss_list.append(val_loss)

            logger.info(f"Epoch:{epoch + 1}/{config.train.epochs}, {epoch_seconds:.2f}s/ep")
            if config.mode.val:
                logger.info(f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                logger.info(f"train_loss={train_loss:.4f}")

            if (epoch + 1) % config.save_freq == 0 or epoch + 1 == config.train.epochs:
                save_ckpt(config, epoch, model, optimizer, lr_scheduler, loss_scaler, dit_model_kwargs, history, logger)

        dist.barrier()
    delete_ddp()


if __name__ == "__main__":
    config = load_config()

    main(config)
