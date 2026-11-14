import torch
import torchvision
import time
import pickle
import torch.distributed as dist
from pathlib import Path
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from hydra.utils import instantiate
from data.build_dataloader import build_dataloaders
from data.data_process import data_process
from utils import unis
from utils.ddp import build_ddp, delete_ddp
from utils.config import Config, load_config
from metrics.losses import vae_loss
from utils.logger import create_logger
from utils.checkpoint import save_ckpt, load_ckpt
from utils.lr_scheduler import ConstantLRScheduler, CosineLRScheduler
from models.autoencoders.vae_midattn import VAE
from models.diffusions.dit import DiffusionTransformerWithFlowModel
from models.modules.fm_lib import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from ..trainer import LDMTrainer


def main(config: Config):
    rank, local_rank, world_size, device = build_ddp(config.distributed)

    start_epoch, max_epochs = config.start_epoch, config.max_epochs
    history = {"train": [], "val": []}
    train_loss_list, val_loss_list = [], []

    seed = config.seed + rank
    if config.seed_everything:
        unis.seed_everything(seed)

    logger = create_logger(config.logging.log_dir, rank, name=config.name)

    train_dataloader, val_dataloader, test_dataloader, train_sampler = build_dataloaders(config.data)

    # load vae model
    autoencoder_ckpt = torch.load(config.model.autoencoder_path, map_location="cpu", weights_only=True)
    autoencoder_model_kwargs = autoencoder_ckpt["model_kwargs"]
    if not config.model.use_cache:
        autoencoder_model = VAE(**autoencoder_model_kwargs).to(device)
        autoencoder_model.load_state_dict(autoencoder_ckpt["model"], strict=True)
        autoencoder_model.requires_grad_(False)
        autoencoder_model.eval()
    else:
        autoencoder_model = None
    del autoencoder_ckpt

    # build Gaussion prob_path
    prob_path = instantiate(config.model.probpath, p_data=None).to(device)

    # Initialize Model with DDP
    diffusion_model = instantiate(config.model.diffusion).to(device)
    diffusion_model = DDP(diffusion_model, device_ids=[local_rank], output_device=local_rank)

    # Instantiate optimizer
    optimizer = instantiate(
        config.optim.optimizer,
        diffusion_model.parameters(),
    )
    # Instantiate lr_scheduler
    lr_scheduler = instantiate(
        config.optim.scheduler,
        optimizer=optimizer,
        total_steps=max_epochs * len(train_dataloader),
    )
    # Initialize loss_scaler
    loss_scaler = torch.amp.GradScaler(enabled=config.optim.amp.enabled)

    # resume from the ckpt
    if config.auto_resume:
        start_epoch, history = load_ckpt(
            config.resume_path,
            diffusion_model,
            optimizer,
            lr_scheduler,
            loss_scaler,
            logger,
        )
    train_loss_list, val_loss_list = history["train"], history["val"]

    # build LDM Trainer
    trainer = LDMTrainer(
        diffusion_model=diffusion_model,
        autoencoder=autoencoder_model,
        prob_path=prob_path,
        time_eps=config.model.time_eps,
        device=device,
        prompt_drop_rate=config.data.dataset.prompt_drop_rate,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        loss_scaler=loss_scaler,
        amp_enabled=config.optim.amp.enabled,
    )

    logger.info(f"{config.exp_name}: ")
    logger.info("Start Training")
    for epoch in range(start_epoch, max_epochs):
        train_sampler.set_epoch(epoch)

        epoch_start = time.perf_counter()

        train_loss = trainer.train(train_dataloader, config.data.dataset.use_cache)

        do_val = config.data.val.training and (epoch + 1) % config.data.val.val_epoch_freq == 0
        if do_val:
            val_loss = trainer.validate(val_dataloader, config.data.dataset.use_cache)

        epoch_seconds = time.perf_counter() - epoch_start

        if rank == config.distributed.main_rank:
            logger.info(f"Epoch:{epoch + 1}/{max_epochs}, {epoch_seconds:.2f}s/ep")
            train_loss_list.append(train_loss)

            if do_val:
                val_loss_list.append(val_loss)
                logger.info(f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                logger.info(f"train_loss={train_loss:.4f}")

            if (epoch + 1) % config.checkpoint.save_freq == 0 or epoch + 1 == max_epochs:
                save_ckpt(config, epoch, diffusion_model, optimizer, lr_scheduler, loss_scaler, history, logger)

        dist.barrier()
    delete_ddp()


if __name__ == "__main__":
    config = load_config()

    if config.auto_resume and not Path(config.resume_path).exists():
        raise FileNotFoundError(f"{config.resume_path} can't be found in the results.")

    main(config)
