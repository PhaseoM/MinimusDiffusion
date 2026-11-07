import os
import torch
from omegaconf import OmegaConf
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.config import Config


def load_ckpt(resume_path, model, optimizer, lr_scheduler, loss_scaler, logger):
    logger.info(f"==============> Resuming from {resume_path}....................")

    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)

    raw_model = model.module if isinstance(model, DDP) else model
    msg = raw_model.load_state_dict(checkpoint["model"], strict=False)
    logger.info(msg)

    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    loss_scaler.load_state_dict(checkpoint["scaler"])

    logger.info(f"=> loaded successfully '{resume_path}' (epoch {checkpoint['epoch']})")

    return checkpoint["epoch"] + 1, checkpoint["history"]


def save_ckpt(config: Config, epoch, model, optimizer, lr_scheduler, loss_scaler, history, logger):
    raw_model = model.module if isinstance(model, DDP) else model

    save_state = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "scaler": loss_scaler.state_dict(),
        "epoch": epoch,
        "config": OmegaConf.to_container(config, resolve=True),
        "history": history,
    }

    save_path = Path(config.checkpoint.ckpt_dir) / f"ckpt_epoch_{epoch + 1:04d}.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"{save_path} saving...")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !")

    return save_path
