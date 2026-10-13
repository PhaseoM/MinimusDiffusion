import os
import torch


def load_ckpt(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    resume_path = config.path.resume_path
    logger.info(f"==============> Resuming from {resume_path}....................")
    if resume_path.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(resume_path, map_location="cpu", check_hash=True)
    else:
        checkpoint = torch.load(resume_path, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    logger.info(msg)
    max_metric = 0.0
    if "optimizer" in checkpoint and "lr_scheduler" in checkpoint and "epoch" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        config.train.start_epoch = checkpoint["epoch"] + 1
        if "scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["scaler"])
        logger.info(f"=> loaded successfully '{resume_path}' (epoch {checkpoint['epoch']})")
        if "max_metric" in checkpoint:
            max_metric = checkpoint["max_metric"]

    del checkpoint
    torch.cuda.empty_cache()
    return max_metric


def save_ckpt(config, epoch, model, max_metric, optimizer, lr_scheduler, loss_scaler, logger):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "scaler": loss_scaler.state_dict(),
        "max_metric": max_metric,
        "epoch": epoch,
        "config": config,
    }

    save_path = os.path.join(config.OUTPUT, f"ckpt_epoch_{epoch}.pth")
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def save_best_ckpt(config, epoch, model, max_metric, optimizer, lr_scheduler, loss_scaler, logger):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "scaler": loss_scaler.state_dict(),
        "max_metric": max_metric,
        "epoch": epoch,
        "config": config,
    }

    os.makedirs(config.path.ckpt_dir, exist_ok=True)
    save_path = os.path.join(config.path.ckpt_dir, "best_ckpt.pth")
    logger.info(f"{save_path} best checkpoint is saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} best checkpoint is saved !!!")
