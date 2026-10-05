import os
import torch


def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    logger.info(f"==============> Resuming from {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(config.MODEL.RESUME, map_location="cpu", check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    logger.info(msg)
    max_dice = 0.0
    if not config.EVAL_MODE and "optimizer" in checkpoint and "lr_scheduler" in checkpoint and "epoch" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        config.TRAIN.START_EPOCH = checkpoint["epoch"] + 1
        if "scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["scaler"])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if "max_dice" in checkpoint:
            max_dice = checkpoint["max_dice"]

    del checkpoint
    torch.cuda.empty_cache()
    return max_dice


def save_checkpoint(config, epoch, model, max_dice, optimizer, lr_scheduler, loss_scaler, logger):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "max_dice": max_dice,
        "scaler": loss_scaler.state_dict(),
        "epoch": epoch,
        "config": config,
    }

    save_path = os.path.join(config.OUTPUT, f"ckpt_epoch_{epoch}.pth")
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
