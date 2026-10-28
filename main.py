import torch
import torchvision
import time
import pickle
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from data.build_dataloader import build_mmcelebahq_dataloader, build_mnist_dataloader
from data.data_process import data_process
from utils import unis
from utils.ddp import new_ddp, delete_ddp
from utils.config import load_config
from metrics.losses import vae_loss
from utils.logger import create_logger
from utils.checkpoint import save_ckpt, load_ckpt
from utils.lr_scheduler import warmup_cosine_decay
from models.autoencoders.vae_midattn import VAE
from engine import vae_engine


def main(conf):
    rank, local_rank, world_size, device = new_ddp(conf)

    seed = conf.seed + rank
    if conf.seed_everything:
        unis.seed_everything(seed)

    logger = create_logger(conf.path.log_info_dir, rank, name=conf.model.name)

    train_dataloader, val_dataloader, test_dataloader, train_sampler = build_mmcelebahq_dataloader(conf, rank)

    # Initialize Model with DDP
    model_kwargs = {
        "in_channels": conf.data.img_channels,
        "init_channels": conf.model.init_channels,
        "z_channels": conf.model.z_channels,
        "x_channels": conf.model.x_channels,
        "ch_mult": list(conf.model.ch_mult),
        "num_groups": conf.model.num_groups,
        "shift_factor": conf.model.shift_factor,
        "scale_factor": conf.model.scale_factor,
    }
    model = VAE(**model_kwargs).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Initialize optimizer, scheduler, scaler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=conf.optim.base_lr,
        betas=tuple(conf.optim.betas),
        weight_decay=conf.optim.weight_decay,
    )
    total_steps = conf.train.epochs * len(train_dataloader)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=warmup_cosine_decay(conf.optim.warmup_step, total_steps),
    )
    loss_scaler = torch.amp.GradScaler("cuda", enabled=conf.dist.amp)

    start_epoch = conf.train.start_epoch
    history = {"train": [], "val": []}
    train_loss_list, val_loss_list = [], []

    # resume from the ckpt
    if conf.auto_resume:
        start_epoch, history = load_ckpt(
            conf.path.resume_path,
            model,
            optimizer,
            lr_scheduler,
            loss_scaler,
            logger,
        )

    train_loss_list = history["train"]
    val_loss_list = history["val"]

    logger.info(f"{conf.model.name}: {conf.path.config_path}")
    logger.info("Start Training")
    for epoch in range(start_epoch, conf.train.epochs):
        train_sampler.set_epoch(epoch)

        epoch_start = time.perf_counter()

        train_loss = vae_engine.train(
            conf=conf,
            model=model,
            device=device,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            loss_scaler=loss_scaler,
            amp_enabled=conf.dist.amp,
        )
        val_loss = vae_engine.validate(
            conf=conf,
            model=model,
            device=device,
            dataloader=val_dataloader,
            amp_enabled=conf.dist.amp,
        )

        epoch_seconds = time.perf_counter() - epoch_start

        if rank == conf.dist.main_rank:
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            if (epoch + 1) % conf.save_freq == 0 or epoch + 1 == conf.train.epochs:
                save_ckpt(conf, epoch, model, optimizer, lr_scheduler, loss_scaler, model_kwargs, history, logger)

            logger.info(f"Epoch:{epoch + 1}/{conf.train.epochs}, {epoch_seconds:.2f}s/ep")
            logger.info(f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        dist.barrier()
    if rank == conf.dist.main_rank:
        test_loss = vae_engine.test(
            conf=conf,
            model=model.module,
            device=device,
            dataloader=test_dataloader,
        )
        unis.store_model_params(
            model=model.module,
            checkpoint_path=conf.path.ckpt_dir,
            model_name=conf.model.name,
        )
        data_dump_file = conf.path.data_dump_dir + "/" + f"{conf.model.name}.pkl"
        with open(data_dump_file, "wb") as f:
            pickle.dump(train_loss_list, f)
            pickle.dump(val_loss_list, f)
            pickle.dump(test_loss, f)
    delete_ddp()


if __name__ == "__main__":
    conf = load_config()
    if conf.mode.train:
        main(conf)
    if conf.mode.process:
        data_process(conf)
