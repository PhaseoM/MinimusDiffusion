import torch
import pickle
import torch.distributed as dist
from torch import nn
from tqdm import tqdm
from data.data_load import create_isic2016_dataloader
from data.data_process import data_process
from utils import unis
from utils.ddp import new_ddp, delete_ddp
from utils.config import load_config
from utils.logger import create_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.lr_scheduler import warmup_cosine_decay
from models.autoencoders.vae import VAE, vae_loss
from engine import vae_engine


def main(conf):
    rank, local_rank, world_size, device = new_ddp(conf)

    logger = create_logger(conf.PATH.LOG_INFO, rank, name=conf.MODEL.NAME)

    train_dataloader, test_dataloader, train_sampler = create_isic2016_dataloader(conf, rank)

    seed = conf.SEED + rank
    unis.seed_everything(seed)

    loss_scaler = torch.amp.GradScaler("cuda", enabled=conf.DATA.AMP)

    model = VAE(conf).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=conf.OPTIM.BASE_LR,
        betas=tuple(conf.OPTIM.BETAS),
        weight_decay=conf.OPTIM.WEIGHT_DECAY,
    )
    total_steps = conf.TRAIN.EPOCHS * len(train_dataloader)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=warmup_cosine_decay(conf.OPTIM.WARMUP_STEP, total_steps),
    )

    train_loss_list, val_loss_list = [], []

    # pbar = (
    #     tqdm(range(conf.TRAIN.START_EPOCH, conf.TRAIN.EPOCHS), dynamic_ncols=True)
    #     if rank == conf.DATA.MAIN_RANK
    #     else range(conf.TRAIN.START_EPOCH, conf.TRAIN.EPOCHS)
    # )
    logger.info(f"{conf.MODEL.NAME}: {conf.PATH.CURRENT_CONFIG}")
    logger.info("Start Training")
    for epoch in range(conf.TRAIN.START_EPOCH, conf.TRAIN.EPOCHS):
        train_sampler.set_epoch(epoch)
        train_loss, train_iou, train_dice = vae_engine.train(
            conf=conf,
            device=device,
            loss_scaler=loss_scaler,
            amp_enabled=conf.DATA.AMP,
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
        )
        dist.barrier()
        val_loss = vae_engine.validate(conf, device, test_dataloader, model.module)
        dist.barrier()
        if rank == conf.DATA.MAIN_RANK:
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            # pbar.set_postfix(
            #     epoch=f"{epoch + 1}/{conf.epochs}",
            #     train_loss=f"{train_loss:.4f}",
            #     train_dice=f"{train_dice:.4f}",
            #     test_loss=f"{test_loss:.4f}",
            #     test_dice=f"{test_dice:.4f}",
            # )
            logger.info(f"train_loss={train_loss:.4f},test_loss={val_loss:.4f},")
        dist.barrier()
    if rank == conf.DATA.MAIN_RANK:
        unis.store_model_params(
            model=model.module,
            checkpoint_path=conf.PATH.CHECKPOINTS,
            model_name=conf.MODEL.NAME,
        )

        data_dump_file = conf.PATH.DATA_DUMP + "/" + f"{conf.MODEL.NAME}.pkl"
        with open(data_dump_file, "wb") as f:
            pickle.dump(train_loss_list, f)
            pickle.dump(val_loss_list, f)
    delete_ddp()


if __name__ == "__main__":
    conf = load_config()
    if conf.TRAIN_MODE:
        main(conf)
    if conf.PROCESS_MODE:
        data_process(conf)
