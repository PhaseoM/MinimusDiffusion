import torch
import pickle
import torch.nn.functional as nnF
import torch.distributed as dist
from torch import nn
from tqdm import tqdm
from config import Config, load_config
from data.data_load import create_isic2016_dataloader
from data.data_process import data_process
from utils import unis
from utils.ddp import new_ddp, delete_ddp
from utils.logger import create_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import iou_sum, dice_sum
from models.unet import UNet
from lr_scheduler import warmup_cosine_decay


def train_one_epoch(conf: Config, device, amp_enabled, loss_scaler, dataloader, model, optimizer, scheduler, criterion):
    model.train()
    train_loss = torch.zeros(1, device=device)
    iou = torch.zeros(1, device=device)
    dice = torch.zeros(1, device=device)
    sample_count = torch.zeros(1, device=device)

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            logits = model(X)
            loss = criterion(logits, y)

        batch_size = y.size(0)
        pred = torch.argmax(logits, dim=1)

        loss_scaler.scale(loss).backward()
        loss_scaler.step(optimizer)
        loss_scaler.update()
        scheduler.step()

        train_loss += loss.detach() * batch_size
        iou += iou_sum(
            num_classes=conf.MODEL.NUM_CLASSES,
            pred=pred,
            mask=y,
            include_background=conf.MODEL.INCLUDE_BG,
            eps=conf.OPTIM.EPS,
        )
        dice += dice_sum(
            num_classes=conf.MODEL.NUM_CLASSES,
            pred=pred,
            mask=y,
            include_background=conf.MODEL.INCLUDE_BG,
            eps=conf.OPTIM.EPS,
        )
        sample_count += batch_size

        # if batch % 100 == 0:
        #     loss_, current = loss.item(), dataloader.batch_size * batch + len(X)
        #     print(f"loss: {loss_:>7f} [{current:>5d}/{size_n:>5d}]")
    metrics = torch.cat([train_loss, iou, dice, sample_count])
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    global_loss = (metrics[0] / metrics[-1]).item()
    global_iou = (metrics[1] / metrics[-1]).item()
    global_dice = (metrics[2] / metrics[-1]).item()
    return global_loss, global_iou, global_dice


@torch.no_grad()
def test(conf: Config, device, dataloader, model, criterion):
    model.eval()

    test_loss, iou, dice = 0, 0, 0
    size_n = len(dataloader.dataset)
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        test_loss += loss.item() * X.size(0)

        pred = torch.argmax(logits, dim=1)
        iou += iou_sum(
            num_classes=conf.MODEL.NUM_CLASSES,
            pred=pred,
            mask=y,
            include_background=conf.MODEL.INCLUDE_BG,
            eps=conf.OPTIM.EPS,
        )
        dice += dice_sum(
            num_classes=conf.MODEL.NUM_CLASSES,
            pred=pred,
            mask=y,
            include_background=conf.MODEL.INCLUDE_BG,
            eps=conf.OPTIM.EPS,
        )
    test_loss = test_loss / size_n
    iou = (iou / size_n).item()
    dice = (dice / size_n).item()
    return test_loss, iou, dice


def main(conf: Config):
    rank, local_rank, world_size, device = new_ddp(conf)

    logger = create_logger(conf.PATH.LOG_INFO, rank, name=conf.MODEL.NAME)

    train_dataloader, test_dataloader, train_sampler = create_isic2016_dataloader(conf, rank)

    seed = conf.SEED + rank
    unis.seed_everything(seed)

    loss_scaler = torch.amp.GradScaler("cuda", enabled=conf.DATA.AMP)

    model = UNet(conf).to(device)
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
    criterion_train = nn.CrossEntropyLoss(label_smoothing=conf.MODEL.LABEL_SMOOTHING)
    criterion_test = nn.CrossEntropyLoss()

    train_loss_list, train_iou_list, train_dice_list = [], [], []
    test_loss_list, test_iou_list, test_dice_list = [], [], []

    # pbar = (
    #     tqdm(range(conf.TRAIN.START_EPOCH, conf.TRAIN.EPOCHS), dynamic_ncols=True)
    #     if rank == conf.DATA.MAIN_RANK
    #     else range(conf.TRAIN.START_EPOCH, conf.TRAIN.EPOCHS)
    # )
    logger.info(f"{conf.MODEL.NAME}: {conf.PATH.CURRENT_CONFIG}")
    logger.info("Start Training")
    for epoch in range(conf.TRAIN.START_EPOCH, conf.TRAIN.EPOCHS):
        train_sampler.set_epoch(epoch)
        train_loss, train_iou, train_dice = train_one_epoch(
            conf=conf,
            device=device,
            loss_scaler=loss_scaler,
            amp_enabled=conf.DATA.AMP,
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            criterion=criterion_train,
        )
        dist.barrier()
        if rank == conf.DATA.MAIN_RANK:
            test_loss, test_iou, test_dice = test(conf, device, test_dataloader, model.module, criterion_test)
            train_loss_list.append(train_loss)
            train_iou_list.append(train_iou)
            train_dice_list.append(train_dice)
            test_loss_list.append(test_loss)
            test_iou_list.append(test_iou)
            test_dice_list.append(test_dice)
            # pbar.set_postfix(
            #     epoch=f"{epoch + 1}/{conf.epochs}",
            #     train_loss=f"{train_loss:.4f}",
            #     train_dice=f"{train_dice:.4f}",
            #     test_loss=f"{test_loss:.4f}",
            #     test_dice=f"{test_dice:.4f}",
            # )
            logger.info(
                f"train_loss={train_loss:.4f},test_loss={test_loss:.4f},"
                f"train_iou={train_iou:.4f},test_iou={test_iou:.4f},"
                f"train_dice={train_dice:.4f},test_dice={test_dice:.4f}"
            )
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
            pickle.dump(test_loss_list, f)
            pickle.dump(train_iou_list, f)
            pickle.dump(test_iou_list, f)
            pickle.dump(train_dice_list, f)
            pickle.dump(test_dice_list, f)
    delete_ddp()


if __name__ == "__main__":
    conf = load_config()
    if conf.TRAIN_MODE:
        main(conf)
    if conf.PROCESS_MODE:
        data_process(conf)
