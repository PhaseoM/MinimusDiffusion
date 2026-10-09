import torch
import torch.distributed as dist
from models.autoencoders.vae import VAE, vae_loss


def train(
    conf,
    model,
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

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            z_mean, z_logvar, x_mean, x_logvar = model(X)
            loss = vae_loss(X, z_mean, z_logvar, x_mean, x_logvar, conf.BETA)  # remember to revise

        batch_size = y.size(0)

        loss_scaler.scale(loss).backward()
        loss_scaler.step(optimizer)
        loss_scaler.update()
        scheduler.step()

        train_loss += loss.detach() * batch_size
        sample_count += batch_size

    metrics = torch.cat([train_loss, sample_count])
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    global_train_loss = (metrics[0] / metrics[-1]).item()
    return global_train_loss


@torch.no_grad()
def validate(
    conf,
    device,
    dataloader,
    model,
    amp_enabled,
):
    model.eval()
    val_loss = torch.zeros(1, device=device)
    sample_count = torch.zeros(1, device=device)

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            z_mean, z_logvar, x_mean, x_logvar = model(X)
            loss = vae_loss(X, z_mean, z_logvar, x_mean, x_logvar, conf.BETA)

        batch_size = y.size(0)
        val_loss += loss.detach() * batch_size
        sample_count += batch_size

    metrics = torch.cat([val_loss, sample_count])
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    global_val_loss = (metrics[0] / metrics[-1]).item()
    return global_val_loss


@torch.no_grad()
def test(
    conf,
    device,
    model,
    dataloader,
):
    model.eval()

    test_loss = 0
    size_n = len(dataloader.dataset)
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        z_mean, z_logvar, x_mean, x_logvar = model(X)
        loss = vae_loss(X, z_mean, z_logvar, x_mean, x_logvar, conf.BETA)

        batch_size = y.size(0)
        test_loss += loss.detach() * batch_size

    test_loss = (test_loss / size_n).item()
    return test_loss
