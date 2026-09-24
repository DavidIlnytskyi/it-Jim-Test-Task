"""Training and validation loops."""


import torch

from .dataclass import TrainOutput
from .utils import save_model


def train_loop(model, criterion, optimizer, data, threshold=0.5, device=0):
    model.train()
    model.to(device)

    epoch_loss = 0.0
    num_samples = 0
    for x_true, y_true in data:
        x_true = x_true.to(dtype=torch.float, device=device)
        y_true = y_true.to(dtype=torch.float, device=device).unsqueeze(1)

        optimizer.zero_grad()

        y_pred = model(x_true).float()

        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(y_true)
        num_samples += len(y_true)

    return epoch_loss / num_samples


def valid_loop(
    model,
    criterion,
    metric_func,
    data,
    threshold=0.5,
    device=0,
):
    model.eval()
    model.to(device)

    val_predictions = []
    val_targets = []
    valid_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for val_x_true, val_y_true in data:
            val_x_true = val_x_true.to(dtype=torch.float, device=device)
            val_y_true = val_y_true.to(device).float().unsqueeze(1)

            val_y_pred = model(val_x_true)

            valid_loss += criterion(
                val_y_pred,
                val_y_true,
            ).item() * len(val_y_pred)
            num_samples += len(val_y_pred)

            val_predictions.append(val_y_pred)
            val_targets.append(val_y_true)

    val_predictions = torch.cat(val_predictions)
    val_targets = torch.cat(val_targets)
    valid_loss /= num_samples

    val_predictions = (
        torch.sigmoid(val_predictions) >= threshold
    ).int()

    val_targets = val_targets.int()

    metric = metric_func(
        val_predictions.flatten(),
        val_targets.flatten(),
    ).item()

    return metric, valid_loss


def train_model(cfg):
    train_losses = []
    valid_losses = []
    metric_values = []
    min_valid_loss = float("inf")
    print(
        f"{'Epoch id':^10}"
        f"{'Train Loss':^15}"
        f"{'Valid Loss':^15}"
        f"{'Metric':^12}"
    )
    for epoch_id in range(1, cfg.epoch_size + 1):
        epoch_loss = train_loop(
            model=cfg.model,
            criterion=cfg.criterion,
            optimizer=cfg.optimizer,
            data=cfg.train_dataloader,
            device=cfg.device,
            threshold=cfg.threshold,
        )
        train_losses.append(epoch_loss)
        if epoch_id % cfg.epoch_period == 0:
            micro_f1, valid_loss = valid_loop(
                model=cfg.model,
                criterion=cfg.criterion,
                data=cfg.valid_dataloader,
                metric_func=cfg.metric_func,
                threshold=cfg.threshold,
                device=cfg.device,
            )

            print(
                f"{epoch_id:^10}"
                f"{epoch_loss:^15.3f}"
                f"{valid_loss:^15.3f}"
                f"{micro_f1:^12.3f}"
            )
            metric_values.append(micro_f1)
            valid_losses.append(valid_loss)

        cfg.scheduler.step()

        for cb in cfg.callbacks:
          cb(cfg=cfg, valid_loss=valid_loss, min_valid_loss=min_valid_loss)


    output = TrainOutput(train_losses, valid_losses, metric_values)
    save_model(cfg, "last.pt")

    return output
