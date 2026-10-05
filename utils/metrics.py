import torch


def dice_score(num_classes, pred, mask, include_background=False, eps=1e-6):
    """
    macro_dice
    args:
        num_classes: int (equal to C)
        pred: [B,H,W] or [B,C,H,W]
        mask: [B,H,W] or [B,1,H,W]
        include_background: bool (default False)
        eps: float
    return:
        [B,C] or [B,C-1](bg is False)
    """
    if pred.ndim == 4:
        pred = torch.argmax(pred, dim=1)
    if mask.ndim == 4:
        mask = torch.squeeze(mask, dim=1)

    start_index = 1 - int(include_background)
    dice = torch.zeros(pred.size(0), num_classes, device=pred.device)
    for i in range(start_index, num_classes):
        pred_fg = (pred == i).float()
        mask_fg = (mask == i).float()

        intersection = (pred_fg * mask_fg).sum(dim=(1, 2))
        sum_union = pred_fg.sum(dim=(1, 2)) + mask_fg.sum(dim=(1, 2))

        dice[:, i] = (2 * intersection + eps) / (sum_union + eps)

    if not include_background:
        dice = dice[:, 1:]
    return dice


def dice_sum(num_classes, pred, mask, include_background=False, eps=1e-6):
    """
    macro_dice_mean
    args:
        num_classes: int (equal to C)
        pred: [B,H,W] or [B,C,H,W]
        mask: [B,H,W] or [B,1,H,W]
        include_background: bool (default False)
        eps: float
    return:
        class_dim mean and batch_dim sum
    """
    dice_sum_ = dice_score(num_classes, pred, mask, include_background, eps)
    dice_sum_ = dice_sum_.mean(dim=1).sum(dim=0)
    return dice_sum_


def dice_mean(num_classes, pred, mask, include_background=False, eps=1e-6):
    """
    macro_dice_mean
    args:
        num_classes: int (equal to C)
        pred: [B,H,W] or [B,C,H,W]
        mask: [B,H,W] or [B,1,H,W]
        include_background: bool (default False)
        eps: float
    return:
        class_dim mean and batch_dim mean
    """
    dice_mean_ = dice_score(num_classes, pred, mask, include_background, eps)
    dice_mean_ = dice_mean_.mean()
    return dice_mean_


def iou_score(num_classes, pred, mask, include_background=False, eps=1e-6):
    """
    macro_iou
    args:
        num_classes: int (equal to C)
        pred: [B,H,W] or [B,C,H,W]
        mask: [B,H,W] or [B,1,H,W]
        include_background: bool (default False)
        eps: float
    return:
        [B,C] or [B,C-1](bg is False)
    """
    if pred.ndim == 4:
        pred = torch.argmax(pred, dim=1)
    if mask.ndim == 4:
        mask = torch.squeeze(mask, dim=1)

    start_index = 1 - int(include_background)
    iou = torch.zeros(pred.size(0), num_classes, device=pred.device)
    for i in range(start_index, num_classes):
        pred_fg = (pred == i).float()
        mask_fg = (mask == i).float()

        intersection = (pred_fg * mask_fg).sum(dim=(1, 2))
        union = pred_fg.sum(dim=(1, 2)) + mask_fg.sum(dim=(1, 2)) - intersection

        iou[:, i] = (intersection + eps) / (union + eps)
    if not include_background:
        iou = iou[:, 1:]
    return iou


def iou_sum(num_classes, pred, mask, include_background=False, eps=1e-6):
    """
    macro_iou_mean
    args:
        num_classes: int (equal to C)
        pred: [B,H,W] or [B,C,H,W]
        mask: [B,H,W] or [B,1,H,W]
        include_background: bool (default False)
        eps: float
    return:
        class_dim mean and batch_dim sum
    """
    iou_sum_ = iou_score(num_classes, pred, mask, include_background, eps)
    iou_sum_ = iou_sum_.mean(dim=1).sum(dim=0)
    return iou_sum_


def iou_mean(num_classes, pred, mask, include_background=False, eps=1e-6):
    """
    macro_iou_mean
    args:
        num_classes: int (equal to C)
        pred: [B,H,W] or [B,C,H,W]
        mask: [B,H,W] or [B,1,H,W]
        include_background: bool (default False)
        eps: float
    return:
        class_dim mean and batch_dim mean
    """
    iou_mean_ = iou_score(num_classes, pred, mask, include_background, eps)
    iou_mean_ = iou_mean_.mean()
    return iou_mean_
