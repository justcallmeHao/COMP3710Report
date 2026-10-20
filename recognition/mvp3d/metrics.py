import torch
from torch.nn import functional as F


def segmentation_loss(logits, target):
    if logits.shape != (target.shape[0], 6, *target.shape[1:]):
        raise ValueError("Logits and target spatial dimensions disagree.")
    logits = logits.float()
    probabilities = logits.softmax(1)
    truth = F.one_hot(target, num_classes=6).movedim(-1, 1).float()
    axes = (2, 3, 4)
    intersection = (probabilities * truth).sum(axes)
    denominator = probabilities.sum(axes) + truth.sum(axes)
    # All five foreground channels contribute, including absent target classes.
    soft_dice = (2 * intersection + 1e-5) / (denominator + 1e-5)
    dice_loss = 1 - soft_dice[:, 1:].mean()
    ce = F.cross_entropy(logits, target)
    return 0.5 * dice_loss + 0.5 * ce, dice_loss, ce


def hard_metrics(prediction, target):
    result = {}
    foreground = []
    for c in range(6):
        pred, truth = prediction == c, target == c
        p, t = int(pred.sum()), int(truth.sum())
        intersection = int((pred & truth).sum())
        dice = 2 * intersection / (p + t) if p + t else None
        union = p + t - intersection
        result[f"class_{c}_dice"] = dice
        result[f"class_{c}_iou"] = intersection / union if union else None
        result[f"class_{c}_pred_voxels"] = p
        result[f"class_{c}_target_voxels"] = t
        if c and dice is not None:
            foreground.append(dice)
    result["foreground_macro_dice"] = (
        sum(foreground) / len(foreground) if foreground else None)
    result["all_background_prediction"] = bool((prediction == 0).all())
    return result
