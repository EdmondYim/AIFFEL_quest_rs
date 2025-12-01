import numpy as np
import torch
from skimage.morphology import binary_dilation, disk


def compute_iou(pred, target, num_classes=2, ignore_index=-100):
    """Mean IoU over classes, optionally ignoring a label."""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    ious = []
    class_iou = {}

    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        pred_mask = pred == cls
        target_mask = target == cls

        if ignore_index is not None:
            valid = target != ignore_index
            pred_mask = np.logical_and(pred_mask, valid)
            target_mask = np.logical_and(target_mask, valid)

        if target_mask.sum() == 0:
            continue

        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()

        iou = intersection / (union + 1e-7)
        ious.append(iou)
        class_iou[cls] = iou

    miou = np.mean(ious) if ious else 0.0
    return miou, class_iou


def mask_to_boundary(mask, radius=1):
    """Extract binary boundary via dilation xor mask."""
    mask = mask.astype(bool)
    return binary_dilation(mask, disk(radius)) ^ mask


def compute_boundary_iou(pred, target, num_classes=2, ignore_index=-100):
    """Boundary IoU using dilation/erosion-style binary boundaries."""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    ious = []
    class_iou = {}
    batch_size = pred.shape[0]

    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        batch_ious = []
        for b in range(batch_size):
            pred_mask = pred[b] == cls
            target_mask = target[b] == cls

            if ignore_index is not None:
                valid = target[b] != ignore_index
                pred_mask = np.logical_and(pred_mask, valid)
                target_mask = np.logical_and(target_mask, valid)

            if target_mask.sum() == 0:
                continue

            boundary_pred = mask_to_boundary(pred_mask)
            boundary_gt = mask_to_boundary(target_mask)

            intersection = np.logical_and(boundary_pred, boundary_gt).sum()
            union = np.logical_or(boundary_pred, boundary_gt).sum()
            if union == 0:
                continue

            batch_ious.append(intersection / (union + 1e-7))

        if batch_ious:
            cls_iou = np.mean(batch_ious)
            ious.append(cls_iou)
            class_iou[cls] = cls_iou

    miou = np.mean(ious) if ious else 0.0
    return miou, class_iou
