import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def compute_dice_accuracy(label, mask):
    smooth = 1.0
    batch = label.size(0)
    m1 = label.view(batch, -1).float()  # Flatten
    m2 = mask.view(batch, -1).float()  # Flatten
    intersection = (m1 * m2).sum(1).float()
    return (2. * intersection + smooth) / (m1.sum(1) + m2.sum(1) + smooth)


def compute_multilabel_dice_accuracy(label, mask):
    assert len(mask.shape) == 4
    dim = mask.shape[1]
    mask = torch.argmax(mask, dim = 1, keepdim=True)
    mask = torch.cat([(mask[:, 0, ...] == i).unsqueeze(dim=1) for i in range(dim)], dim = 1).float()

    smooth = 1.0
    batch = label.size(0)
    chn = label.size(1)
    m1 = label.view(batch, chn, -1).float()  # Flatten
    m2 = mask.view(batch, chn, -1).float()  # Flatten
    intersection = (m1 * m2).sum(2).float()
    return (2. * intersection + smooth) / (m1.sum(2) + m2.sum(2) + smooth)


def compute_multilabel_IoU(label, mask):
    assert len(mask.shape) == 4
    dim = mask.shape[1]
    mask = torch.argmax(mask, dim = 1, keepdim=True)
    mask = torch.cat([(mask[:, 0, ...] == i).unsqueeze(dim=1) for i in range(dim)], dim = 1).float()

    smooth = 1.0
    batch = label.size(0)
    chn = label.size(1)
    m1 = label.view(batch, chn, -1).float()  # Flatten
    m2 = mask.view(batch, chn, -1).float()  # Flatten
    union = ((m1 + m2) > 0).float()
    intersection = (m1 * m2).sum(2).float()
    return (intersection + smooth) / (union.sum(2) + smooth)


# Dice loss
def dice_loss(label, mask):
    mask = (torch.sigmoid(mask) > 0.5).float()
    return torch.mean(1.0 - compute_dice_accuracy(label, mask))


def multilabel_dice_loss(label, mask):
    return torch.mean(1.0 - compute_multilabel_dice_accuracy(label, mask))


if __name__ == '__main__':
    mask = torch.rand([1, 11, 1000, 1000]) > 0.5
    pred = torch.rand([1, 11, 1000, 1000]) > 0.5
    dice = compute_dice_accuracy(mask, pred)
    print(dice.shape)
