import torch
import torch.nn as nn


def jaccard_iou(preds, targets, smooth=1):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    total = (preds + targets).sum()
    union = total - intersection
    return (intersection + smooth) / (union + smooth)

class JaccardLoss(nn.Module):
    def __init__(self, smooth=1):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth
 
    def forward(self, preds, targets):
        return 1 - jaccard_iou(preds, targets, self.smooth)


def dice_coeff(preds, targets, smooth=1):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    unionset = preds.sum() + targets.sum()
    return (2 * intersection + smooth) / (unionset + smooth)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
 
    def forward(self, preds, targets):
        return 1 - dice_coeff(preds, targets, self.smooth)


def tversky_measure(preds, targets, alpha=0.5, beta=0.5, smooth=1):
    preds = preds.view(-1)
    targets = targets.view(-1)
    tp = (preds * targets).sum()
    fp = ((1-targets) * preds).sum()
    fn = (targets * (1-preds)).sum()
    return (tp + smooth) / (tp + alpha*fp + beta*fn + smooth)

class TverskyLoss(nn.Module):
    """ !!! This might need a lower learning rate to work well """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # false positive weight
        self.beta = beta  # false negative weight
        self.smooth = smooth
 
    def forward(self, preds, targets):
        return 1- tversky_measure(preds, targets, self.alpha, self.beta, self.smooth)


def focal_metric(preds, targets, alpha, gamma):
    preds = preds.view(-1)
    targets = targets.view(-1)
    bce = nn.BCELoss(reduction='mean')(preds, targets)
    bce_exp = torch.exp(-bce)
    return alpha * bce * (1-bce_exp)**gamma

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
 
    def forward(self, preds, targets):
        return focal_metric(preds, targets, self.alpha, self.gamma)


def pixel_accuracy(preds, targets, threshold=0.5):
    accsum = 0
    preds = preds > threshold
    correct = (preds == targets).sum()
    total = targets.shape[0]*targets.shape[2]*targets.shape[3]
    return correct/total

