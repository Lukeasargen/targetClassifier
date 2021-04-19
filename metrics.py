import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class WeightedTaskLoss(nn.Module):
    """ https://arxiv.org/abs/1705.07115 """
    def __init__(self, num_tasks):
        super(WeightedTaskLoss, self).__init__()
        self.num_tasks = num_tasks
        self.sigma = nn.Parameter(torch.ones(num_tasks))
        self.mse = nn.MSELoss()
        self.cel = nn.CrossEntropyLoss()
        self.ones = nn.Parameter(torch.ones(num_tasks), requires_grad=False)

    def forward(self, output, target):
        loss = 0
        for i in range(self.num_tasks):
            y = target[i]  # get task target, a tensor of scalars
            if target.ndim != 1:  # check for multiple task target
                y = target[:, i]  # target is a tensor of indices
            if output[i].shape[1] == 1:  # output is a scalar not class logits
                loss += (0.5*self.mse(output[i].squeeze(), y)/self.sigma[i]**2.0) + torch.log(self.sigma[i])
            else:  # class cross entropy loss
                loss += self.cel(output[i] / self.sigma[i]**2.0, y.long())
        return loss


def jaccard_iou(preds, targets, smooth: float = 1.0):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    total = (preds + targets).sum()
    union = total - intersection
    return (intersection + smooth) / (union + smooth)

class JaccardLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth
 
    def forward(self, preds, targets):
        return 1.0 - jaccard_iou(preds, targets, self.smooth)


def dice_coeff(preds, targets, smooth: float = 1.0):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    unionset = (preds + targets).sum()
    return (2.0 * intersection + smooth) / (unionset + smooth)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
 
    def forward(self, preds, targets):
        return 1.0 - dice_coeff(preds, targets, self.smooth)


def tversky_measure(preds, targets, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0):
    preds = preds.view(-1)
    targets = targets.view(-1)
    tp = (preds * targets).sum()
    fp = ((1.0-targets) * preds).sum()
    fn = (targets * (1.0-preds)).sum()
    return (tp + smooth) / (tp + alpha*fp + beta*fn + smooth)

class TverskyLoss(nn.Module):
    """ !!! This might need a lower learning rate to work well """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # false positive weight
        self.beta = beta  # false negative weight
        self.smooth = smooth
 
    def forward(self, preds, targets):
        return 1.0 - tversky_measure(preds, targets, self.alpha, self.beta, self.smooth)


def focal_metric(logits, targets, alpha: float, gamma: float):
    logits = logits.view(-1)
    targets = targets.view(-1)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    bce_exp = torch.exp(-bce)
    return alpha * bce * (1.0-bce_exp)**gamma

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
 
    def forward(self, logits, targets):
        return focal_metric(logits, targets, self.alpha, self.gamma)


def pixel_accuracy(preds, targets, threshold: float = 0.5):
    accsum = 0.0
    preds = preds > threshold
    correct = (preds == targets).sum()
    total = targets.shape[0]*targets.shape[2]*targets.shape[3]
    return correct/total

