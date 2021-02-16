import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.05):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss


class MixCrossEntropyLoss(nn.Module):
    """
    Fmix, Cutmix, Mixup calc loss
        forward input: pred, target1, target2, lam
        return loss
    """
    def __init__(self, reduction="none"):
        super(MixCrossEntropyLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, pred, target1, target2, lam):
        return lam * self.criterion(pred, target1) + (1 - lam) * self.criterion(pred, target2)