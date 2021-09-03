import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    :param n: int, number of class.
    :param weight: torch.Tensor, weight of each class.
    :param gamma: loss coefficient.
    :param reduction: return loss type.

    example:
        >>> y1   = torch.arange(0.1, 1, 0.2).reshape(-1,1)
        >>> y2   = torch.arange(0.9, 0, -0.2).reshape(-1,1)
        >>> pred = torch.Tensor(torch.cat([y1, y2], dim=1)).requires_grad_(True)
        >>> true = torch.Tensor([1] * 5)
        >>> focal_loss = FocalLoss(2, gamma=1, reduction='raw')
        >>> focal_loss(pred, true)
        tensor([[0.0105],
                [0.1070],
                [0.3466],
                [0.8428],
                [2.0723]], grad_fn=<MulBackward0>)
    """
    def __init__(self, n, weight=None, gamma=2, reduction='mean'):
        super().__init__()
        self.weight = torch.ones(n) if weight is None else torch.Tensor(weight)
        self.weight = self.weight.view(-1,1)
        self.gamma = gamma
        self.reduction = reduction
        assert len(self.weight) == n

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]
        # target = N * 1
        target = target.view(-1, 1).long()
        # class_mask = N * class, where class_mask[:,target] = 1 else 0
        class_mask = torch.zeros(pred.shape, device=pred.device).scatter(1, target, 1.)
        # prob = N * 1
        probs = (pred * class_mask).sum(1).view(-1, 1)
        # self.weight = class * 1
        self.weight = self.weight.to(pred.device)
        # weight = N * 1
        weight = self.weight[target.view(-1)]
        # batch_loss = (N * 1) * (N * 1) ** 1 * (N * 1) = (N * 1)
        batch_loss = -weight * (1-probs)**self.gamma * probs.log()
        if self.reduction == 'mean':
            loss = batch_loss.mean()
        elif self.reduction == 'sum':
            loss = batch_loss.sum()
        elif self.reduction == 'raw':
            loss = batch_loss
        else:
            raise ValueError('unknown reduction.')
        return loss