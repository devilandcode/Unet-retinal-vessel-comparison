"""Loss functions shared by U-Net, U-Net++ and Attention U-Net."""

import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class VesselFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=3.0, vessel_weight=2.0, smooth=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.vessel_weight = vessel_weight
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.clamp(pred, self.smooth, 1.0 - self.smooth)
        ce_loss = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        vessel_boost = 1 + (self.vessel_weight - 1) * target
        return (alpha_t * focal_weight * ce_loss * vessel_boost).mean()


_PRED_EPS = 1e-7


def _sanitize_pred(pred, eps=_PRED_EPS):
    """Keep probabilities in (eps, 1-eps) so BCELoss does not assert on CUDA."""
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    return torch.clamp(pred, min=eps, max=1.0 - eps)


class CombinedLoss(nn.Module):
    """0.4 Tversky + 0.4 VesselFocal + 0.2 BCE — identical for all three U-Nets."""

    def __init__(self):
        super().__init__()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        self.vessel_focal = VesselFocalLoss(alpha=0.25, gamma=3.0, vessel_weight=2.0)
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        pred = _sanitize_pred(pred)
        return (0.4 * self.tversky(pred, target)
                + 0.4 * self.vessel_focal(pred, target)
                + 0.2 * self.bce(pred, target))
