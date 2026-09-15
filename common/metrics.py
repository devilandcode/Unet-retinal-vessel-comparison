"""Segmentation metrics and fold aggregation (mean, std, 95% CI)."""

from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

# t_{0.975, df=4} for n=5; used if scipy is unavailable.
_T_CRIT_DF4 = 2.7764451051977987


def confusion_counts(pred, target, threshold: float = 0.5):
    """Return (tp, fp, fn, tn) as Python floats. Accepts torch tensors or ndarrays."""
    try:
        import torch
        if isinstance(pred, torch.Tensor):
            pred_b = (pred > threshold).float()
            tgt = target.float()
            tp = (pred_b * tgt).sum().item()
            fp = (pred_b * (1.0 - tgt)).sum().item()
            fn = ((1.0 - pred_b) * tgt).sum().item()
            tn = ((1.0 - pred_b) * (1.0 - tgt)).sum().item()
            return tp, fp, fn, tn
    except ImportError:
        pass

    pred_arr = np.asarray(pred)
    tgt_arr = np.asarray(target).astype(np.float64)
    pred_b = (pred_arr > threshold).astype(np.float64)
    tp = float((pred_b * tgt_arr).sum())
    fp = float((pred_b * (1.0 - tgt_arr)).sum())
    fn = float(((1.0 - pred_b) * tgt_arr).sum())
    tn = float(((1.0 - pred_b) * (1.0 - tgt_arr)).sum())
    return tp, fp, fn, tn


def metrics_from_counts(tp: float, fp: float, fn: float, tn: float) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def calculate_metrics(pred, target, threshold: float = 0.5) -> Dict[str, float]:
    """Per-batch metrics (used for training logs)."""
    tp, fp, fn, tn = confusion_counts(pred, target, threshold)
    return metrics_from_counts(tp, fp, fn, tn)


def mean_std_ci(values: Sequence[float], confidence: float = 0.95) -> Dict[str, float]:
    """Mean, sample std, and two-sided t-interval. n=1 → std=0, CI=mean."""
    arr = np.asarray(list(values), dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {'mean': 0.0, 'std': 0.0, 'ci95_low': 0.0, 'ci95_high': 0.0, 'n': 0}
    mean = float(arr.mean())
    if n == 1:
        return {'mean': mean, 'std': 0.0, 'ci95_low': mean, 'ci95_high': mean, 'n': 1}
    std = float(arr.std(ddof=1))
    se = std / math.sqrt(n)
    try:
        from scipy import stats
        tcrit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    except Exception:
        tcrit = _T_CRIT_DF4 if n == 5 else 1.96
    half = tcrit * se
    return {
        'mean': mean,
        'std': std,
        'ci95_low': mean - half,
        'ci95_high': mean + half,
        'n': n,
    }


def aggregate_fold_metrics(
    per_fold: Sequence[Mapping[str, float]],
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, float]]:
    if not per_fold:
        return {}
    if keys is None:
        keys = [k for k, v in per_fold[0].items() if isinstance(v, (int, float))]
    return {key: mean_std_ci([float(fold[key]) for fold in per_fold]) for key in keys}


def measure_inference_speed(model, dataloader, device, num_samples: int = 20) -> Dict[str, float]:
    import torch
    model.eval()
    times: List[float] = []
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_samples:
                break
            images = images.to(device)
            if i == 0 and getattr(device, 'type', None) == 'cuda':
                for _ in range(3):
                    _ = model(images)
                torch.cuda.synchronize()
            start = time.time()
            _ = model(images)
            if getattr(device, 'type', None) == 'cuda':
                torch.cuda.synchronize()
            times.append((time.time() - start) / max(int(images.size(0)), 1))
    usable = times[1:] if len(times) > 1 else times
    if not usable:
        return {'avg_time_per_image': 0.0, 'std_time': 0.0, 'fps': 0.0, 'total_samples': 0}
    avg_time = float(np.mean(usable))
    std_time = float(np.std(usable))
    fps = 1.0 / avg_time if avg_time > 0 else 0.0
    return {
        'avg_time_per_image': avg_time,
        'std_time': std_time,
        'fps': fps,
        'total_samples': len(usable),
    }
