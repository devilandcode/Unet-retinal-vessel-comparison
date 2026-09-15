"""Training plots and qualitative prediction collages (same layout as the original scripts)."""

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_COL_TITLES = ('Original', 'Ground Truth', 'Prediction', 'Overlay')


def plot_training_history(history: Dict, save_path: Optional[str] = None, title: str = 'Training History'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue', alpha=0.7)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red', alpha=0.7)
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(history['train_iou'], label='Train IoU', color='green', alpha=0.7)
    axes[0, 1].plot(history['val_iou'], label='Val IoU', color='orange', alpha=0.7)
    axes[0, 1].set_title('IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(history['train_dice'], label='Train Dice', color='purple', alpha=0.7)
    axes[1, 0].plot(history['val_dice'], label='Val Dice', color='brown', alpha=0.7)
    axes[1, 0].set_title('Dice')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(history['lr'], label='LR', color='red', alpha=0.7)
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _minmax01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin < 1e-8:
        return np.clip(arr, 0.0, 1.0)
    return (arr - vmin) / (vmax - vmin)


def tensor_to_display_rgb(image) -> np.ndarray:
    """Undo per-image z-score enough to show a recognizable (CLAHE) fundus."""
    arr = image.detach().cpu().float().numpy() if hasattr(image, 'detach') else np.asarray(image, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    arr = _minmax01(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.clip(arr, 0.0, 1.0)


def numpy_to_display_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.dtype == np.uint8 or float(np.nanmax(arr)) > 1.5:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
    return np.clip(arr, 0.0, 1.0)


def as_binary_hw(mask, threshold: float = 0.5) -> np.ndarray:
    arr = mask.detach().cpu().numpy() if hasattr(mask, 'detach') else np.asarray(mask)
    arr = np.squeeze(arr)
    return (arr > threshold).astype(np.float32)


def green_overlay(image_vis: np.ndarray, pred_binary: np.ndarray) -> np.ndarray:
    """Bright lime/neon green vessels on the original RGB (old u-net.png style)."""
    overlay = np.clip(np.asarray(image_vis, dtype=np.float32).copy(), 0.0, 1.0)
    if overlay.ndim == 2:
        overlay = np.repeat(overlay[:, :, None], 3, axis=2)
    pred = np.squeeze(np.asarray(pred_binary, dtype=np.float32))
    overlay[:, :, 1] = np.maximum(overlay[:, :, 1], pred)
    return overlay


def _save_collage(
    rows: Sequence[tuple],
    save_path: Union[str, Path],
    title: str = 'Model Predictions Visualization',
):
    n = len(rows)
    if n == 0:
        raise ValueError('No samples to visualize')
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = np.asarray(axes).reshape(1, -1)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    for j, col in enumerate(_COL_TITLES):
        axes[0, j].set_title(col, fontsize=12, fontweight='bold')
    for i, (orig, gt, pred, overlay) in enumerate(rows):
        axes[i, 0].imshow(orig)
        axes[i, 0].axis('off')
        axes[i, 1].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].axis('off')
        axes[i, 2].imshow(pred, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].axis('off')
        axes[i, 3].imshow(overlay)
        axes[i, 3].axis('off')
    plt.tight_layout()
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_prediction_collage(
    model,
    dataloader,
    device,
    save_path: Union[str, Path],
    n_samples: int = 4,
    threshold: float = 0.5,
    title: str = 'Model Predictions Visualization',
):
    """Val-set collage: Original | Ground Truth | Prediction | Overlay (neon green)."""
    import torch

    model.eval()
    images_cpu, masks_cpu, preds_cpu = [], [], []
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            preds = model(images)
            for b in range(images.size(0)):
                images_cpu.append(images[b].detach().cpu())
                masks_cpu.append(masks[b].detach().cpu())
                preds_cpu.append(preds[b].detach().cpu())
                if len(images_cpu) >= n_samples:
                    break
            if len(images_cpu) >= n_samples:
                break
    rows = []
    for image, mask, pred in zip(images_cpu, masks_cpu, preds_cpu):
        orig = tensor_to_display_rgb(image)
        gt = as_binary_hw(mask, threshold=0.5)
        pred_bin = as_binary_hw(pred, threshold=threshold)
        rows.append((orig, gt, pred_bin, green_overlay(orig, pred_bin)))
    _save_collage(rows, save_path, title=title)
    return Path(save_path)


def save_numpy_prediction_collage(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    preds: Sequence[np.ndarray],
    save_path: Union[str, Path],
    threshold: float = 0.5,
    title: str = 'Model Predictions Visualization',
):
    """Same collage from numpy RGB / masks (nnU-Net eval)."""
    rows = []
    for image, mask, pred in zip(images, masks, preds):
        orig = numpy_to_display_rgb(image)
        gt = as_binary_hw(mask, threshold=0.5)
        pred_bin = as_binary_hw(pred, threshold=threshold)
        if pred_bin.shape != orig.shape[:2]:
            from PIL import Image as PILImage
            pred_bin = np.array(
                PILImage.fromarray((pred_bin * 255).astype(np.uint8)).resize(
                    (orig.shape[1], orig.shape[0]), PILImage.NEAREST,
                ),
                dtype=np.float32,
            ) / 255.0
        if gt.shape != orig.shape[:2]:
            from PIL import Image as PILImage
            gt = np.array(
                PILImage.fromarray((gt * 255).astype(np.uint8)).resize(
                    (orig.shape[1], orig.shape[0]), PILImage.NEAREST,
                ),
                dtype=np.float32,
            ) / 255.0
        rows.append((orig, gt, pred_bin, green_overlay(orig, pred_bin)))
    _save_collage(rows, save_path, title=title)
    return Path(save_path)
